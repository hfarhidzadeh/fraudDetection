"""Synthetic bank transaction fraud detection and model comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TARGET = "is_fraud"
FEATURES = [
    "log_amount",
    "amount_vs_customer_average",
    "hour",
    "is_weekend",
    "card_not_present",
    "is_international",
    "distance_from_home_km",
    "transactions_last_hour",
    "merchant_risk",
    "device_risk",
    "account_age_days",
    "failed_authentications",
]
# Generator's own fraud probability: kept for the positive control, never a feature.
TRUE_PROBABILITY = "true_fraud_probability"
BEST_POSSIBLE = "Best possible (true fraud probability)"
NO_SKILL = "No skill (random scores)"
# Validation and test each receive about a fifth of the fraud cases.
MIN_FRAUD_CASES = 50


def _calibrate_intercept(risk_score: np.ndarray, fraud_rate: float) -> float:
    """Find a logistic intercept whose expected positive rate matches fraud_rate."""
    low, high = -40.0, 10.0
    for _ in range(80):
        middle = (low + high) / 2
        probabilities = 1 / (1 + np.exp(-np.clip(risk_score + middle, -40, 40)))
        if probabilities.mean() < fraud_rate:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def generate_synthetic_transactions(
    n_transactions: int = 50_000,
    fraud_rate: float = 0.015,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Create reproducible transaction data with realistic fraud-risk signals."""
    if n_transactions < 100:
        raise ValueError("n_transactions must be at least 100.")
    if not 0 < fraud_rate < 0.5:
        raise ValueError("fraud_rate must be greater than 0 and less than 0.5.")

    rng = np.random.default_rng(random_state)
    amount = rng.lognormal(mean=3.4, sigma=1.0, size=n_transactions)
    customer_average = rng.lognormal(mean=3.2, sigma=0.65, size=n_transactions)
    amount_ratio = amount / customer_average
    hour = rng.integers(0, 24, size=n_transactions)
    is_weekend = rng.binomial(1, 2 / 7, size=n_transactions)
    card_not_present = rng.binomial(1, 0.62, size=n_transactions)
    is_international = rng.binomial(1, 0.08, size=n_transactions)
    distance = rng.exponential(scale=35, size=n_transactions)
    transactions_last_hour = rng.poisson(lam=1.2, size=n_transactions)
    merchant_risk = rng.beta(a=2, b=5, size=n_transactions)
    device_risk = rng.beta(a=2, b=4, size=n_transactions)
    account_age_days = rng.integers(1, 5000, size=n_transactions)
    failed_authentications = rng.poisson(lam=0.12, size=n_transactions)

    risk_score = (
        0.85 * np.log1p(amount_ratio)
        + 1.15 * card_not_present
        + 1.35 * is_international
        + 0.012 * np.minimum(distance, 300)
        + 0.34 * transactions_last_hour
        + 2.2 * merchant_risk
        + 1.6 * device_risk
        + 0.55 * failed_authentications
        - 0.00015 * account_age_days
        + 0.3 * (hour < 5)
        + rng.normal(0, 0.45, size=n_transactions)
    )
    intercept = _calibrate_intercept(risk_score, fraud_rate)
    fraud_probability = 1 / (
        1 + np.exp(-np.clip(risk_score + intercept, -40, 40))
    )
    is_fraud = rng.binomial(1, fraud_probability)

    return pd.DataFrame(
        {
            "transaction_id": np.arange(1, n_transactions + 1),
            "amount": amount.round(2),
            "customer_average_amount": customer_average.round(2),
            "log_amount": np.log1p(amount),
            "amount_vs_customer_average": amount_ratio,
            "hour": hour,
            "is_weekend": is_weekend,
            "card_not_present": card_not_present,
            "is_international": is_international,
            "distance_from_home_km": distance.round(2),
            "transactions_last_hour": transactions_last_hour,
            "merchant_risk": merchant_risk,
            "device_risk": device_risk,
            "account_age_days": account_age_days,
            "failed_authentications": failed_authentications,
            "true_fraud_probability": fraud_probability,
            "is_fraud": is_fraud,
        }
    )


def _random_oversample(
    features: pd.DataFrame, labels: pd.Series, random_state: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Replicate minority-class training rows to balance the training set."""
    rng = np.random.default_rng(random_state)
    counts = labels.value_counts()
    majority_label = counts.idxmax()
    minority_label = counts.idxmin()
    majority_count = int(counts[majority_label])
    minority_indices = np.flatnonzero(labels.to_numpy() == minority_label)
    extra_indices = rng.choice(
        minority_indices, size=majority_count - len(minority_indices), replace=True
    )
    sampled_indices = np.concatenate(
        [np.arange(len(labels)), extra_indices]
    )
    sampled_features = features.iloc[sampled_indices].reset_index(drop=True)
    sampled_labels = labels.iloc[sampled_indices].reset_index(drop=True)
    return sampled_features, sampled_labels


def _models() -> list[tuple[str, object, bool]]:
    """Build the classifiers and imbalance-treatment variants to compare."""
    return [
        (
            "Logistic regression",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            ),
            False,
        ),
        (
            "Logistic regression (class-weighted)",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
            False,
        ),
        (
            "Random forest (class-weighted)",
            RandomForestClassifier(
                n_estimators=180,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
        (
            "Random forest (oversampled)",
            RandomForestClassifier(
                n_estimators=180,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            True,
        ),
        # Early stopping is off for both boosting models: its internal split
        # would hold copies of training fraud rows once data is oversampled.
        (
            "Histogram gradient boosting",
            HistGradientBoostingClassifier(
                max_iter=140,
                learning_rate=0.08,
                max_leaf_nodes=20,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
        (
            "Histogram gradient boosting (oversampled)",
            HistGradientBoostingClassifier(
                max_iter=140,
                learning_rate=0.08,
                max_leaf_nodes=20,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=RANDOM_STATE,
            ),
            True,
        ),
    ]


def _select_f2_threshold(labels: pd.Series, scores: np.ndarray) -> float:
    """Choose the probability threshold that maximizes F2 on validation data."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return 0.5
    f2 = (5 * precision[:-1] * recall[:-1]) / (
        4 * precision[:-1] + recall[:-1] + 1e-12
    )
    return float(thresholds[int(np.argmax(f2))])


def _evaluate(
    name: str,
    y_validation: pd.Series,
    validation_scores: np.ndarray,
    y_test: pd.Series,
    test_scores: np.ndarray,
) -> tuple[dict[str, object], tuple[np.ndarray, np.ndarray]]:
    """Score the test set at the F2 threshold chosen on validation."""
    threshold = _select_f2_threshold(y_validation, validation_scores)
    predictions = (test_scores >= threshold).astype(int)
    precision, recall, _ = precision_recall_curve(y_test, test_scores)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    row = {
        "model": name,
        "threshold_from_validation": threshold,
        "average_precision": average_precision_score(y_test, test_scores),
        "roc_auc": roc_auc_score(y_test, test_scores),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "f2": fbeta_score(y_test, predictions, beta=2, zero_division=0),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }
    return row, (recall, precision)


def compare_models(
    transactions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Train and compare classifiers using train-only fitting and validation thresholds.

    Adds a best-possible row when the true fraud probability is present and a
    no-skill row from random scores, bracketing what the models can achieve.
    """
    features = transactions[FEATURES]
    labels = transactions[TARGET]
    fraud_count = int(labels.sum())
    if fraud_count < MIN_FRAUD_CASES:
        raise ValueError(
            f"Need at least {MIN_FRAUD_CASES} fraud cases to split and evaluate; "
            f"got {fraud_count}. Increase the number of transactions or the fraud rate."
        )
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        features,
        labels,
        test_size=0.4,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_holdout,
        y_holdout,
        test_size=0.5,
        stratify=y_holdout,
        random_state=RANDOM_STATE,
    )

    rows = []
    curves = {}
    for name, model, oversample in _models():
        fit_features, fit_labels = (
            _random_oversample(x_train, y_train, RANDOM_STATE)
            if oversample
            else (x_train, y_train)
        )
        model.fit(fit_features, fit_labels)
        row, curve = _evaluate(
            name,
            y_validation,
            model.predict_proba(x_validation)[:, 1],
            y_test,
            model.predict_proba(x_test)[:, 1],
        )
        rows.append(row)
        curves[name] = curve

    # Positive control: the generator's own probabilities set the attainable ceiling.
    if TRUE_PROBABILITY in transactions:
        true_probability = transactions[TRUE_PROBABILITY]
        row, curve = _evaluate(
            BEST_POSSIBLE,
            y_validation,
            true_probability.loc[x_validation.index].to_numpy(),
            y_test,
            true_probability.loc[x_test.index].to_numpy(),
        )
        rows.append(row)
        curves[BEST_POSSIBLE] = curve

    # Negative control: uninformative scores should give AP near the fraud rate.
    rng = np.random.default_rng(RANDOM_STATE)
    row, _ = _evaluate(
        NO_SKILL,
        y_validation,
        rng.random(len(y_validation)),
        y_test,
        rng.random(len(y_test)),
    )
    rows.append(row)

    results = pd.DataFrame(rows).sort_values(
        ["average_precision", "recall"], ascending=False
    )
    return results.reset_index(drop=True), curves


def _save_precision_recall_plot(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    results: pd.DataFrame,
    fraud_rate: float,
    output_path: Path,
) -> None:
    plt.figure(figsize=(9, 6))
    for name, (recall, precision) in curves.items():
        average_precision = float(
            results.loc[results["model"] == name, "average_precision"].iloc[0]
        )
        style = {"color": "black", "linestyle": "--"} if name == BEST_POSSIBLE else {}
        plt.plot(
            recall, precision, label=f"{name} (AP={average_precision:.3f})", **style
        )
    plt.axhline(
        fraud_rate,
        color="gray",
        linestyle=":",
        label=f"No skill (fraud rate = {fraud_rate:.2%})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Fraud detection: test-set precision-recall curves")
    plt.legend(fontsize=8, loc="best")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    """Generate synthetic data, compare models, and save results and a plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", type=int, default=50_000)
    parser.add_argument("--fraud-rate", type=float, default=0.015)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = generate_synthetic_transactions(
        n_transactions=args.transactions, fraud_rate=args.fraud_rate
    )
    results, curves = compare_models(data)

    data.to_csv(args.output_dir / "synthetic_transactions.csv", index=False)
    results.to_csv(args.output_dir / "model_comparison.csv", index=False)
    _save_precision_recall_plot(
        curves,
        results,
        data[TARGET].mean(),
        args.output_dir / "precision_recall_curves.png",
    )
    print(
        f"Generated {len(data):,} transactions; "
        f"observed fraud rate: {data[TARGET].mean():.2%}"
    )
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nOutputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
