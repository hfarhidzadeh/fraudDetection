import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import fbeta_score

import fraud_detection
from fraud_detection import (
    BEST_POSSIBLE,
    FEATURES,
    MIN_FRAUD_CASES,
    NO_SKILL,
    TARGET,
    TRUE_PROBABILITY,
    _random_oversample,
    _save_precision_recall_plot,
    _select_f2_threshold,
    compare_models,
    generate_synthetic_transactions,
)


@pytest.fixture(scope="module")
def transactions():
    # A high fraud rate gives ~400 test frauds while keeping the run fast; with
    # fewer, chance AP/ROC swings overlap the gap between real and no-skill scores.
    return generate_synthetic_transactions(10_000, fraud_rate=0.2, random_state=5)


@pytest.fixture(scope="module")
def comparison(transactions):
    return compare_models(transactions)


def _test_fraud_rate(results):
    row = results.iloc[0]
    frauds = row["true_positives"] + row["false_negatives"]
    return frauds / (frauds + row["false_positives"] + row["true_negatives"])


def test_synthetic_data_is_reproducible_and_imbalanced():
    first = generate_synthetic_transactions(2_000, random_state=7)
    second = generate_synthetic_transactions(2_000, random_state=7)

    assert first.equals(second)
    assert set(first[TARGET].unique()) == {0, 1}
    assert 0.005 < first[TARGET].mean() < 0.05
    assert set(FEATURES).issubset(first.columns)


def test_true_probability_is_calibrated_and_never_a_model_input():
    data = generate_synthetic_transactions(2_000, fraud_rate=0.02, random_state=7)

    assert data[TRUE_PROBABILITY].mean() == pytest.approx(0.02, abs=1e-6)
    assert data[TRUE_PROBABILITY].between(0, 1).all()
    assert TRUE_PROBABILITY not in FEATURES
    assert TARGET not in FEATURES


def test_generator_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        generate_synthetic_transactions(99)
    with pytest.raises(ValueError):
        generate_synthetic_transactions(1_000, fraud_rate=0)
    with pytest.raises(ValueError):
        generate_synthetic_transactions(1_000, fraud_rate=0.5)


def test_random_oversampling_keeps_training_rows_and_adds_only_fraud_copies():
    # A shuffled index mimics a training split, so row selection must be positional.
    data = generate_synthetic_transactions(500, random_state=11).sample(
        frac=1, random_state=0
    )
    features, labels = data[FEATURES], data[TARGET]
    assert labels.nunique() == 2

    sampled_features, sampled_labels = _random_oversample(features, labels, 11)

    n_original = len(labels)
    n_legitimate = int((labels == 0).sum())
    assert sampled_labels.value_counts().to_dict() == {0: n_legitimate, 1: n_legitimate}
    pd.testing.assert_frame_equal(
        sampled_features.iloc[:n_original], features.reset_index(drop=True)
    )
    pd.testing.assert_series_equal(
        sampled_labels.iloc[:n_original], labels.reset_index(drop=True)
    )
    assert (sampled_labels.iloc[n_original:] == 1).all()
    fraud_rows = set(map(tuple, features[labels == 1].to_numpy()))
    added_rows = sampled_features.iloc[n_original:].to_numpy()
    assert all(tuple(row) in fraud_rows for row in added_rows)


def test_threshold_selection_finds_perfect_separation():
    threshold = _select_f2_threshold(
        np.array([0, 0, 0, 1, 1]),
        np.array([0.01, 0.1, 0.25, 0.6, 0.9]),
    )

    assert threshold == 0.6


def test_threshold_selection_maximizes_f2():
    rng = np.random.default_rng(0)
    labels = rng.binomial(1, 0.2, size=300)
    scores = 0.3 * labels + rng.random(300)

    threshold = _select_f2_threshold(labels, scores)

    def f2_at(cutoff):
        return fbeta_score(labels, (scores >= cutoff).astype(int), beta=2)

    assert f2_at(threshold) == pytest.approx(max(map(f2_at, np.unique(scores))))


def test_boosting_models_do_not_early_stop_on_oversampled_copies():
    boosting = [
        model
        for _, model, _ in fraud_detection._models()
        if isinstance(model, HistGradientBoostingClassifier)
    ]

    assert boosting
    assert all(model.early_stopping is False for model in boosting)


def test_compare_models_rejects_too_few_fraud_cases():
    data = generate_synthetic_transactions(300, random_state=7)
    assert data[TARGET].sum() < MIN_FRAUD_CASES

    with pytest.raises(ValueError, match="fraud cases"):
        compare_models(data)


def test_compare_models_reports_every_model_and_both_controls(comparison):
    results, curves = comparison
    model_names = [name for name, _, _ in fraud_detection._models()]

    assert set(results["model"]) == {*model_names, BEST_POSSIBLE, NO_SKILL}
    assert set(curves) == {*model_names, BEST_POSSIBLE}
    counts = results[
        ["true_positives", "false_positives", "false_negatives", "true_negatives"]
    ]
    # Every row is scored on the same test partition.
    assert counts.sum(axis=1).nunique() == 1
    assert (results["true_positives"] + results["false_negatives"]).nunique() == 1


def test_controls_bracket_the_models(comparison):
    results, _ = comparison
    fraud_rate = _test_fraud_rate(results)
    ap = results.set_index("model")["average_precision"]
    models = ap.drop([BEST_POSSIBLE, NO_SKILL])

    # Negative control: random scores sit near the fraud rate; real models clear it.
    assert ap[NO_SKILL] < 1.5 * fraud_rate
    assert (models > 1.5 * fraud_rate).all()
    # Positive control: a model beating the true probability would signal leakage.
    assert (models < ap[BEST_POSSIBLE]).all()


def test_models_find_no_signal_in_shuffled_labels(transactions):
    shuffled = transactions.copy()
    shuffled[TARGET] = np.random.default_rng(0).permutation(shuffled[TARGET].to_numpy())

    results, _ = compare_models(shuffled)

    # ~400 test frauds give a null ROC-AUC SD of ~0.016; 0.08 is about 5 SDs.
    assert (results["average_precision"] < 1.5 * _test_fraud_rate(results)).all()
    assert ((results["roc_auc"] - 0.5).abs() < 0.08).all()


def test_best_possible_row_is_skipped_without_true_probability(
    transactions, monkeypatch
):
    logistic_only = fraud_detection._models()[:1]
    monkeypatch.setattr(fraud_detection, "_models", lambda: logistic_only)

    results, curves = compare_models(transactions.drop(columns=TRUE_PROBABILITY))

    assert set(results["model"]) == {logistic_only[0][0], NO_SKILL}
    assert BEST_POSSIBLE not in curves


def test_precision_recall_plot_is_written(comparison, tmp_path):
    results, curves = comparison
    output_path = tmp_path / "curves.png"

    _save_precision_recall_plot(curves, results, _test_fraud_rate(results), output_path)

    assert output_path.stat().st_size > 0
