# Synthetic Banking Fraud Detection

A reproducible machine-learning project that generates synthetic bank
transactions, compares classifiers and imbalance treatments, and evaluates
fraud detection with metrics suited to rare positive events.

> **Synthetic data only:** the generator is for experimentation and education.
> Its patterns and results do not represent real customers, a production bank,
> or a validated fraud-control system.

## Quick start

```bash
python -m pip install -r requirements.txt
python fraud_detection.py
```

By default, the script generates 50,000 transactions with an expected fraud
rate of 1.5%. It saves:

- `outputs/synthetic_transactions.csv` — generated data
- `outputs/model_comparison.csv` — held-out test metrics
- `outputs/precision_recall_curves.png` — test-set precision-recall curves

Customize the run:

```bash
python fraud_detection.py --transactions 100000 --fraud-rate 0.01 --output-dir outputs
```

The comparison needs at least 50 fraud cases (about 10 each in the validation
and test partitions) and stops with an error otherwise.

Run the tests with `pytest` from the project root.

## Synthetic transaction data

The generator produces transaction amount and customer-average amount,
amount-to-average ratio, hour and weekend flag, card-not-present and
international flags, distance from home, recent transaction velocity,
merchant and device risk scores, account age, failed authentication count,
and the binary `is_fraud` label. Fraud probabilities depend on combinations
of these risk signals; a calibrated intercept targets the requested *expected*
fraud rate. The realized rate varies due to random sampling.

`transaction_id`, raw `amount`, `customer_average_amount`, and
`true_fraud_probability` (the probability each label was drawn from) are
retained in the CSV for inspection. The models use the engineered `FEATURES`
list in `fraud_detection.py`, so identifiers, the true probability, and the
target are never model inputs.

## Modeling and imbalance strategy

The project compares six configurations:

| Model | Imbalance treatment |
|---|---|
| Logistic regression | None (baseline) |
| Logistic regression | Balanced class weights |
| Random forest | Balanced subsample class weights |
| Random forest | Random minority-class oversampling |
| Histogram gradient boosting | None |
| Histogram gradient boosting | Random minority-class oversampling |

Oversampling is performed on training data only. The data is split into
stratified train, validation, and test partitions. Each model's decision
threshold is selected by maximizing F2 on the validation partition, which
weights fraud recall more heavily than precision. The test partition is used
only for final evaluation. Gradient boosting runs a fixed number of
iterations: its built-in early stopping would hold out copies of training
fraud rows once the data is oversampled.

Two control rows bracket the models in the comparison table:

- **Best possible (true fraud probability)** — positive control. Ranking by the
  generator's own probabilities gives the highest score any model can expect;
  a model that beats it suggests leakage.
- **No skill (random scores)** — negative control. Random scores give average
  precision near the fraud rate; the precision-recall plot draws this level
  as a dotted line.

Average precision (area under the precision-recall curve) is the primary
ranking metric for imbalanced data. ROC-AUC is included for context; precision,
recall, F1, F2, and confusion-matrix counts are reported at the validation-
selected threshold. Thresholds should ultimately reflect operational review
capacity and the relative costs of missed fraud and false alarms.

## Limitations

This is a randomized row-level demonstration, not a production evaluation.
It does not model customers as groups, time-based drift, delayed labels,
transaction authorization latency, investigation capacity, or monetary loss.
Before real-world use, evaluate with privacy-approved data, time-aware and
customer-aware splits, leakage checks, calibration, cost-based thresholds,
monitoring, and appropriate legal, security, and model-risk review.
