# Model Card — churn_model v1

- Created: 2026-10-13T12:42:55+00:00
- Data version: `customers@v1`
- Git commit: `None`
- Description: PyTorch MLP churn classifier (90-day churn)

## Metrics

| metric | value |
|---|---|
| test_roc_auc | 0.7549 |
| test_pr_auc | 0.4741 |
| test_f1 | 0.4499 |
| test_precision | 0.4657 |
| test_recall | 0.4352 |
| test_accuracy | 0.7885 |
| test_brier | 0.1360 |
| test_log_loss | 0.4291 |
| test_ks | 0.3745 |
| test_threshold | 0.3900 |
| test_tp | 346.0000 |
| test_fp | 397.0000 |
| test_tn | 2808.0000 |
| test_fn | 449.0000 |
| baseline_logreg_roc_auc | 0.7630 |
| baseline_logreg_pr_auc | 0.4837 |
| baseline_logreg_f1 | 0.3346 |
| baseline_logreg_brier | 0.1330 |
| val_roc_auc | 0.7555 |

## Parameters

- **hidden_dims**: [128, 64, 32]
- **dropout**: 0.25
- **lr**: 0.001
- **weight_decay**: 1e-05
- **batch_size**: 256
- **max_epochs**: 60
- **patience**: 8
- **use_class_weight**: False
- **threshold**: 0.38999999999999996
- **n_features_encoded**: 45
- **framework**: pytorch
