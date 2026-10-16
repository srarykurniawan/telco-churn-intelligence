# Model Card — churn_model v2

- Created: 2026-10-15T07:52:29+00:00
- Data version: `customers@v1`
- Git commit: `789bd41`
- Description: PyTorch MLP churn classifier (90-day churn)

## Metrics

| metric | value |
|---|---|
| test_roc_auc | 0.7542 |
| test_pr_auc | 0.4731 |
| test_f1 | 0.4418 |
| test_precision | 0.4721 |
| test_recall | 0.4151 |
| test_accuracy | 0.7915 |
| test_brier | 0.1360 |
| test_log_loss | 0.4295 |
| test_ks | 0.3733 |
| test_threshold | 0.4000 |
| test_tp | 330.0000 |
| test_fp | 369.0000 |
| test_tn | 2836.0000 |
| test_fn | 465.0000 |
| baseline_logreg_roc_auc | 0.7630 |
| baseline_logreg_pr_auc | 0.4837 |
| baseline_logreg_f1 | 0.3346 |
| baseline_logreg_brier | 0.1330 |
| val_roc_auc | 0.7566 |

## Parameters

- **hidden_dims**: [128, 64, 32]
- **dropout**: 0.25
- **lr**: 0.001
- **weight_decay**: 1e-05
- **batch_size**: 256
- **max_epochs**: 60
- **patience**: 8
- **use_class_weight**: False
- **threshold**: 0.39999999999999997
- **n_features_encoded**: 45
- **framework**: pytorch
