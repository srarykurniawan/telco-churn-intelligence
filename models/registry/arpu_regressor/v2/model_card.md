# Model Card — arpu_regressor v2

- Created: 2026-10-15T07:54:25+00:00
- Data version: `customers@v1`
- Git commit: `789bd41`
- Description: PyTorch MLP regressor for monthly ARPU

## Metrics

| metric | value |
|---|---|
| test_rmse | 7071.6769 |
| test_mae | 4273.3971 |
| test_mape | 4.9220 |
| test_r2 | 0.9914 |
| baseline_linreg_rmse | 30468.6205 |
| baseline_linreg_mae | 14963.1685 |
| baseline_linreg_mape | 12.0026 |
| baseline_linreg_r2 | 0.8395 |

## Parameters

- **hidden_dims**: [64, 32]
- **dropout**: 0.1
- **lr**: 0.002
- **weight_decay**: 1e-05
- **batch_size**: 256
- **max_epochs**: 80
- **patience**: 10
- **target_transform**: log1p
