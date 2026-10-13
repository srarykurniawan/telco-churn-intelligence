# Model Card — arpu_regressor v1

- Created: 2026-10-13T12:43:22+00:00
- Data version: `customers@v1`
- Git commit: `None`
- Description: PyTorch MLP regressor for monthly ARPU

## Metrics

| metric | value |
|---|---|
| test_rmse | 6810.3049 |
| test_mae | 4223.5577 |
| test_mape | 4.8874 |
| test_r2 | 0.9920 |
| baseline_linreg_rmse | 30468.6341 |
| baseline_linreg_mae | 14963.1825 |
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
