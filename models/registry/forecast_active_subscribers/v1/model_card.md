# Model Card — forecast_active_subscribers v1

- Created: 2026-10-13T12:43:34+00:00
- Data version: `monthly_kpi@v1`
- Git commit: `None`
- Description: LSTM monthly forecaster for active_subscribers

## Metrics

| metric | value |
|---|---|
| backtest_lstm_rmse | 41970.9299 |
| backtest_lstm_mae | 30825.0138 |
| backtest_lstm_mape | 0.1956 |
| backtest_seasonal_naive_rmse | 82426.6814 |
| backtest_seasonal_naive_mae | 71007.7653 |
| backtest_seasonal_naive_mape | 0.4519 |
| backtest_holt_winters_rmse | 92496.5847 |
| backtest_holt_winters_mae | 74874.3691 |
| backtest_holt_winters_mape | 0.4789 |

## Parameters

- **targets**: ['revenue_idr_bn', 'active_subscribers']
- **window**: 12
- **horizon**: 6
- **hidden_size**: 32
- **num_layers**: 1
- **epochs**: 400
- **lr**: 0.005
- **target**: active_subscribers
