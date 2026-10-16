# Model Card — forecast_revenue_idr_bn v2

- Created: 2026-10-15T07:54:43+00:00
- Data version: `monthly_kpi@v1`
- Git commit: `789bd41`
- Description: LSTM monthly forecaster for revenue_idr_bn

## Metrics

| metric | value |
|---|---|
| backtest_lstm_rmse | 12.0288 |
| backtest_lstm_mae | 11.2821 |
| backtest_lstm_mape | 1.3037 |
| backtest_seasonal_naive_rmse | 9.3837 |
| backtest_seasonal_naive_mae | 8.1535 |
| backtest_seasonal_naive_mape | 0.9451 |
| backtest_holt_winters_rmse | 13.2728 |
| backtest_holt_winters_mae | 10.7945 |
| backtest_holt_winters_mape | 1.2569 |

## Parameters

- **targets**: ['revenue_idr_bn', 'active_subscribers']
- **window**: 12
- **horizon**: 6
- **hidden_size**: 32
- **num_layers**: 1
- **epochs**: 400
- **lr**: 0.005
- **target**: revenue_idr_bn
