# Model Card — forecast_active_subscribers v2

- Created: 2026-10-15T07:54:54+00:00
- Data version: `monthly_kpi@v1`
- Git commit: `789bd41`
- Description: LSTM monthly forecaster for active_subscribers

## Metrics

| metric | value |
|---|---|
| backtest_lstm_rmse | 42589.4363 |
| backtest_lstm_mae | 30535.4262 |
| backtest_lstm_mape | 0.1937 |
| backtest_seasonal_naive_rmse | 82426.6814 |
| backtest_seasonal_naive_mae | 71007.7653 |
| backtest_seasonal_naive_mape | 0.4519 |
| backtest_holt_winters_rmse | 92495.3227 |
| backtest_holt_winters_mae | 74873.5299 |
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
