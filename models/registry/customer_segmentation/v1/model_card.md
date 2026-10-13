# Model Card — customer_segmentation v1

- Created: 2026-10-13T12:43:26+00:00
- Data version: `customers@v1`
- Git commit: `None`
- Description: K-Means behavioural segmentation

## Metrics

| metric | value |
|---|---|
| k | 3.0000 |
| silhouette | 0.1738 |
| silhouette_k3 | 0.1738 |
| silhouette_k4 | 0.1659 |
| silhouette_k5 | 0.1359 |
| silhouette_k6 | 0.1274 |
| silhouette_k7 | 0.1295 |

## Parameters

- **k_min**: 3
- **k_max**: 7
- **sample_size**: 6000
- **features**: ['arpu_idr', 'data_usage_gb', 'voice_minutes', 'tenure_months', 'num_complaints', 'nps_score', 'days_since_last_topup']
- **log_features**: ['arpu_idr', 'data_usage_gb', 'voice_minutes', 'tenure_months']
