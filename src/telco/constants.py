"""Registry model names shared by pipelines and the app."""

CHURN_MODEL = "churn_model"
ARPU_MODEL = "arpu_regressor"
SEGMENT_MODEL = "customer_segmentation"
FORECAST_PREFIX = "forecast_"
CUSTOMER_DATASET = "customers"
KPI_DATASET = "monthly_kpi"


def forecast_model_name(target: str) -> str:
    return f"{FORECAST_PREFIX}{target}"
