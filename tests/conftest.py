import pytest

from telco.config import load_config


@pytest.fixture()
def small_cfg(tmp_path):
    """Fast config writing everything into a temp dir (no side effects on the repo)."""
    return load_config(overrides={
        "paths": {
            "raw_data": str(tmp_path / "raw" / "customers.csv"),
            "timeseries_data": str(tmp_path / "raw" / "monthly_kpi.csv"),
            "incoming_dir": str(tmp_path / "incoming"),
            "data_manifest": str(tmp_path / "versions" / "manifest.json"),
            "registry_dir": str(tmp_path / "registry"),
            "monitoring_dir": str(tmp_path / "monitoring"),
            "reports_dir": str(tmp_path / "reports"),
        },
        "data": {"n_customers": 2500},
        "churn_model": {"max_epochs": 6, "patience": 3, "hidden_dims": [32, 16]},
        "arpu_model": {"max_epochs": 6, "patience": 3, "hidden_dims": [32]},
        "forecast": {"epochs": 40, "hidden_size": 8},
        "segmentation": {"k_min": 3, "k_max": 4, "sample_size": 1000},
        "monitoring": {"min_new_samples": 500, "default_batch_size": 1500},
    })
