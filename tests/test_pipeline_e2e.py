"""Integration test: train → score → monitor (drift) → retrain → champion/challenger."""

import pandas as pd

from telco.constants import ARPU_MODEL, CHURN_MODEL, SEGMENT_MODEL, forecast_model_name
from telco.data.generator import generate_customers
from telco.mlops.registry import ModelRegistry
from telco.pipelines.monitor import generate_batch, load_history, run_monitoring
from telco.pipelines.predict import score_customers
from telco.pipelines.train import run_training


def test_end_to_end(small_cfg):
    summary = run_training(small_cfg, regenerate=True, log=lambda m: None)
    reg = ModelRegistry(small_cfg.path("registry_dir"))
    for name in (CHURN_MODEL, ARPU_MODEL, SEGMENT_MODEL, forecast_model_name("revenue_idr_bn")):
        assert reg.get_production(name) is not None, name
    assert summary["churn"]["metrics"]["test_roc_auc"] > 0.65
    assert (small_cfg.path("reports_dir") / "training_summary.json").exists()

    # batch scoring on unlabeled data
    new = generate_customers(200, seed=99).drop(columns=["churn"])
    scored = score_customers(new, small_cfg)
    assert scored["churn_probability"].between(0, 1).all()
    assert set(scored["risk_tier"]) <= {"High", "Medium", "Low"}
    assert scored["recommended_action"].notna().all()

    # stable batch → no retrain
    calm = run_monitoring(small_cfg, generate_batch(small_cfg, "none", seed=11), log=lambda m: None)
    assert calm["decision"]["retrain"] is False

    # severe drift → retrain trigger → retrain → v2 registered
    batch = generate_batch(small_cfg, "severe", seed=12)
    report = run_monitoring(small_cfg, batch, log=lambda m: None)
    assert report["decision"]["retrain"] is True
    assert report["n_drifted_features"] >= 2
    s2 = run_training(small_cfg, extra_data=[batch], skip_forecast=True, log=lambda m: None)
    assert s2["churn"]["version"] == 2
    assert s2["churn"]["promotion"]["comparison_rows"] > 0
    assert len(pd.read_csv(small_cfg.path("raw_data"))) == 2500 + 1500
    assert len(load_history(small_cfg)) == 2
