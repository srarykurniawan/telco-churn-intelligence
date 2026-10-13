import pandas as pd
import pytest

from telco.mlops.registry import ModelRegistry
from telco.mlops.retraining import evaluate_retrain_trigger

POLICY = {"psi_threshold": 0.2, "min_drifted_features": 2, "auc_drop_threshold": 0.05,
          "max_model_age_days": 90, "min_new_samples": 1000}


def _artifact(tmp_path, name="a"):
    d = tmp_path / name
    d.mkdir()
    (d / "model.txt").write_text("weights")
    return d


def test_registry_lifecycle(tmp_path):
    reg = ModelRegistry(tmp_path / "reg")
    v1 = reg.register("m", _artifact(tmp_path, "a1"), {"auc": 0.7}, stage="Production")
    v2 = reg.register("m", _artifact(tmp_path, "a2"), {"auc": 0.8})
    assert v1["version"] == 1 and v2["version"] == 2 and v2["stage"] == "Staging"
    assert reg.get_production("m")["version"] == 1
    reg.transition("m", 2, "Production")
    assert reg.get_production("m")["version"] == 2
    assert reg.get("m", 1)["stage"] == "Archived"
    assert (reg.artifact_dir(reg.get("m", 2)) / "model_card.md").exists()
    reg.transition("m", 1, "Production")  # rollback
    assert reg.get_production("m")["version"] == 1
    with pytest.raises(KeyError):
        reg.transition("m", 99, "Production")


def _drift(n_drift):
    return pd.DataFrame({"status": ["Drift"] * n_drift + ["OK"] * 5})


def test_trigger_no_retrain_when_stable():
    d = evaluate_retrain_trigger(_drift(0), 0.01, 0.77, 0.77, 5, 5000, POLICY)
    assert not d.retrain and d.severity == "ok"


def test_trigger_on_drift_and_performance():
    d = evaluate_retrain_trigger(_drift(3), 0.3, 0.70, 0.77, 5, 5000, POLICY)
    assert d.retrain and d.severity == "critical" and len(d.reasons) == 3


def test_trigger_on_staleness_only():
    d = evaluate_retrain_trigger(_drift(0), 0.0, None, 0.77, 120, 5000, POLICY)
    assert d.retrain and d.severity == "high"


def test_trigger_deferred_without_enough_data():
    d = evaluate_retrain_trigger(_drift(4), 0.5, None, 0.77, 1, 100, POLICY)
    assert not d.retrain and any("ditunda" in r for r in d.reasons)
