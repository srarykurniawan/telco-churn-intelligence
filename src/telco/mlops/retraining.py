"""Retraining trigger policy.

A retrain is triggered when *any* of these conditions hold (and there is
enough new labelled data to learn from):

1. Data drift     — ≥ ``min_drifted_features`` features with PSI ≥ ``psi_threshold``
2. Concept drift  — prediction-score PSI ≥ ``psi_threshold``
3. Performance    — ROC-AUC on fresh labelled data dropped ≥ ``auc_drop_threshold``
4. Staleness      — production model older than ``max_model_age_days``
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd


@dataclass
class RetrainDecision:
    retrain: bool
    severity: str
    reasons: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_retrain_trigger(drift_df: pd.DataFrame, score_psi: float, current_auc: float | None,
                             reference_auc: float | None, model_age_days: float, n_new_samples: int,
                             policy: dict) -> RetrainDecision:
    psi_thr = policy.get("psi_threshold", 0.2)
    n_drifted = int((drift_df["status"] == "Drift").sum()) if not drift_df.empty else 0
    n_warning = int((drift_df["status"] == "Warning").sum()) if not drift_df.empty else 0
    auc_drop = (reference_auc - current_auc) if (current_auc is not None and reference_auc is not None) else None

    checks = {
        "data_drift": {"triggered": n_drifted >= policy.get("min_drifted_features", 2),
                       "value": n_drifted, "rule": f"≥ {policy.get('min_drifted_features', 2)} fitur PSI ≥ {psi_thr}"},
        "prediction_drift": {"triggered": bool(score_psi >= psi_thr), "value": round(float(score_psi), 4),
                             "rule": f"PSI skor ≥ {psi_thr}"},
        "performance_decay": {"triggered": bool(auc_drop is not None and auc_drop >= policy.get("auc_drop_threshold", 0.05)),
                              "value": None if auc_drop is None else round(float(auc_drop), 4),
                              "rule": f"penurunan AUC ≥ {policy.get('auc_drop_threshold', 0.05)}"},
        "staleness": {"triggered": model_age_days >= policy.get("max_model_age_days", 90),
                      "value": round(float(model_age_days), 1),
                      "rule": f"umur model ≥ {policy.get('max_model_age_days', 90)} hari"},
    }
    reasons = [f"{k}: {v['value']} ({v['rule']})" for k, v in checks.items() if v["triggered"]]
    enough = n_new_samples >= policy.get("min_new_samples", 1000)
    checks["enough_new_data"] = {"triggered": enough, "value": n_new_samples,
                                 "rule": f"≥ {policy.get('min_new_samples', 1000)} sampel berlabel"}

    if checks["performance_decay"]["triggered"] or (checks["data_drift"]["triggered"] and checks["prediction_drift"]["triggered"]):
        severity = "critical"
    elif reasons:
        severity = "high"
    elif n_warning:
        severity = "warning"
    else:
        severity = "ok"
    retrain = bool(reasons) and enough
    if reasons and not enough:
        reasons.append("Retrain ditunda: data berlabel baru belum cukup.")
    return RetrainDecision(retrain, severity, reasons, checks)
