"""Batch scoring with all production models.

    python -m telco.pipelines.predict --input customers.csv --output scored.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from telco.analytics.business_metrics import customer_lifetime_value
from telco.config import Config, load_config
from telco.constants import ARPU_MODEL, CHURN_MODEL, SEGMENT_MODEL
from telco.data.validation import coerce_types, validate_customers
from telco.mlops.registry import ModelRegistry
from telco.models.pipeline import TabularPipeline
from telco.models.segmentation import SEGMENT_ACTIONS, CustomerSegmenter


def load_production_models(cfg: Config) -> dict:
    reg = ModelRegistry(cfg.path("registry_dir"))
    out = {}
    for name in (CHURN_MODEL, ARPU_MODEL):
        entry = reg.get_production(name)
        if entry is None:
            raise RuntimeError(f"No production version for {name}. Train first.")
        out[name] = (TabularPipeline.load(reg.artifact_dir(entry)), entry)
    seg_entry = reg.get_production(SEGMENT_MODEL)
    if seg_entry:
        out[SEGMENT_MODEL] = (CustomerSegmenter.load(reg.artifact_dir(seg_entry) / "segmenter.joblib"), seg_entry)
    return out


def risk_tier(prob: np.ndarray, threshold: float) -> np.ndarray:
    return np.select([prob >= threshold, prob >= 0.6 * threshold], ["High", "Medium"], "Low")


def recommend_action(prob: float, threshold: float, clv: float, clv_p75: float, segment: str) -> str:
    if prob >= threshold and clv >= clv_p75:
        return "Prioritas retensi: outbound call agen + penawaran Discount personal"
    if prob >= threshold:
        return "Retensi otomatis: Bonus Data via SMS / push notification"
    if prob >= 0.6 * threshold:
        return "Nurture: Loyalty Points + survei kepuasan (cek keluhan & jaringan)"
    return SEGMENT_ACTIONS.get(segment, "Upsell / cross-sell paket & add-on")


def score_customers(df: pd.DataFrame, cfg: Config, models: dict | None = None) -> pd.DataFrame:
    report = validate_customers(df, cfg, require_target=False)
    report.raise_if_failed()
    models = models or load_production_models(cfg)
    clean = coerce_types(df, cfg)
    churn_pipe, churn_entry = models[CHURN_MODEL]
    arpu_pipe, _ = models[ARPU_MODEL]
    biz = cfg["business"]
    prob = churn_pipe.predict(clean)
    pred_arpu = arpu_pipe.predict(clean)
    clv = customer_lifetime_value(clean["arpu_idr"].fillna(pd.Series(pred_arpu, index=clean.index)),
                                  prob, biz["gross_margin"], biz["max_lifetime_months"])
    thr = churn_pipe.threshold
    out = df.copy()
    out["churn_probability"] = np.round(prob, 4)
    out["risk_tier"] = risk_tier(prob, thr)
    out["predicted_arpu_idr"] = np.round(pred_arpu, -2)
    out["clv_idr"] = np.round(clv, -2)
    if SEGMENT_MODEL in models:
        out["segment"] = models[SEGMENT_MODEL][0].predict_names(clean)
    else:
        out["segment"] = "N/A"
    p75 = float(churn_pipe.meta.get("clv_p75", np.percentile(clv, 75)))
    out["recommended_action"] = [recommend_action(p, thr, c, p75, s)
                                 for p, c, s in zip(prob, clv, out["segment"], strict=False)]
    out["model_version"] = f"{CHURN_MODEL}:v{churn_entry['version']}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Score customers with production models")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    scored = score_customers(pd.read_csv(args.input), cfg)
    scored.to_csv(args.output, index=False)
    print(scored["risk_tier"].value_counts().to_string())


if __name__ == "__main__":
    main()
