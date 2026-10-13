"""Monitoring pipeline: drift + performance checks → retrain trigger → (optional) retrain.

    python -m telco.pipelines.monitor --scenario severe            # simulate a new batch & monitor
    python -m telco.pipelines.monitor --batch data/incoming/x.csv  # monitor a real batch
    python -m telco.pipelines.monitor --scenario severe --auto-retrain

Writes ``monitoring/latest_report.json``, ``monitoring/latest_drift.csv`` and
appends to ``monitoring/history.jsonl``. In GitHub Actions it also writes
``retrained=true|false`` to ``$GITHUB_OUTPUT``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from telco.config import Config, load_config
from telco.constants import CHURN_MODEL
from telco.data.generator import DRIFT_SCENARIOS, generate_customers
from telco.data.validation import coerce_types, validate_customers
from telco.mlops.monitoring import drift_report, prediction_drift
from telco.mlops.registry import ModelRegistry
from telco.mlops.retraining import evaluate_retrain_trigger
from telco.models.pipeline import TabularPipeline
from telco.utils import get_logger

Logger = Callable[[str], None]


def generate_batch(cfg: Config, scenario: str = "mild", n: int | None = None, seed: int | None = None) -> Path:
    """Simulate a newly-labelled batch of customers (labels arrive after 90 days)."""
    if scenario not in DRIFT_SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    n = n or int(cfg["monitoring"].get("default_batch_size", 5000))
    seed = seed if seed is not None else int(datetime.now().timestamp()) % 1_000_000
    df = generate_customers(n, seed=seed, target_churn_rate=cfg["data"]["target_churn_rate"],
                            treatment_rate=cfg["data"]["campaign_treatment_rate"], drift=scenario,
                            id_prefix=f"B{stamp}")
    out_dir = cfg.path("incoming_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"batch_{stamp}_{scenario}.csv"
    df.to_csv(path, index=False)
    return path


def list_batches(cfg: Config) -> list[Path]:
    d = cfg.path("incoming_dir")
    return sorted(d.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True) if d.exists() else []


def _age_days(iso: str) -> float:
    created = datetime.fromisoformat(iso)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - created).total_seconds() / 86400


def run_monitoring(cfg: Config, batch_path: str | Path, log: Logger | None = None) -> dict:
    log = log or get_logger("monitor").info
    registry = ModelRegistry(cfg.path("registry_dir"))
    entry = registry.get_production(CHURN_MODEL)
    if entry is None:
        raise RuntimeError("No production churn model. Run the training pipeline first.")
    art = registry.artifact_dir(entry)
    pipe = TabularPipeline.load(art)
    reference = pd.read_csv(art / "reference_sample.csv")

    batch_path = Path(batch_path)
    raw = pd.read_csv(batch_path)
    has_labels = cfg["features"]["target"] in raw.columns
    val = validate_customers(raw, cfg, require_target=has_labels)
    val.raise_if_failed()
    current = coerce_types(raw, cfg)
    feats = cfg["features"]
    mon = cfg["monitoring"]

    drift = drift_report(reference, current, feats["numeric"], feats["categorical"],
                         mon["psi_warning"], mon["psi_threshold"])
    scores = pipe.predict(current)
    pdrift = prediction_drift(reference["_score"], scores)
    current_auc = None
    if has_labels and current[feats["target"]].nunique() == 2:
        current_auc = float(roc_auc_score(current[feats["target"]], scores))
    ref_auc = entry["metrics"].get("test_roc_auc")
    age = _age_days(entry["created_at"])
    decision = evaluate_retrain_trigger(drift, pdrift["score_psi"], current_auc, ref_auc, age, len(current), mon)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "batch": batch_path.name, "n_rows": int(len(current)),
        "model": CHURN_MODEL, "model_version": entry["version"],
        "n_drifted_features": int((drift["status"] == "Drift").sum()),
        "n_warning_features": int((drift["status"] == "Warning").sum()),
        **pdrift, "reference_auc": ref_auc, "current_auc": current_auc,
        "current_churn_rate": float(current[feats["target"]].mean()) if has_labels else None,
        "reference_churn_rate": float(reference[feats["target"]].mean()) if feats["target"] in reference else None,
        "model_age_days": round(age, 2), "decision": decision.to_dict(),
        "validation_warnings": val.warnings,
    }
    mdir = cfg.path("monitoring_dir")
    mdir.mkdir(parents=True, exist_ok=True)
    drift.to_csv(mdir / "latest_drift.csv", index=False)
    (mdir / "latest_report.json").write_text(json.dumps(report, indent=2, default=str))
    with open(mdir / "history.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({k: report[k] for k in ("timestamp", "batch", "n_rows", "model_version",
                                                     "n_drifted_features", "score_psi", "current_auc",
                                                     "reference_auc", "current_churn_rate")}
                            | {"retrain": decision.retrain, "severity": decision.severity}, default=str) + "\n")
    log(f"Monitoring {batch_path.name}: drifted={report['n_drifted_features']} score_psi={pdrift['score_psi']:.3f} "
        f"AUC={current_auc if current_auc is None else round(current_auc, 4)} → retrain={decision.retrain} "
        f"({decision.severity})")
    for r in decision.reasons:
        log(f"  - {r}")
    report["drift_table"] = drift
    return report


def load_history(cfg: Config) -> pd.DataFrame:
    path = cfg.path("monitoring_dir") / "history.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Monitor production churn model")
    ap.add_argument("--config", default=None)
    ap.add_argument("--batch", default=None, help="path to a labelled/unlabelled CSV batch")
    ap.add_argument("--scenario", default="mild", choices=list(DRIFT_SCENARIOS))
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--auto-retrain", action="store_true")
    args = ap.parse_args()
    log = get_logger("monitor").info
    cfg = load_config(args.config)
    batch = Path(args.batch) if args.batch else generate_batch(cfg, args.scenario, args.n, args.seed)
    report = run_monitoring(cfg, batch, log)
    retrained = False
    if args.auto_retrain and report["decision"]["retrain"]:
        from telco.pipelines.train import run_training

        log("Retrain triggered → running training pipeline with new batch")
        summary = run_training(cfg, extra_data=[batch], skip_forecast=True, log=log)
        retrained = True
        log(f"Churn promotion: {summary['churn']['promotion']}")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as fh:
            fh.write(f"retrained={'true' if retrained else 'false'}\n")
            fh.write(f"severity={report['decision']['severity']}\n")
    print(json.dumps({k: v for k, v in report.items() if k != "drift_table"}, indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()
