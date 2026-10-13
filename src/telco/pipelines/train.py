"""End-to-end training pipeline.

    python -m telco.pipelines.train                    # train everything
    python -m telco.pipelines.train --regenerate-data  # fresh synthetic data
    python -m telco.pipelines.train --extra-data data/incoming/batch_x.csv   # retrain with new labels

Steps: data → validation → data versioning → churn MLP (+ logistic baseline,
business threshold, permutation importance) → ARPU MLP regressor (+ linear
baseline) → K-Means segmentation → LSTM forecasters (+ Holt-Winters &
seasonal-naive baselines) → model registry with champion/challenger promotion.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from telco.analytics.business_metrics import customer_lifetime_value
from telco.config import Config, load_config
from telco.constants import ARPU_MODEL, CHURN_MODEL, CUSTOMER_DATASET, KPI_DATASET, SEGMENT_MODEL, forecast_model_name
from telco.data.generator import generate_customers, generate_monthly_kpis
from telco.data.validation import coerce_types, validate_customers
from telco.data.versioning import DataVersionManager
from telco.evaluation.metrics import classification_metrics, optimal_threshold, permutation_importance, regression_metrics
from telco.features.engineering import build_preprocessor, feature_lists, prepare_frame
from telco.mlops.registry import ModelRegistry
from telco.models.forecaster import LSTMForecaster, holt_winters_forecast, seasonal_naive_forecast
from telco.models.pipeline import TabularPipeline
from telco.models.segmentation import CustomerSegmenter
from telco.models.torch_tabular import TorchTabularModel
from telco.utils import get_logger, set_seed

Logger = Callable[[str], None]


# --------------------------------------------------------------------------- data
def ensure_data(cfg: Config, regenerate: bool = False, log: Logger = print) -> tuple[Path, Path]:
    cust_path, ts_path = cfg.path("raw_data"), cfg.path("timeseries_data")
    d = cfg["data"]
    if regenerate or not cust_path.exists():
        log(f"Generating {d['n_customers']:,} synthetic customers → {cust_path.name}")
        cust_path.parent.mkdir(parents=True, exist_ok=True)
        generate_customers(d["n_customers"], seed=cfg.seed, target_churn_rate=d["target_churn_rate"],
                           treatment_rate=d["campaign_treatment_rate"]).to_csv(cust_path, index=False)
    if regenerate or not ts_path.exists():
        log(f"Generating {d['n_months']} months of KPI time series → {ts_path.name}")
        ts_path.parent.mkdir(parents=True, exist_ok=True)
        generate_monthly_kpis(d["start_month"], d["n_months"], seed=cfg.seed).to_csv(ts_path, index=False)
    return cust_path, ts_path


def load_training_data(cfg: Config, extra_data: list[Path] | None, log: Logger) -> pd.DataFrame:
    """Load base data, append new labelled batches (sliding window) and persist."""
    path = cfg.path("raw_data")
    base = pd.read_csv(path).assign(_source="base")
    if not extra_data:
        return base
    frames = [base]
    for p in extra_data:
        new = pd.read_csv(p)
        report = validate_customers(new, cfg, require_target=True)
        report.raise_if_failed()
        frames.append(new.assign(_source="new"))
        log(f"Appending {len(new):,} labelled rows from {Path(p).name}")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(cfg["features"]["id_col"], keep="last")
    df = df.tail(int(cfg["data"].get("max_training_rows", 60000))).reset_index(drop=True)
    df.drop(columns="_source").to_csv(path, index=False)
    log(f"Training set now {len(df):,} rows (sliding window); saved to {path.name}")
    return df


# --------------------------------------------------------------------------- promotion
def champion_challenger(registry: ModelRegistry, name: str, entry: dict, challenger_score: float,
                        champion_score_fn: Callable[[Path], float], higher_is_better: bool,
                        tolerance: float, log: Logger) -> dict:
    champion = next((v for v in registry.list_versions(name)
                     if v["stage"] == "Production" and v["version"] != entry["version"]), None)
    if champion is None:
        registry.transition(name, entry["version"], "Production", "first production model")
        log(f"  {name} v{entry['version']} → Production (no previous champion)")
        return {"promoted": True, "champion_version": None, "reason": "first model"}
    try:
        champ_score = float(champion_score_fn(registry.artifact_dir(champion)))
    except Exception as exc:  # incompatible champion (e.g. schema change) → replace it
        registry.transition(name, entry["version"], "Production", f"champion not evaluable: {exc}")
        return {"promoted": True, "champion_version": champion["version"], "reason": f"champion error: {exc}"}
    better = (challenger_score >= champ_score - tolerance) if higher_is_better \
        else (challenger_score <= champ_score + tolerance)
    result = {"champion_version": champion["version"], "champion_score": champ_score,
              "challenger_score": challenger_score, "promoted": bool(better)}
    registry.update_metrics(name, entry["version"], {"champion_score_same_holdout": champ_score})
    if better:
        registry.transition(name, entry["version"], "Production",
                            f"challenger {challenger_score:.4f} vs champion {champ_score:.4f}")
        result["reason"] = "challenger ≥ champion (within tolerance)"
    else:
        result["reason"] = "challenger worse than champion; kept in Staging"
    log(f"  {name}: challenger={challenger_score:.4f} champion v{champion['version']}={champ_score:.4f} "
        f"→ {'PROMOTED' if better else 'kept in Staging'}")
    return result


# --------------------------------------------------------------------------- churn
def train_churn(cfg: Config, df: pd.DataFrame, data_version: str, registry: ModelRegistry,
                log: Logger) -> dict:
    target, seed = cfg["features"]["target"], cfg.seed
    num, cat = feature_lists(cfg, "churn")
    y = df[target].astype(int).to_numpy()
    idx = np.arange(len(df))
    sp = cfg["split"]
    idx_trv, idx_te = train_test_split(idx, test_size=sp["test_size"], stratify=y, random_state=seed)
    idx_tr, idx_va = train_test_split(idx_trv, test_size=sp["val_size"] / (1 - sp["test_size"]),
                                      stratify=y[idx_trv], random_state=seed)
    d_tr, d_va, d_te = df.iloc[idx_tr], df.iloc[idx_va], df.iloc[idx_te]

    pre = build_preprocessor(num, cat)
    X_tr = pre.fit_transform(prepare_frame(d_tr, num, cat)).astype(np.float32)
    X_va = pre.transform(prepare_frame(d_va, num, cat)).astype(np.float32)
    X_te = pre.transform(prepare_frame(d_te, num, cat)).astype(np.float32)

    mcfg = cfg["churn_model"]
    t0 = time.time()
    model = TorchTabularModel(task="binary", hidden_dims=mcfg["hidden_dims"], dropout=mcfg["dropout"],
                              lr=mcfg["lr"], weight_decay=mcfg["weight_decay"], batch_size=mcfg["batch_size"],
                              max_epochs=mcfg["max_epochs"], patience=mcfg["patience"],
                              use_class_weight=mcfg.get("use_class_weight", False), seed=seed)
    model.fit(X_tr, y[idx_tr], X_va, y[idx_va])
    log(f"  churn MLP trained in {time.time() - t0:.1f}s (best epoch {model.best_epoch})")
    baseline = LogisticRegression(max_iter=3000).fit(X_tr, y[idx_tr])

    biz = cfg["business"]
    value_va = d_va["arpu_idr"].to_numpy() * biz["gross_margin"] * biz["retention_horizon_months"]
    thr, _ = optimal_threshold(y[idx_va], model.predict_proba(X_va), value_va,
                               biz["retention_offer_cost_idr"], biz["retention_success_rate"])
    p_te = model.predict_proba(X_te)
    p_te_base = baseline.predict_proba(X_te)[:, 1]
    metrics = {f"test_{k}": v for k, v in classification_metrics(y[idx_te], p_te, thr).items()}
    metrics.update({f"baseline_logreg_{k}": v for k, v in
                    classification_metrics(y[idx_te], p_te_base, 0.5).items() if k in ("roc_auc", "pr_auc", "f1", "brier")})
    metrics["val_roc_auc"] = float(max(h["val_score"] for h in model.history))
    clv_train = customer_lifetime_value(d_tr["arpu_idr"], model.predict_proba(X_tr), biz["gross_margin"],
                                        biz["max_lifetime_months"])
    pipe = TabularPipeline(pre, model, num, cat, threshold=thr,
                           meta={"clv_p75": float(np.percentile(clv_train, 75)),
                                 "train_rows": len(idx_tr), "data_version": data_version})

    sample = d_te.sample(min(3000, len(d_te)), random_state=seed)
    raw_feats = [c for c in cfg["features"]["numeric"] + cfg["features"]["categorical"]]
    imp = permutation_importance(pipe.predict, sample, sample[target].to_numpy(), raw_feats, n_repeats=2, seed=seed)

    with tempfile.TemporaryDirectory() as tmp:
        art = Path(tmp)
        pipe.save(art)
        pd.DataFrame({"customer_id": d_te[cfg["features"]["id_col"]].to_numpy(), "y_true": y[idx_te],
                      "p_model": p_te, "p_baseline": p_te_base,
                      "arpu_idr": d_te["arpu_idr"].to_numpy()}).to_csv(art / "test_predictions.csv", index=False)
        pd.DataFrame(model.history).to_csv(art / "training_history.csv", index=False)
        imp.to_csv(art / "feature_importance.csv", index=False)
        ref = d_tr.drop(columns=["_source"], errors="ignore").sample(min(5000, len(d_tr)), random_state=seed)
        ref.assign(_score=pipe.predict(ref)).to_csv(art / "reference_sample.csv", index=False)
        params = {**mcfg, "threshold": thr, "n_features_encoded": int(X_tr.shape[1]), "framework": "pytorch"}
        entry = registry.register(CHURN_MODEL, art, metrics, params, data_version,
                                  description="PyTorch MLP churn classifier (90-day churn)", stage="Staging")
    log(f"  churn_model v{entry['version']}: AUC={metrics['test_roc_auc']:.4f} "
        f"(logreg {metrics['baseline_logreg_roc_auc']:.4f}), threshold={thr:.2f}")

    # Champion/challenger on data the champion has never seen (new batch) when available
    cmp = d_te[d_te.get("_source", pd.Series("base", index=d_te.index)) == "new"]
    cmp = cmp if len(cmp) >= 500 and cmp[target].nunique() == 2 else d_te
    chall = float(roc_auc_score(cmp[target], pipe.predict(cmp)))
    promo = champion_challenger(registry, CHURN_MODEL, entry, chall,
                                lambda p: roc_auc_score(cmp[target], TabularPipeline.load(p).predict(cmp)),
                                True, cfg["monitoring"]["promotion_tolerance"], log)
    promo["comparison_rows"] = int(len(cmp))
    return {"version": entry["version"], "metrics": metrics, "promotion": promo}


# --------------------------------------------------------------------------- ARPU
def train_arpu(cfg: Config, df: pd.DataFrame, data_version: str, registry: ModelRegistry,
               log: Logger) -> dict:
    seed = cfg.seed
    num, cat = feature_lists(cfg, "arpu")
    y = df["arpu_idr"].to_numpy(dtype=float)
    y_log = np.log1p(y)
    idx_trv, idx_te = train_test_split(np.arange(len(df)), test_size=cfg["split"]["test_size"], random_state=seed)
    idx_tr, idx_va = train_test_split(idx_trv, test_size=cfg["split"]["val_size"], random_state=seed)
    pre = build_preprocessor(num, cat)
    X_tr = pre.fit_transform(prepare_frame(df.iloc[idx_tr], num, cat)).astype(np.float32)
    X_va = pre.transform(prepare_frame(df.iloc[idx_va], num, cat)).astype(np.float32)
    X_te = pre.transform(prepare_frame(df.iloc[idx_te], num, cat)).astype(np.float32)
    mcfg = cfg["arpu_model"]
    model = TorchTabularModel(task="regression", hidden_dims=mcfg["hidden_dims"], dropout=mcfg["dropout"],
                              lr=mcfg["lr"], weight_decay=mcfg["weight_decay"], batch_size=mcfg["batch_size"],
                              max_epochs=mcfg["max_epochs"], patience=mcfg["patience"], seed=seed)
    model.fit(X_tr, y_log[idx_tr], X_va, y_log[idx_va])
    pred = np.expm1(model.predict(X_te))
    lin = LinearRegression().fit(X_tr, y_log[idx_tr])
    pred_lin = np.expm1(lin.predict(X_te))
    metrics = {f"test_{k}": v for k, v in regression_metrics(y[idx_te], pred).items()}
    metrics.update({f"baseline_linreg_{k}": v for k, v in regression_metrics(y[idx_te], pred_lin).items()})
    pipe = TabularPipeline(pre, model, num, cat, target_transform="log1p", meta={"data_version": data_version})
    d_te = df.iloc[idx_te]
    with tempfile.TemporaryDirectory() as tmp:
        art = Path(tmp)
        pipe.save(art)
        pd.DataFrame({"y_true": y[idx_te], "y_pred": pred, "y_pred_baseline": pred_lin,
                      "plan_type": d_te["plan_type"].to_numpy(), "package": d_te["package"].to_numpy()}
                     ).to_csv(art / "test_predictions.csv", index=False)
        pd.DataFrame(model.history).to_csv(art / "training_history.csv", index=False)
        entry = registry.register(ARPU_MODEL, art, metrics, {**mcfg, "target_transform": "log1p"},
                                  data_version, description="PyTorch MLP regressor for monthly ARPU")
    log(f"  arpu_regressor v{entry['version']}: R²={metrics['test_r2']:.4f} MAPE={metrics['test_mape']:.2f}% "
        f"(linreg R²={metrics['baseline_linreg_r2']:.4f})")
    promo = champion_challenger(
        registry, ARPU_MODEL, entry, metrics["test_r2"],
        lambda p: r2_score(y[idx_te], TabularPipeline.load(p).predict(d_te)), True,
        cfg["monitoring"]["promotion_tolerance"], log)
    return {"version": entry["version"], "metrics": metrics, "promotion": promo}


# --------------------------------------------------------------------------- segmentation
def train_segmentation(cfg: Config, df: pd.DataFrame, data_version: str, registry: ModelRegistry,
                       log: Logger) -> dict:
    s = cfg["segmentation"]
    seg = CustomerSegmenter(s["features"], s["log_features"], s["k_min"], s["k_max"], s["sample_size"], cfg.seed)
    seg.fit(df)
    sample = df.sample(min(3000, len(df)), random_state=cfg.seed)
    xy = seg.project_2d(sample)
    with tempfile.TemporaryDirectory() as tmp:
        art = Path(tmp)
        seg.save(art / "segmenter.joblib")
        seg.profile_.to_csv(art / "segment_profile.csv", index_label="segment")
        pd.DataFrame({"pc1": xy[:, 0], "pc2": xy[:, 1], "segment": seg.predict_names(sample),
                      "churn": sample[cfg["features"]["target"]].to_numpy(),
                      "arpu_idr": sample["arpu_idr"].to_numpy()}).to_csv(art / "pca_sample.csv", index=False)
        metrics = {"k": seg.k, "silhouette": seg.silhouette_scores[seg.k],
                   **{f"silhouette_k{k}": v for k, v in seg.silhouette_scores.items()}}
        entry = registry.register(SEGMENT_MODEL, art, metrics, s, data_version,
                                  description="K-Means behavioural segmentation", stage="Production")
    log(f"  customer_segmentation v{entry['version']}: k={seg.k} silhouette={metrics['silhouette']:.3f}")
    return {"version": entry["version"], "metrics": metrics}


# --------------------------------------------------------------------------- forecasting
def train_forecasters(cfg: Config, ts: pd.DataFrame, data_version: str, registry: ModelRegistry,
                      log: Logger) -> dict:
    f = cfg["forecast"]
    ts = ts.sort_values("month")
    dates = pd.DatetimeIndex(pd.to_datetime(ts["month"]))
    out = {}
    for target in f["targets"]:
        values = ts[target].to_numpy(dtype=float)
        h = f["horizon"]
        kw = dict(window=f["window"], hidden_size=f["hidden_size"], num_layers=f["num_layers"],
                  epochs=f["epochs"], lr=f["lr"], seed=cfg.seed)
        bt = LSTMForecaster(**kw).fit(dates[:-h], values[:-h]).predict(h)
        actual = values[-h:]
        preds = {"lstm": bt["forecast"].to_numpy(), "seasonal_naive": seasonal_naive_forecast(values[:-h], h)}
        try:
            preds["holt_winters"] = holt_winters_forecast(values[:-h], h)
        except Exception as exc:  # pragma: no cover
            log(f"  Holt-Winters failed: {exc}")
        metrics = {}
        for m, p in preds.items():
            for k, v in regression_metrics(actual, p).items():
                if k != "r2":
                    metrics[f"backtest_{m}_{k}"] = v
        final = LSTMForecaster(**kw).fit(dates, values)
        fc = final.predict(h)
        with tempfile.TemporaryDirectory() as tmp:
            art = Path(tmp)
            final.save(art / "lstm.pt")
            pd.DataFrame({"month": dates[-h:], "actual": actual, **preds}).to_csv(art / "backtest.csv", index=False)
            fc.to_csv(art / "forecast.csv", index=False)
            name = forecast_model_name(target)
            entry = registry.register(name, art, metrics, {**f, "target": target}, data_version,
                                      description=f"LSTM monthly forecaster for {target}", stage="Production")
        log(f"  {name} v{entry['version']}: backtest MAPE LSTM={metrics['backtest_lstm_mape']:.2f}% "
            f"HW={metrics.get('backtest_holt_winters_mape', float('nan')):.2f}% "
            f"sNaive={metrics['backtest_seasonal_naive_mape']:.2f}%")
        out[target] = {"version": entry["version"], "metrics": metrics}
    return out


# --------------------------------------------------------------------------- orchestration
def run_training(cfg: Config | None = None, extra_data: list[str | Path] | None = None,
                 regenerate: bool = False, skip_forecast: bool = False,
                 log: Logger | None = None) -> dict:
    cfg = cfg or load_config()
    log = log or get_logger("train").info
    set_seed(cfg.seed)
    t0 = time.time()
    _, ts_path = ensure_data(cfg, regenerate, log)
    df = load_training_data(cfg, [Path(p) for p in extra_data] if extra_data else None, log)
    report = validate_customers(df, cfg, require_target=True)
    for w in report.warnings:
        log(f"  [validation warning] {w}")
    report.raise_if_failed()
    df = coerce_types(df, cfg)

    dvm = DataVersionManager(cfg.path("data_manifest"))
    dv = dvm.register(cfg.path("raw_data"), CUSTOMER_DATASET, "customer snapshot used for training")
    ts_dv = dvm.register(ts_path, KPI_DATASET, "monthly company KPIs", target=None)
    log(f"Data version {dv['version_id']} ({dv['rows']:,} rows, sha256 {dv['sha256'][:10]}…)")

    registry = ModelRegistry(cfg.path("registry_dir"))
    summary = {"data_version": dv["version_id"], "rows": len(df)}
    log("Training churn model …")
    summary["churn"] = train_churn(cfg, df, dv["version_id"], registry, log)
    log("Training ARPU regressor …")
    summary["arpu"] = train_arpu(cfg, df, dv["version_id"], registry, log)
    log("Training segmentation …")
    summary["segmentation"] = train_segmentation(cfg, df, dv["version_id"], registry, log)
    if not skip_forecast:
        log("Training LSTM forecasters …")
        summary["forecast"] = train_forecasters(cfg, pd.read_csv(ts_path), ts_dv["version_id"], registry, log)
    summary["duration_sec"] = round(time.time() - t0, 1)
    reports = cfg.path("reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"Pipeline finished in {summary['duration_sec']}s")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Train all telco models")
    ap.add_argument("--config", default=None)
    ap.add_argument("--regenerate-data", action="store_true")
    ap.add_argument("--extra-data", nargs="*", default=None, help="new labelled CSV batches to append")
    ap.add_argument("--skip-forecast", action="store_true")
    args = ap.parse_args()
    run_training(load_config(args.config), args.extra_data, args.regenerate_data, args.skip_forecast)


if __name__ == "__main__":
    main()
