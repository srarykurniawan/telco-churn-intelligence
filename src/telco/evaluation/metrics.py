"""Evaluation metrics for classification, regression and forecasting,
plus business-oriented evaluation (lift, profit curve, optimal threshold)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn import metrics as skm


# --------------------------------------------------------------------------- classification
def ks_statistic(y_true, p) -> float:
    y, p = np.asarray(y_true), np.asarray(p)
    if y.min() == y.max():
        return float("nan")
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def classification_metrics(y_true, p, threshold: float = 0.5) -> dict:
    y, p = np.asarray(y_true).astype(int), np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = skm.confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(skm.roc_auc_score(y, p)),
        "pr_auc": float(skm.average_precision_score(y, p)),
        "f1": float(skm.f1_score(y, pred, zero_division=0)),
        "precision": float(skm.precision_score(y, pred, zero_division=0)),
        "recall": float(skm.recall_score(y, pred, zero_division=0)),
        "accuracy": float(skm.accuracy_score(y, pred)),
        "brier": float(skm.brier_score_loss(y, p)),
        "log_loss": float(skm.log_loss(y, p, labels=[0, 1])),
        "ks": ks_statistic(y, p),
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def _downsample(df: pd.DataFrame, max_points: int = 300) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    idx = np.unique(np.linspace(0, len(df) - 1, max_points).astype(int))
    return df.iloc[idx].reset_index(drop=True)


def roc_points(y_true, p) -> pd.DataFrame:
    fpr, tpr, thr = skm.roc_curve(y_true, p)
    return _downsample(pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}))


def pr_points(y_true, p) -> pd.DataFrame:
    prec, rec, _ = skm.precision_recall_curve(y_true, p)
    return _downsample(pd.DataFrame({"recall": rec, "precision": prec}))


def lift_table(y_true, p, n_bins: int = 10) -> pd.DataFrame:
    """Decile analysis: decile 1 = highest predicted risk."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(p)}).sort_values("p", ascending=False)
    df["decile"] = (np.arange(len(df)) * n_bins // len(df)) + 1
    base = df["y"].mean()
    t = df.groupby("decile").agg(customers=("y", "size"), churners=("y", "sum"),
                                 avg_score=("p", "mean")).reset_index()
    t["churn_rate"] = t["churners"] / t["customers"]
    t["lift"] = t["churn_rate"] / base if base > 0 else np.nan
    t["cum_capture"] = t["churners"].cumsum() / t["churners"].sum()
    t["cum_customers_pct"] = t["customers"].cumsum() / t["customers"].sum()
    t["cum_lift"] = t["cum_capture"] / t["cum_customers_pct"]
    return t


def calibration_table(y_true, p, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    return (df.groupby("bin", observed=True)
            .agg(mean_predicted=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size"))
            .reset_index(drop=True))


def profit_curve(y_true, p, value_if_saved, cost_per_contact: float, success_rate: float,
                 thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Expected campaign profit when contacting everyone with score ≥ threshold.

    profit = Σ_TP success_rate × value_i  −  (#contacted × cost_per_contact)
    ``value_if_saved`` may be a scalar or a per-customer array (e.g. ARPU × margin × horizon).
    """
    y, p = np.asarray(y_true).astype(int), np.asarray(p, dtype=float)
    value = np.broadcast_to(np.asarray(value_if_saved, dtype=float), y.shape)
    thresholds = np.linspace(0.02, 0.95, 94) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        sel = p >= t
        tp = int((sel & (y == 1)).sum())
        gain = float((value * (sel & (y == 1))).sum() * success_rate)
        cost = float(sel.sum() * cost_per_contact)
        rows.append({"threshold": float(t), "contacted": int(sel.sum()), "tp": tp,
                     "fp": int(sel.sum() - tp), "gain": gain, "cost": cost, "profit": gain - cost})
    return pd.DataFrame(rows)


def optimal_threshold(y_true, p, value_if_saved, cost_per_contact: float,
                      success_rate: float) -> tuple[float, float]:
    curve = profit_curve(y_true, p, value_if_saved, cost_per_contact, success_rate)
    best = curve.loc[curve["profit"].idxmax()]
    return float(best["threshold"]), float(best["profit"])


# --------------------------------------------------------------------------- regression / forecast
def regression_metrics(y_true, y_pred) -> dict:
    y, yhat = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    nonzero = np.abs(y) > 1e-9
    return {
        "rmse": float(np.sqrt(np.mean((y - yhat) ** 2))),
        "mae": float(np.mean(np.abs(y - yhat))),
        "mape": float(np.mean(np.abs((y[nonzero] - yhat[nonzero]) / y[nonzero])) * 100),
        "r2": float(skm.r2_score(y, yhat)) if len(y) > 1 else float("nan"),
    }


# --------------------------------------------------------------------------- explainability
def permutation_importance(predict_fn: Callable[[pd.DataFrame], np.ndarray], df: pd.DataFrame,
                           y, features: list[str], metric_fn: Callable = skm.roc_auc_score,
                           n_repeats: int = 3, seed: int = 42, higher_is_better: bool = True) -> pd.DataFrame:
    """Model-agnostic permutation importance on *raw* input columns."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    base = metric_fn(y, predict_fn(df))
    rows = []
    for col in features:
        drops = []
        for _ in range(n_repeats):
            shuffled = df.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            score = metric_fn(y, predict_fn(shuffled))
            drops.append(base - score if higher_is_better else score - base)
        rows.append({"feature": col, "importance": float(np.mean(drops)), "std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
