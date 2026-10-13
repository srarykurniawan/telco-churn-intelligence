"""Data & prediction drift monitoring (PSI, KS, chi-square) and performance tracking."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-4


def psi_numeric(reference, current, bins: int = 10) -> float:
    """Population Stability Index on reference-quantile bins.

    < 0.10 stable · 0.10–0.20 moderate shift · ≥ 0.20 significant shift.
    """
    ref = pd.to_numeric(pd.Series(reference), errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(pd.Series(current), errors="coerce").dropna().to_numpy()
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(reference, current) -> float:
    r = pd.Series(reference).astype(str).value_counts(normalize=True)
    c = pd.Series(current).astype(str).value_counts(normalize=True)
    cats = r.index.union(c.index)
    r = r.reindex(cats, fill_value=0).clip(lower=EPS)
    c = c.reindex(cats, fill_value=0).clip(lower=EPS)
    return float(np.sum((c - r) * np.log(c / r)))


def _status(psi: float, warning: float, threshold: float) -> str:
    if np.isnan(psi):
        return "N/A"
    if psi >= threshold:
        return "Drift"
    if psi >= warning:
        return "Warning"
    return "OK"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, numeric: list[str],
                 categorical: list[str], psi_warning: float = 0.1, psi_threshold: float = 0.2) -> pd.DataFrame:
    """Per-feature drift table. Status is PSI-based; KS/chi² p-values are reported for
    reference (with large samples they flag even tiny, practically irrelevant shifts)."""
    rows = []
    for col in numeric:
        if col not in reference or col not in current:
            continue
        ref, cur = reference[col].dropna(), current[col].dropna()
        psi = psi_numeric(ref, cur)
        ks = stats.ks_2samp(ref, cur) if len(ref) and len(cur) else None
        rows.append({"feature": col, "type": "numeric", "psi": psi,
                     "test": "KS", "statistic": ks.statistic if ks else np.nan,
                     "p_value": ks.pvalue if ks else np.nan,
                     "ref_mean": float(ref.mean()), "cur_mean": float(cur.mean()),
                     "status": _status(psi, psi_warning, psi_threshold)})
    for col in categorical:
        if col not in reference or col not in current:
            continue
        psi = psi_categorical(reference[col], current[col])
        r = reference[col].astype(str).value_counts()
        c = current[col].astype(str).value_counts()
        table = pd.concat([r, c], axis=1).fillna(0).to_numpy()
        try:
            chi2, p, _, _ = stats.chi2_contingency(table)
        except ValueError:
            chi2, p = np.nan, np.nan
        rows.append({"feature": col, "type": "categorical", "psi": psi, "test": "Chi²",
                     "statistic": chi2, "p_value": p, "ref_mean": np.nan, "cur_mean": np.nan,
                     "status": _status(psi, psi_warning, psi_threshold)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def prediction_drift(ref_scores, cur_scores) -> dict:
    ref, cur = np.asarray(ref_scores, dtype=float), np.asarray(cur_scores, dtype=float)
    return {"score_psi": psi_numeric(ref, cur), "ref_mean_score": float(ref.mean()),
            "cur_mean_score": float(cur.mean()),
            "ks_p_value": float(stats.ks_2samp(ref, cur).pvalue)}
