"""Feature engineering and the sklearn preprocessing graph shared by all tabular models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from telco.config import Config

DERIVED_NUMERIC = ["revenue_per_gb", "complaint_rate"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-driven derived features (safe for missing values)."""
    out = df.copy()
    arpu = pd.to_numeric(out.get("arpu_idr"), errors="coerce")
    data = pd.to_numeric(out.get("data_usage_gb"), errors="coerce")
    tenure = pd.to_numeric(out.get("tenure_months"), errors="coerce")
    complaints = pd.to_numeric(out.get("num_complaints"), errors="coerce")
    out["revenue_per_gb"] = arpu / (data.fillna(0) + 1)
    out["complaint_rate"] = complaints / (tenure.fillna(0) / 12 + 1)
    return out


def feature_lists(cfg: Config, task: str = "churn") -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) model inputs for a task: 'churn' | 'arpu'."""
    feats = cfg["features"]
    numeric = list(feats["numeric"]) + DERIVED_NUMERIC
    categorical = list(feats["categorical"])
    if task == "arpu":
        exclude = set(feats.get("arpu_exclude", []))
        numeric = [c for c in numeric if c not in exclude]
        categorical = [c for c in categorical if c not in exclude]
    elif task != "churn":
        raise ValueError(f"Unknown task '{task}'")
    return numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([("num", num_pipe, numeric), ("cat", cat_pipe, categorical)],
                             verbose_feature_names_out=False)


def prepare_frame(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Select model inputs with stable dtypes; fails loudly on missing columns."""
    df = add_features(df)
    missing = [c for c in numeric + categorical if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")
    out = pd.DataFrame(index=df.index)
    for c in numeric:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    for c in categorical:
        col = df[c].astype(object)
        out[c] = col.where(col.notna(), "Unknown").astype(str).astype(object)
    return out
