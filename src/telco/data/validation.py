"""Schema & quality validation for customer data (training and scoring)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from telco.config import Config

VALUE_RANGES: dict[str, tuple[float, float]] = {
    "tenure_months": (0, 240),
    "age": (10, 100),
    "monthly_charge_idr": (0, 5_000_000),
    "arpu_idr": (0, 10_000_000),
    "data_usage_gb": (0, 2_000),
    "voice_minutes": (0, 20_000),
    "sms_count": (0, 10_000),
    "network_quality_score": (0, 10),
    "avg_call_drop_rate": (0, 100),
    "num_complaints": (0, 100),
    "days_since_last_topup": (0, 365),
    "usage_change_pct": (-100, 1_000),
    "nps_score": (0, 10),
}

KNOWN_CATEGORIES: dict[str, set[str]] = {
    "plan_type": {"Prepaid", "Postpaid"},
    "package": {"Basic", "Standard", "Premium", "Unlimited"},
    "device_type": {"4G", "5G"},
    "has_streaming_addon": {"Yes", "No"},
    "campaign_type": {"No Campaign", "Discount", "Bonus Data", "Loyalty Points"},
}


class DataValidationError(ValueError):
    """Raised when data fails hard validation rules."""


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_rows: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            raise DataValidationError("; ".join(self.errors))


def coerce_types(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return a copy with numeric columns as float and categoricals as clean strings."""
    out = df.copy()
    feats = cfg["features"]
    for col in feats["numeric"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    for col in feats["categorical"]:
        if col in out.columns:
            out[col] = out[col].astype(object).where(out[col].notna(), "Unknown").astype(str).str.strip()
    target = feats["target"]
    if target in out.columns:
        out[target] = pd.to_numeric(out[target], errors="coerce")
    return out


def validate_customers(df: pd.DataFrame, cfg: Config, require_target: bool = True) -> ValidationReport:
    feats = cfg["features"]
    report = ValidationReport(n_rows=len(df))
    if df.empty:
        report.errors.append("Dataset kosong.")
        return report

    required = [feats["id_col"], *feats["numeric"], *feats["categorical"]]
    if require_target:
        required.append(feats["target"])
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.errors.append(f"Kolom wajib tidak ada: {missing}")
        return report

    for col in feats["numeric"]:
        coerced = pd.to_numeric(df[col], errors="coerce")
        bad_type = coerced.isna().sum() - df[col].isna().sum()
        if bad_type > 0.05 * len(df):
            report.errors.append(f"'{col}': {bad_type} nilai tidak bisa dikonversi ke numerik.")
        null_rate = coerced.isna().mean()
        if null_rate > 0.20:
            report.errors.append(f"'{col}': missing rate {null_rate:.0%} (> 20%).")
        elif null_rate > 0:
            report.warnings.append(f"'{col}': missing rate {null_rate:.1%} — akan diimputasi median.")
        if col in VALUE_RANGES:
            lo, hi = VALUE_RANGES[col]
            out_of_range = ((coerced < lo) | (coerced > hi)).sum()
            if out_of_range:
                report.warnings.append(f"'{col}': {out_of_range} nilai di luar rentang wajar [{lo}, {hi}].")

    for col, known in KNOWN_CATEGORIES.items():
        unknown = set(df[col].dropna().astype(str).unique()) - known
        if unknown:
            report.warnings.append(f"'{col}': kategori baru {sorted(unknown)[:5]} (di-encode sebagai 'unknown').")

    if require_target:
        target = pd.to_numeric(df[feats["target"]], errors="coerce")
        if not set(target.dropna().unique()).issubset({0, 1}):
            report.errors.append(f"Target '{feats['target']}' harus biner 0/1.")
        if target.isna().any():
            report.errors.append(f"Target '{feats['target']}' memiliki nilai kosong.")
        elif target.nunique() < 2:
            report.errors.append("Target hanya memiliki satu kelas.")

    dup = df[feats["id_col"]].duplicated().sum()
    if dup:
        report.warnings.append(f"{dup} customer_id duplikat.")
    if len(df) < 200 and require_target:
        report.warnings.append("Data latih < 200 baris — model mungkin tidak stabil.")
    return report


def numeric_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Compact profile used by data versioning."""
    present = [c for c in cols if c in df.columns]
    desc = df[present].apply(pd.to_numeric, errors="coerce").describe().T
    desc["missing_pct"] = df[present].isna().mean().values * 100
    return desc.replace({np.nan: None})
