"""Synthetic but realistic Indonesian telco data.

Two datasets are produced:

1. ``customers`` — one row per subscriber snapshot with demographics, plan,
   usage, network experience, NPS, retention-campaign assignment (a randomised
   controlled trial, so campaign uplift is causally measurable) and the
   label ``churn`` = churned within the next 90 days.
2. ``monthly_kpi`` — company-level monthly KPIs (subscribers, churn rate,
   ARPU, revenue, data traffic, NPS) with trend, seasonality and the
   Lebaran (Idul Fitri) effect, for time-series forecasting.

All latent randomness is drawn once and features are computed
deterministically from it. This lets us apply a *drift scenario* on top of
the very same customers, while keeping the churn-model intercept calibrated
on the un-drifted population — so drift genuinely changes the churn rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats

REGIONS = ["Jawa", "Sumatera", "Kalimantan", "Sulawesi", "Bali-Nusra", "Papua-Maluku"]
REGION_PROBS = [0.55, 0.21, 0.07, 0.08, 0.05, 0.04]
REGION_NETWORK_QUALITY = {
    "Jawa": 7.9, "Sumatera": 7.2, "Kalimantan": 6.7,
    "Sulawesi": 6.8, "Bali-Nusra": 7.4, "Papua-Maluku": 5.6,
}
PACKAGES = ["Basic", "Standard", "Premium", "Unlimited"]
PACKAGE_RANK = {p: i for i, p in enumerate(PACKAGES)}
PACKAGE_PROBS = {"Prepaid": [0.35, 0.35, 0.20, 0.10], "Postpaid": [0.15, 0.35, 0.30, 0.20]}
PACKAGE_PRICE_IDR = {
    "Prepaid": {"Basic": 35_000, "Standard": 65_000, "Premium": 110_000, "Unlimited": 160_000},
    "Postpaid": {"Basic": 75_000, "Standard": 125_000, "Premium": 200_000, "Unlimited": 300_000},
}
DATA_QUOTA_GB = {"Basic": 10, "Standard": 25, "Premium": 50, "Unlimited": np.inf}
BASE_DATA_USAGE_GB = {"Basic": 6, "Standard": 16, "Premium": 35, "Unlimited": 60}
CONTRACT_EFFECT = {"No Contract": 0.0, "Month-to-Month": 0.45, "12 Months": -0.35, "24 Months": -0.80}
CONTROL_LABEL = "No Campaign"  # NB: not "None" — pandas parses "None" as NaN
CAMPAIGNS = ["Discount", "Bonus Data", "Loyalty Points"]
CAMPAIGN_PROBS = [0.40, 0.35, 0.25]
CAMPAIGN_EFFECT = {"Discount": -0.55, "Bonus Data": -0.40, "Loyalty Points": -0.25}
CAMPAIGN_TAKEUP = {"Discount": 0.42, "Bonus Data": 0.55, "Loyalty Points": 0.30}

# Idul Fitri months — data traffic & SIM activations spike around Lebaran.
LEBARAN_MONTHS = {
    "2021-05", "2022-05", "2023-04", "2024-04", "2025-03", "2026-03",
    "2027-03", "2028-02", "2029-02", "2030-02",
}

CUSTOMER_COLUMNS = [
    "customer_id", "region", "plan_type", "package", "contract_type", "payment_method",
    "device_type", "has_streaming_addon", "tenure_months", "age", "monthly_charge_idr",
    "arpu_idr", "data_usage_gb", "voice_minutes", "sms_count", "network_quality_score",
    "avg_call_drop_rate", "num_complaints", "days_since_last_topup", "usage_change_pct",
    "nps_score", "campaign_type", "campaign_accepted", "churn",
]


@dataclass(frozen=True)
class DriftSpec:
    """Describes how a new population differs from the training population."""

    network_quality_shift: float = 0.0
    price_multiplier: float = 1.0
    usage_multiplier: float = 1.0
    complaint_multiplier: float = 1.0
    prepaid_share: float = 0.78
    churn_intercept_shift: float = 0.0


DRIFT_SCENARIOS: dict[str, DriftSpec] = {
    "none": DriftSpec(),
    "mild": DriftSpec(network_quality_shift=-0.5, price_multiplier=1.05, usage_multiplier=1.10,
                      complaint_multiplier=1.25, churn_intercept_shift=0.15),
    "severe": DriftSpec(network_quality_shift=-1.6, price_multiplier=1.20, usage_multiplier=1.40,
                        complaint_multiplier=1.9, prepaid_share=0.90, churn_intercept_shift=0.45),
}


def _pick(u: np.ndarray, categories: list[str], probs: list[float]) -> np.ndarray:
    cum = np.cumsum(probs) / np.sum(probs)
    idx = np.minimum(np.searchsorted(cum, u, side="right"), len(categories) - 1)
    return np.asarray(categories, dtype=object)[idx]


def _map(values: np.ndarray, mapping: dict, dtype=float) -> np.ndarray:
    """Vectorised dict lookup (np.vectorize infers dtype from the first element)."""
    return np.array([mapping[v] for v in values], dtype=dtype)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _draw_latents(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    uniforms = ["region", "plan", "package", "contract", "payment", "device", "streaming",
                "treat", "ctype", "takeup", "sms", "complaints", "topup", "churn"]
    normals = ["tenure", "age", "data", "voice", "nq", "drop", "usage_change", "nps", "arpu"]
    latents = {k: rng.random(n) for k in uniforms}
    latents.update({k: rng.standard_normal(n) for k in normals})
    return latents


def _build_features(d: dict[str, np.ndarray], drift: DriftSpec, treatment_rate: float) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute customer features and the churn logit (without intercept)."""
    n = len(d["region"])
    region = _pick(d["region"], REGIONS, REGION_PROBS)
    plan = np.where(d["plan"] < drift.prepaid_share, "Prepaid", "Postpaid").astype(object)
    is_pre = plan == "Prepaid"
    package = np.where(is_pre, _pick(d["package"], PACKAGES, PACKAGE_PROBS["Prepaid"]),
                       _pick(d["package"], PACKAGES, PACKAGE_PROBS["Postpaid"]))
    rank = _map(package, PACKAGE_RANK)
    contract = np.where(is_pre, "No Contract",
                        _pick(d["contract"], ["Month-to-Month", "12 Months", "24 Months"], [0.5, 0.3, 0.2]))
    payment = np.where(is_pre,
                       _pick(d["payment"], ["Pulsa Reload", "E-Wallet", "Retail Outlet"], [0.5, 0.35, 0.15]),
                       _pick(d["payment"], ["Auto-Debit", "Virtual Account", "Credit Card", "E-Wallet"],
                             [0.35, 0.30, 0.15, 0.20]))
    device = np.where(d["device"] < 0.15 + 0.12 * rank, "5G", "4G")
    streaming = d["streaming"] < 0.08 + 0.08 * rank

    tenure = np.clip(np.round(np.exp(np.where(is_pre, np.log(16), np.log(30)) + 0.85 * d["tenure"])), 1, 120)
    age = np.clip(np.round(32 + 10 * d["age"]), 17, 70)
    base_data = _map(package, BASE_DATA_USAGE_GB)
    data_gb = (base_data * np.exp(0.45 * d["data"]) * np.where(device == "5G", 1.25, 1.0)
               * np.clip(1 + (32 - age) * 0.008, 0.6, 1.4) * drift.usage_multiplier)
    voice = np.round(110 * np.exp(0.6 * d["voice"]) * np.clip(1 + (age - 32) * 0.01, 0.5, 1.8))
    sms = stats.poisson.ppf(d["sms"], 14).astype(int)
    nq = np.clip(_map(region, REGION_NETWORK_QUALITY) + 1.0 * d["nq"] + drift.network_quality_shift, 1, 10)
    drop = np.clip(0.6 + (10 - nq) * 0.35 + 0.4 * d["drop"], 0.0, 15.0)
    lam = (0.25 + 0.22 * np.clip(8 - nq, 0, None) + 0.05 * drop) * drift.complaint_multiplier
    complaints = stats.poisson.ppf(d["complaints"], lam).astype(int)
    topup = np.where(is_pre, np.clip(-np.log(1 - d["topup"]) * 9, 0, 90), d["topup"] * 30)
    usage_change = np.clip(-2 + 14 * d["usage_change"] - 3 * complaints, -80, 80)

    price = np.array([PACKAGE_PRICE_IDR[p][k] for p, k in zip(plan, package, strict=False)], dtype=float) * drift.price_multiplier
    quota = _map(package, DATA_QUOTA_GB)
    overage_gb = np.where(np.isinf(quota), 0.0, np.clip(data_gb - quota, 0, None))
    arpu = (price + 30_000 * streaming + overage_gb * 4_000 + np.clip(voice - 200, 0, None) * 150
            + 4_000 * d["arpu"])
    arpu = np.round(np.maximum(arpu, 0.8 * price), -2)

    nps = np.clip(np.round(8.4 - 0.7 * complaints + 0.35 * (nq - 7) + 0.012 * np.minimum(tenure, 60)
                           - 0.15 * drop + 1.6 * d["nps"]), 0, 10).astype(int)

    treated = d["treat"] < treatment_rate
    ctype = np.where(treated, _pick(d["ctype"], CAMPAIGNS, CAMPAIGN_PROBS), CONTROL_LABEL)
    takeup_p = np.array([CAMPAIGN_TAKEUP.get(c, 0.0) for c in ctype])
    accepted = treated & (d["takeup"] < takeup_p)
    campaign_effect = np.array([CAMPAIGN_EFFECT.get(c, 0.0) for c in ctype]) * np.where(accepted, 1.0, 0.35)

    logit = (0.42 * complaints - 0.22 * (nq - 7) - 0.025 * np.minimum(tenure, 60) + 0.55 * is_pre
             + _map(contract, CONTRACT_EFFECT) - 0.03 * usage_change
             + 0.035 * topup * is_pre - 0.20 * (nps - 7) + 0.12 * drop - 0.35 * streaming
             + 0.25 * (age < 25) + 0.6 * np.log(drift.price_multiplier) + campaign_effect)

    df = pd.DataFrame({
        "region": region, "plan_type": plan, "package": package, "contract_type": contract,
        "payment_method": payment, "device_type": device,
        "has_streaming_addon": np.where(streaming, "Yes", "No"),
        "tenure_months": tenure.astype(int), "age": age.astype(int),
        "monthly_charge_idr": np.round(price, -2), "arpu_idr": arpu,
        "data_usage_gb": np.round(data_gb, 2), "voice_minutes": voice.astype(int), "sms_count": sms,
        "network_quality_score": np.round(nq, 1), "avg_call_drop_rate": np.round(drop, 2),
        "num_complaints": complaints, "days_since_last_topup": np.round(topup).astype(int),
        "usage_change_pct": np.round(usage_change, 1), "nps_score": nps,
        "campaign_type": ctype, "campaign_accepted": accepted.astype(int),
    })
    assert len(df) == n
    return df, logit


def solve_intercept(logit: np.ndarray, target_rate: float) -> float:
    """Find the intercept so that mean(sigmoid(b + logit)) == target_rate."""
    return float(optimize.brentq(lambda b: _sigmoid(b + logit).mean() - target_rate, -20, 20))


def generate_customers(
    n: int = 20_000,
    seed: int = 42,
    target_churn_rate: float = 0.20,
    treatment_rate: float = 0.35,
    drift: DriftSpec | str | None = None,
    id_prefix: str = "CUST",
) -> pd.DataFrame:
    """Generate a customer snapshot. ``drift`` may be a DriftSpec or a scenario name."""
    if isinstance(drift, str):
        if drift not in DRIFT_SCENARIOS:
            raise ValueError(f"Unknown drift scenario '{drift}'. Options: {list(DRIFT_SCENARIOS)}")
        drift = DRIFT_SCENARIOS[drift]
    drift = drift or DriftSpec()
    rng = np.random.default_rng(seed)
    latents = _draw_latents(rng, n)

    _, base_logit = _build_features(latents, DriftSpec(), treatment_rate)
    intercept = solve_intercept(base_logit, target_churn_rate)

    df, logit = _build_features(latents, drift, treatment_rate)
    p = _sigmoid(intercept + logit + drift.churn_intercept_shift)
    df["churn"] = (latents["churn"] < p).astype(int)
    df.insert(0, "customer_id", [f"{id_prefix}-{i:06d}" for i in range(1, n + 1)])
    return df[CUSTOMER_COLUMNS]


def _lebaran_flags(months: pd.PeriodIndex) -> tuple[np.ndarray, np.ndarray]:
    keys = months.strftime("%Y-%m")
    leb = np.array([k in LEBARAN_MONTHS for k in keys], dtype=float)
    post = np.array([(m - 1).strftime("%Y-%m") in LEBARAN_MONTHS for m in months], dtype=float)
    return leb, post


def generate_monthly_kpis(start: str = "2021-10", n_months: int = 60, seed: int = 42) -> pd.DataFrame:
    """Company-level monthly KPIs with trend, yearly seasonality and Lebaran effect."""
    rng = np.random.default_rng(seed + 1)
    months = pd.period_range(start=start, periods=n_months, freq="M")
    t = np.arange(n_months)
    moy = months.month.to_numpy()
    leb, post = _lebaran_flags(months)

    season = 1 + 0.025 * (moy == 12) - 0.015 * (moy == 1) - 0.01 * (moy == 2) + 0.06 * leb - 0.012 * post
    active = 12.0e6 * (1 + 0.0055 * t) * (1 + 0.015 * leb) * (1 + rng.normal(0, 0.003, n_months))
    churn_rate = np.clip(0.024 + 0.004 * post + 0.002 * (moy == 1) - 0.00008 * t
                         + rng.normal(0, 0.0012, n_months), 0.01, 0.06)
    churned = active * churn_rate
    new = np.empty(n_months)
    new[0] = churned[0] * 1.15
    new[1:] = np.diff(active) + churned[1:]
    arpu = 46_000 * (1 + 0.0035 * t) * season + rng.normal(0, 350, n_months)
    revenue_bn = active * arpu / 1e9
    avg_gb = 11.0 * (1 + 0.012 * t) * (1 + 0.08 * leb) * (1 + rng.normal(0, 0.01, n_months))
    traffic_pb = active * avg_gb / 1e6
    nps = 28 + 0.1 * t - 3.0 * post + rng.normal(0, 2.0, n_months)

    return pd.DataFrame({
        "month": months.to_timestamp(),
        "active_subscribers": np.round(active).astype(int),
        "new_subscribers": np.round(np.maximum(new, 0)).astype(int),
        "churned_subscribers": np.round(churned).astype(int),
        "churn_rate": np.round(churn_rate, 5),
        "arpu_idr": np.round(arpu, 0),
        "revenue_idr_bn": np.round(revenue_bn, 2),
        "avg_data_per_user_gb": np.round(avg_gb, 2),
        "data_traffic_pb": np.round(traffic_pb, 2),
        "nps": np.round(nps, 1),
        "is_lebaran": leb.astype(int),
    })
