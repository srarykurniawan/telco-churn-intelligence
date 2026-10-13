"""Subscription-business metrics: churn, ARPU, usage, NPS, CLV and campaign KPIs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from telco.analytics.statistics import two_proportion_ztest

HEAVY_USER_GB = 50


# --------------------------------------------------------------------------- churn
def churn_rate(df: pd.DataFrame, target: str = "churn") -> float:
    return float(df[target].mean()) if len(df) else float("nan")


def churn_by(df: pd.DataFrame, col: str, target: str = "churn") -> pd.DataFrame:
    out = (df.groupby(col, observed=True)[target]
           .agg(customers="size", churners="sum", churn_rate="mean").reset_index())
    return out.sort_values("churn_rate", ascending=False)


def monthly_hazard_from_window(p_window, window_months: int = 3) -> np.ndarray:
    """Convert churn probability over a window (default 90 days) into a monthly hazard."""
    p = np.clip(np.asarray(p_window, dtype=float), 1e-6, 1 - 1e-6)
    return 1 - (1 - p) ** (1 / window_months)


# --------------------------------------------------------------------------- revenue / ARPU
def arpu(df: pd.DataFrame) -> float:
    """Average Revenue Per User = total revenue / subscribers."""
    return float(df["arpu_idr"].mean()) if len(df) else float("nan")


def arpu_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = (df.groupby(col, observed=True)
           .agg(customers=("arpu_idr", "size"), arpu=("arpu_idr", "mean"), revenue=("arpu_idr", "sum"))
           .reset_index())
    out["revenue_share"] = out["revenue"] / out["revenue"].sum()
    return out.sort_values("arpu", ascending=False)


def revenue_summary(df: pd.DataFrame, target: str = "churn") -> dict:
    monthly_revenue = float(df["arpu_idr"].sum())
    at_risk = float(df.loc[df[target] == 1, "arpu_idr"].sum()) if target in df else float("nan")
    return {
        "monthly_revenue_idr": monthly_revenue,
        "arpu_idr": arpu(df),
        "revenue_at_risk_monthly_idr": at_risk,
        "revenue_at_risk_annual_idr": at_risk * 12,
        "revenue_at_risk_pct": at_risk / monthly_revenue if monthly_revenue else float("nan"),
    }


def customer_lifetime_value(arpu_idr, churn_prob_window, margin: float = 0.55,
                            max_months: int = 60, window_months: int = 3) -> np.ndarray:
    """CLV = ARPU × gross margin × expected lifetime (1 / monthly hazard, capped)."""
    hazard = monthly_hazard_from_window(churn_prob_window, window_months)
    lifetime = np.minimum(1 / hazard, max_months)
    return np.asarray(arpu_idr, dtype=float) * margin * lifetime


# --------------------------------------------------------------------------- usage
def usage_summary(df: pd.DataFrame) -> dict:
    return {
        "avg_data_gb": float(df["data_usage_gb"].mean()),
        "median_data_gb": float(df["data_usage_gb"].median()),
        "avg_voice_min": float(df["voice_minutes"].mean()),
        "avg_sms": float(df["sms_count"].mean()),
        "heavy_user_share": float((df["data_usage_gb"] >= HEAVY_USER_GB).mean()),
        "declining_usage_share": float((df["usage_change_pct"] < -10).mean()),
        "five_g_share": float((df["device_type"] == "5G").mean()),
    }


def usage_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return (df.groupby(col, observed=True)
            .agg(customers=("data_usage_gb", "size"), avg_data_gb=("data_usage_gb", "mean"),
                 avg_voice_min=("voice_minutes", "mean"), avg_sms=("sms_count", "mean"),
                 avg_usage_change_pct=("usage_change_pct", "mean"))
            .reset_index())


# --------------------------------------------------------------------------- NPS
def nps_category(scores) -> pd.Series:
    s = pd.Series(scores)
    return pd.Series(np.select([s >= 9, s >= 7], ["Promoter", "Passive"], "Detractor"), index=s.index)


def nps(scores) -> dict:
    """Net Promoter Score = %Promoters (9-10) − %Detractors (0-6), range −100…100."""
    s = pd.to_numeric(pd.Series(scores), errors="coerce").dropna()
    if s.empty:
        return {"nps": float("nan"), "promoters_pct": 0.0, "passives_pct": 0.0, "detractors_pct": 0.0, "n": 0}
    prom, det = (s >= 9).mean(), (s <= 6).mean()
    return {"nps": float((prom - det) * 100), "promoters_pct": float(prom * 100),
            "passives_pct": float((1 - prom - det) * 100), "detractors_pct": float(det * 100), "n": int(len(s))}


def nps_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = [{col: k, **nps(g["nps_score"])} for k, g in df.groupby(col, observed=True)]
    return pd.DataFrame(rows).sort_values("nps", ascending=False)


# --------------------------------------------------------------------------- campaigns
def campaign_performance(df: pd.DataFrame, campaign_cost: dict[str, float], margin: float = 0.55,
                         horizon_months: int = 12, target: str = "churn",
                         control_label: str = "No Campaign") -> pd.DataFrame:
    """Evaluate each retention campaign against the randomised control group.

    Metrics: reach, take-up (conversion) rate, churn treated vs control,
    absolute & relative uplift with 95% CI and p-value, customers saved,
    campaign cost, incremental gross profit, ROI and cost per saved customer.
    """
    control = df[df["campaign_type"] == control_label]
    if control.empty:
        raise ValueError("No control group found.")
    c_churn, n_c = int(control[target].sum()), len(control)
    rows = []
    for camp, g in df[df["campaign_type"] != control_label].groupby("campaign_type"):
        n_t, t_churn = len(g), int(g[target].sum())
        test = two_proportion_ztest(c_churn, n_c, t_churn, n_t)  # control - treated = churn reduction
        uplift = test.details["diff"]
        saved = uplift * n_t
        cost = campaign_cost.get(camp, 0.0) * n_t
        incremental_gp = saved * g["arpu_idr"].mean() * margin * horizon_months
        rows.append({
            "campaign": camp, "treated": n_t, "control": n_c,
            "take_up_rate": float(g["campaign_accepted"].mean()) if "campaign_accepted" in g else np.nan,
            "churn_treated": t_churn / n_t, "churn_control": c_churn / n_c,
            "uplift_abs": uplift, "uplift_ci_low": test.details["ci_low"], "uplift_ci_high": test.details["ci_high"],
            "uplift_rel": uplift / (c_churn / n_c) if c_churn else np.nan,
            "p_value": test.p_value, "significant": test.significant,
            "customers_saved": saved, "cost_idr": cost, "incremental_gross_profit_idr": incremental_gp,
            "roi": (incremental_gp - cost) / cost if cost else np.nan,
            "cost_per_saved_idr": cost / saved if saved > 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values("roi", ascending=False).reset_index(drop=True)


def executive_kpis(df: pd.DataFrame, target: str = "churn") -> dict:
    out = {"customers": int(len(df)), "churn_rate": churn_rate(df, target)}
    out.update(revenue_summary(df, target))
    out.update({f"nps_{k}": v for k, v in nps(df["nps_score"]).items()})
    out.update(usage_summary(df))
    return out
