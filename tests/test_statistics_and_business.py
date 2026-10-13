import numpy as np
import pandas as pd
import pytest

from telco.analytics import business_metrics as bm
from telco.analytics import statistics as stx
from telco.data.generator import generate_customers

rng = np.random.default_rng(0)


def test_welch_detects_difference():
    res = stx.welch_ttest(rng.normal(1, 1, 500), rng.normal(0, 1, 500))
    assert res.significant and 0.8 < res.effect_size < 1.2


def test_welch_no_difference():
    assert not stx.welch_ttest(rng.normal(0, 1, 300), rng.normal(0, 1, 300), alpha=0.001).significant


def test_two_proportion_ztest_known_values():
    res = stx.two_proportion_ztest(200, 1000, 150, 1000)
    assert res.details["diff"] == pytest.approx(0.05)
    assert res.p_value < 0.01
    assert res.details["ci_low"] < 0.05 < res.details["ci_high"]


def test_sample_size_matches_reference():
    # classical result: p=0.20 → 0.15 at α=.05, power=.8 ≈ 900 per group
    n = stx.sample_size_two_proportions(0.20, -0.05)
    assert 850 < n < 950


def test_chi_square_and_anova():
    df = generate_customers(3000, seed=1)
    chi = stx.chi_square_independence(df["contract_type"], df["churn"])
    assert chi.significant and 0 <= chi.effect_size <= 1
    an = stx.anova_oneway(df["arpu_idr"], df["package"])
    assert an.significant and an.effect_size > 0.14


def test_adf_and_regression():
    assert stx.adf_test(rng.normal(size=200)).significant  # white noise is stationary
    df = generate_customers(2000, seed=2)
    res = stx.ols_regression(df, "np.log(arpu_idr) ~ data_usage_gb + C(package)")
    assert res.rsquared > 0.5
    logit = stx.logit_regression(df, "churn ~ num_complaints + tenure_months")
    table = stx.coefficient_table(logit, odds_ratio=True)
    assert table.loc["num_complaints", "odds_ratio"] > 1


def test_nps_exact():
    out = bm.nps([10, 9, 8, 7, 6, 0])
    assert out["nps"] == pytest.approx((2 / 6 - 2 / 6) * 100)
    assert bm.nps([10, 10])["nps"] == 100


def test_clv_and_hazard():
    h = bm.monthly_hazard_from_window(0.271, 3)
    assert h == pytest.approx(0.1, abs=1e-3)
    clv = bm.customer_lifetime_value([100_000], [0.271], margin=0.5, max_months=60)
    assert clv[0] == pytest.approx(100_000 * 0.5 * 10, rel=0.01)


def test_campaign_performance():
    df = generate_customers(20000, seed=4)
    cp = bm.campaign_performance(df, {"Discount": 25000, "Bonus Data": 12000, "Loyalty Points": 8000})
    assert set(cp["campaign"]) == {"Discount", "Bonus Data", "Loyalty Points"}
    disc = cp.set_index("campaign").loc["Discount"]
    assert disc["uplift_abs"] > 0 and disc["significant"]


def test_executive_kpis():
    k = bm.executive_kpis(generate_customers(1000, seed=1))
    assert k["customers"] == 1000 and k["arpu_idr"] > 0 and -100 <= k["nps_nps"] <= 100
    assert isinstance(bm.churn_by(pd.DataFrame({"g": ["a", "b"], "churn": [1, 0]}), "g"), pd.DataFrame)
