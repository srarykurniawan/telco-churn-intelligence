"""Statistical foundations: hypothesis tests, effect sizes, A/B testing,
regression (OLS / Logit) and time-series diagnostics.

Every test returns a :class:`TestResult` with the statistic, p-value,
an effect size (statistical significance ≠ practical significance) and a
plain-language conclusion.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class TestResult:
    test: str
    statistic: float
    p_value: float
    alpha: float = 0.05
    effect_size: float | None = None
    effect_name: str | None = None
    effect_magnitude: str | None = None
    null_hypothesis: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def significant(self) -> bool:
        return bool(self.p_value < self.alpha)

    @property
    def conclusion(self) -> str:
        p = f"p = {self.p_value:.4g}"
        if self.significant:
            msg = f"Tolak H0 ({p} < α = {self.alpha}). Perbedaan/hubungan signifikan secara statistik."
        else:
            msg = f"Gagal menolak H0 ({p} ≥ α = {self.alpha}). Tidak cukup bukti adanya perbedaan/hubungan."
        if self.effect_size is not None:
            msg += f" Effect size {self.effect_name} = {self.effect_size:.3f} ({self.effect_magnitude})."
        return msg

    def to_dict(self) -> dict:
        d = asdict(self)
        d["significant"] = self.significant
        d["conclusion"] = self.conclusion
        return d


def _magnitude(value: float, cuts: tuple[float, float, float]) -> str:
    v = abs(value)
    if v < cuts[0]:
        return "sangat kecil"
    if v < cuts[1]:
        return "kecil"
    if v < cuts[2]:
        return "sedang"
    return "besar"


def _clean(x) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    return arr[~np.isnan(arr)]


# --------------------------------------------------------------------------- two-sample tests
def welch_ttest(a, b, alpha: float = 0.05) -> TestResult:
    """Welch's t-test (unequal variances) + Cohen's d."""
    a, b = _clean(a), _clean(b)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each group needs at least 2 observations.")
    t, p = stats.ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    d = (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0
    return TestResult("Welch t-test", float(t), float(p), alpha, float(d), "Cohen's d",
                      _magnitude(d, (0.2, 0.5, 0.8)), "Rata-rata kedua grup sama (μ1 = μ2)",
                      {"mean_a": a.mean(), "mean_b": b.mean(), "n_a": len(a), "n_b": len(b),
                       "diff": a.mean() - b.mean()})


def mann_whitney(a, b, alpha: float = 0.05) -> TestResult:
    """Non-parametric alternative to the t-test + rank-biserial correlation."""
    a, b = _clean(a), _clean(b)
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    r = 2 * u / (len(a) * len(b)) - 1
    return TestResult("Mann-Whitney U", float(u), float(p), alpha, float(r), "rank-biserial r",
                      _magnitude(r, (0.1, 0.3, 0.5)), "Distribusi kedua grup sama",
                      {"median_a": float(np.median(a)), "median_b": float(np.median(b))})


def chi_square_independence(x, y, alpha: float = 0.05) -> TestResult:
    """Chi-square test of independence + Cramér's V."""
    table = pd.crosstab(pd.Series(x, name="x"), pd.Series(y, name="y"))
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("Both variables need at least 2 categories.")
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return TestResult("Chi-square independence", float(chi2), float(p), alpha, float(v), "Cramér's V",
                      _magnitude(v, (0.1, 0.3, 0.5)), "Kedua variabel independen",
                      {"dof": int(dof), "min_expected": float(expected.min()),
                       "contingency": table})


def anova_oneway(values, groups, alpha: float = 0.05) -> TestResult:
    """One-way ANOVA + eta-squared."""
    df = pd.DataFrame({"v": pd.to_numeric(pd.Series(values), errors="coerce"), "g": pd.Series(groups)}).dropna()
    samples = [g["v"].to_numpy() for _, g in df.groupby("g") if len(g) > 1]
    if len(samples) < 2:
        raise ValueError("Need at least 2 groups.")
    f, p = stats.f_oneway(*samples)
    grand = df["v"].mean()
    ss_between = sum(len(s) * (s.mean() - grand) ** 2 for s in samples)
    ss_total = ((df["v"] - grand) ** 2).sum()
    eta2 = ss_between / ss_total if ss_total > 0 else 0.0
    return TestResult("One-way ANOVA", float(f), float(p), alpha, float(eta2), "η²",
                      _magnitude(eta2, (0.01, 0.06, 0.14)), "Rata-rata semua grup sama",
                      {"k_groups": len(samples), "group_means": df.groupby("g")["v"].mean().to_dict()})


def kruskal_wallis(values, groups, alpha: float = 0.05) -> TestResult:
    df = pd.DataFrame({"v": pd.to_numeric(pd.Series(values), errors="coerce"), "g": pd.Series(groups)}).dropna()
    samples = [g["v"].to_numpy() for _, g in df.groupby("g") if len(g) > 1]
    h, p = stats.kruskal(*samples)
    n, k = len(df), len(samples)
    eps2 = (h - k + 1) / (n - k) if n > k else 0.0
    return TestResult("Kruskal-Wallis H", float(h), float(p), alpha, float(max(eps2, 0.0)), "ε²",
                      _magnitude(eps2, (0.01, 0.06, 0.14)), "Distribusi semua grup sama",
                      {"group_medians": df.groupby("g")["v"].median().to_dict()})


def correlation_test(x, y, method: str = "pearson", alpha: float = 0.05) -> TestResult:
    df = pd.DataFrame({"x": pd.to_numeric(pd.Series(x), errors="coerce"),
                       "y": pd.to_numeric(pd.Series(y), errors="coerce")}).dropna()
    fn = stats.pearsonr if method == "pearson" else stats.spearmanr
    r, p = fn(df["x"], df["y"])
    return TestResult(f"{method.title()} correlation", float(r), float(p), alpha, float(r), "r",
                      _magnitude(r, (0.1, 0.3, 0.5)), "Tidak ada korelasi (ρ = 0)", {"n": len(df)})


# --------------------------------------------------------------------------- A/B testing
def two_proportion_ztest(x1: int, n1: int, x2: int, n2: int, alpha: float = 0.05) -> TestResult:
    """Two-sided z-test for p1 vs p2 with an unpooled CI for the difference and Cohen's h."""
    if min(n1, n2) <= 0:
        raise ValueError("Sample sizes must be positive.")
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se_pooled if se_pooled > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z))
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zc = stats.norm.ppf(1 - alpha / 2)
    diff = p1 - p2
    h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
    return TestResult("Two-proportion z-test", float(z), float(p), alpha, float(h), "Cohen's h",
                      _magnitude(h, (0.2, 0.5, 0.8)), "Proporsi kedua grup sama (p1 = p2)",
                      {"p1": p1, "p2": p2, "diff": diff, "ci_low": diff - zc * se, "ci_high": diff + zc * se,
                       "relative_lift": diff / p2 if p2 > 0 else np.nan, "n1": n1, "n2": n2})


def sample_size_two_proportions(p_base: float, mde_abs: float, alpha: float = 0.05,
                                power: float = 0.80) -> int:
    """Required sample size per group to detect an absolute change ``mde_abs``."""
    p1, p2 = p_base, p_base + mde_abs
    if not (0 < p1 < 1 and 0 < p2 < 1) or mde_abs == 0:
        raise ValueError("Proportions must lie in (0, 1) and MDE must be non-zero.")
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / mde_abs ** 2))


def bootstrap_diff_ci(a, b, statistic: Callable = np.mean, n_boot: int = 2000,
                      ci: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """Percentile bootstrap CI for statistic(a) - statistic(b)."""
    a, b = _clean(a), _clean(b)
    rng = np.random.default_rng(seed)
    boots = np.array([statistic(rng.choice(a, len(a))) - statistic(rng.choice(b, len(b)))
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(statistic(a) - statistic(b)), float(lo), float(hi)


# --------------------------------------------------------------------------- regression
def ols_regression(df: pd.DataFrame, formula: str):
    """OLS with heteroskedasticity-robust (HC3) standard errors."""
    import statsmodels.formula.api as smf

    return smf.ols(formula, data=df).fit(cov_type="HC3")


def logit_regression(df: pd.DataFrame, formula: str):
    import statsmodels.formula.api as smf

    return smf.logit(formula, data=df).fit(disp=False, maxiter=200)


def coefficient_table(result, odds_ratio: bool = False) -> pd.DataFrame:
    ci = result.conf_int()
    table = pd.DataFrame({
        "coef": result.params, "std_err": result.bse, "p_value": result.pvalues,
        "ci_low": ci[0], "ci_high": ci[1],
    })
    if odds_ratio:
        table["odds_ratio"] = np.exp(table["coef"])
        table["or_ci_low"] = np.exp(table["ci_low"])
        table["or_ci_high"] = np.exp(table["ci_high"])
    table["significant"] = table["p_value"] < 0.05
    return table


# --------------------------------------------------------------------------- time series
def adf_test(series, alpha: float = 0.05) -> TestResult:
    """Augmented Dickey-Fuller: H0 = series has a unit root (non-stationary)."""
    import warnings

    from statsmodels.tsa.stattools import adfuller

    x = _clean(series)
    with warnings.catch_warnings():  # statsmodels ≥0.15 warns about a future return type
        warnings.simplefilter("ignore", FutureWarning)
        res = adfuller(x, autolag="AIC")
    stat, p, lags, nobs, crit = res[0], res[1], res[2], res[3], res[4]
    return TestResult("Augmented Dickey-Fuller", float(stat), float(p), alpha,
                      null_hypothesis="Series memiliki unit root (tidak stasioner)",
                      details={"lags": int(lags), "nobs": int(nobs), "critical_values": crit})


def stl_decompose(series: pd.Series, period: int = 12) -> pd.DataFrame:
    from statsmodels.tsa.seasonal import STL

    res = STL(series, period=period, robust=True).fit()
    return pd.DataFrame({"observed": series, "trend": res.trend, "seasonal": res.seasonal,
                         "resid": res.resid})
