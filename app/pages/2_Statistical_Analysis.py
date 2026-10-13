import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make app/ importable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_common import load_customers, load_kpis, page_setup, show_result
from telco.analytics import statistics as stx

cfg = page_setup("Statistical Analysis", "🧪",
                 "Hypothesis testing · A/B testing & sample size · Regression · Correlation · Time-series diagnostics")
df = load_customers()
NUMERIC = cfg["features"]["numeric"]
CATEGORICAL = cfg["features"]["categorical"]
alpha = st.sidebar.select_slider("Significance level α", [0.01, 0.05, 0.10], value=0.05)

tabs = st.tabs(["Hypothesis Testing", "A/B Test & Sample Size", "Regression", "Correlation", "Time-series Tests"])

# --------------------------------------------------------------------------- hypothesis testing
with tabs[0]:
    st.markdown("Pilih **pertanyaan bisnis** siap pakai, atau rancang uji sendiri.")
    preset = st.selectbox("Pertanyaan", [
        "Apakah ARPU churner berbeda dengan non-churner?",
        "Apakah jumlah keluhan berhubungan dengan churn?",
        "Apakah ARPU berbeda antar region?",
        "Apakah pengguna 5G memakai data lebih banyak dari 4G?",
        "Custom test",
    ])
    if preset.startswith("Apakah ARPU churner"):
        a, b = df.loc[df.churn == 1, "arpu_idr"], df.loc[df.churn == 0, "arpu_idr"]
        st.markdown("**Parametrik** (Welch t-test) dan **non-parametrik** (Mann-Whitney) — ARPU skewed, jadi keduanya dilaporkan.")
        show_result(stx.welch_ttest(a, b, alpha))
        show_result(stx.mann_whitney(a, b, alpha))
        est, lo, hi = stx.bootstrap_diff_ci(a, b, np.median, n_boot=1000)
        st.caption(f"Bootstrap 95% CI selisih median: Rp {est:,.0f} [{lo:,.0f}; {hi:,.0f}]")
        st.plotly_chart(px.box(df, x="churn", y="arpu_idr", points=False, title="ARPU per status churn"), width="stretch")
    elif preset.startswith("Apakah jumlah keluhan"):
        x = df["num_complaints"].clip(upper=3).astype(str).replace("3", "3+")
        res = stx.chi_square_independence(x, df["churn"], alpha)
        show_result(res)
        ct = res.details["contingency"]
        st.dataframe((ct.div(ct.sum(axis=1), axis=0) * 100).round(1).rename(columns={0: "retain %", 1: "churn %"}),
                     width="stretch")
        st.caption(f"Expected count minimum = {res.details['min_expected']:.1f} (asumsi chi-square: ≥ 5).")
    elif preset.startswith("Apakah ARPU berbeda antar"):
        show_result(stx.anova_oneway(df["arpu_idr"], df["region"], alpha))
        show_result(stx.kruskal_wallis(df["arpu_idr"], df["region"], alpha))
        st.plotly_chart(px.box(df, x="region", y="arpu_idr", points=False, title="ARPU per region"), width="stretch")
    elif preset.startswith("Apakah pengguna 5G"):
        a, b = df.loc[df.device_type == "5G", "data_usage_gb"], df.loc[df.device_type == "4G", "data_usage_gb"]
        show_result(stx.welch_ttest(a, b, alpha))
        st.plotly_chart(px.histogram(df, x="data_usage_gb", color="device_type", barmode="overlay", log_x=False,
                                     nbins=80, opacity=0.6, title="Distribusi data usage"), width="stretch")
    else:
        kind = st.radio("Jenis uji", ["Numerik vs grup", "Kategorik vs kategorik"], horizontal=True)
        if kind == "Numerik vs grup":
            c1, c2 = st.columns(2)
            var = c1.selectbox("Variabel numerik", NUMERIC)
            grp = c2.selectbox("Grup", ["churn"] + CATEGORICAL)
            k = df[grp].nunique()
            if k == 2:
                levels = sorted(df[grp].unique())
                a, b = df.loc[df[grp] == levels[0], var], df.loc[df[grp] == levels[1], var]
                st.caption(f"2 grup ({levels[0]} vs {levels[1]}) → Welch t-test & Mann-Whitney")
                show_result(stx.welch_ttest(a, b, alpha))
                show_result(stx.mann_whitney(a, b, alpha))
            else:
                st.caption(f"{k} grup → One-way ANOVA & Kruskal-Wallis")
                show_result(stx.anova_oneway(df[var], df[grp], alpha))
                show_result(stx.kruskal_wallis(df[var], df[grp], alpha))
            st.plotly_chart(px.box(df, x=grp, y=var, points=False), width="stretch")
        else:
            c1, c2 = st.columns(2)
            v1 = c1.selectbox("Variabel 1", CATEGORICAL)
            v2 = c2.selectbox("Variabel 2", ["churn"] + [c for c in CATEGORICAL if c != v1])
            show_result(stx.chi_square_independence(df[v1], df[v2], alpha))

# --------------------------------------------------------------------------- A/B
with tabs[1]:
    left, right = st.columns(2)
    with left:
        st.subheader("Analisis hasil A/B test")
        st.caption("Default: churn grup kontrol vs grup yang menerima kampanye Discount (data aktual).")
        ctrl = df[df.campaign_type == "No Campaign"]
        trt = df[df.campaign_type == "Discount"]
        n1 = st.number_input("n kontrol", 1, value=len(ctrl))
        x1 = st.number_input("churn kontrol", 0, value=int(ctrl.churn.sum()))
        n2 = st.number_input("n treatment", 1, value=len(trt))
        x2 = st.number_input("churn treatment", 0, value=int(trt.churn.sum()))
        if x1 > n1 or x2 > n2:
            st.error("Jumlah event tidak boleh melebihi n.")
        else:
            res = stx.two_proportion_ztest(x1, n1, x2, n2, alpha)
            show_result(res)
            d = res.details
            st.markdown(f"Churn kontrol **{d['p1']:.2%}** vs treatment **{d['p2']:.2%}** → selisih "
                        f"**{d['diff'] * 100:.2f} pp** (CI {1 - alpha:.0%}: {d['ci_low'] * 100:.2f} … "
                        f"{d['ci_high'] * 100:.2f} pp), relative lift {d['relative_lift']:.1%}.")
    with right:
        st.subheader("Kalkulator sample size")
        base = st.slider("Baseline churn rate", 0.02, 0.50, 0.20, 0.01)
        mde = st.slider("Minimum detectable effect (pp, penurunan)", 0.5, 10.0, 2.0, 0.5) / 100
        power = st.select_slider("Power (1 − β)", [0.7, 0.8, 0.9, 0.95], value=0.8)
        n = stx.sample_size_two_proportions(base, -mde, alpha, power)
        st.metric("Sampel per grup", f"{n:,}".replace(",", "."), f"total {2 * n:,}".replace(",", "."),
                  delta_color="off", border=True)
        mdes = np.arange(0.5, 6.01, 0.25) / 100
        curve = pd.DataFrame({"MDE (pp)": mdes * 100,
                              "n per grup": [stx.sample_size_two_proportions(base, -m, alpha, power) for m in mdes]})
        fig = px.line(curve, x="MDE (pp)", y="n per grup", log_y=True, markers=True,
                      title="Trade-off efek minimum vs ukuran sampel")
        st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- regression
with tabs[2]:
    kind = st.radio("Model", ["OLS — log(ARPU)", "Logit — churn (odds ratio)"], horizontal=True)
    if kind.startswith("OLS"):
        default = ["data_usage_gb", "voice_minutes", "tenure_months", "C(package)", "C(plan_type)", "C(has_streaming_addon)"]
        options = [c for c in NUMERIC if c != "arpu_idr"] + [f"C({c})" for c in CATEGORICAL]
        terms = st.multiselect("Prediktor", options, default=default)
        formula = "np.log(arpu_idr) ~ " + (" + ".join(terms) if terms else "1")
    else:
        default = ["num_complaints", "network_quality_score", "tenure_months", "nps_score",
                   "usage_change_pct", "C(contract_type)", "C(campaign_type)"]
        options = NUMERIC + [f"C({c})" for c in CATEGORICAL]
        terms = st.multiselect("Prediktor", options, default=default)
        formula = "churn ~ " + (" + ".join(terms) if terms else "1")
    st.code(formula, language="r")
    try:
        if kind.startswith("OLS"):
            res = stx.ols_regression(df, formula)
            c = st.columns(4)
            c[0].metric("R²", f"{res.rsquared:.3f}", border=True)
            c[1].metric("Adj. R²", f"{res.rsquared_adj:.3f}", border=True)
            c[2].metric("AIC", f"{res.aic:,.0f}", border=True)
            c[3].metric("n", f"{int(res.nobs):,}", border=True)
            tab = stx.coefficient_table(res)
            st.caption("SE robust HC3. Koefisien pada log(ARPU): exp(β) − 1 ≈ % perubahan ARPU.")
        else:
            res = stx.logit_regression(df, formula)
            c = st.columns(4)
            c[0].metric("Pseudo R²", f"{res.prsquared:.3f}", border=True)
            c[1].metric("Log-likelihood", f"{res.llf:,.0f}", border=True)
            c[2].metric("AIC", f"{res.aic:,.0f}", border=True)
            c[3].metric("LLR p-value", f"{res.llr_pvalue:.2e}", border=True)
            tab = stx.coefficient_table(res, odds_ratio=True)
            st.caption("Odds ratio > 1 → meningkatkan peluang churn; < 1 → protektif.")
        plot = tab.drop(index="Intercept", errors="ignore").reset_index(names="term")
        xcol = "odds_ratio" if "odds_ratio" in plot else "coef"
        lo, hi = ("or_ci_low", "or_ci_high") if xcol == "odds_ratio" else ("ci_low", "ci_high")
        fig = go.Figure(go.Scatter(x=plot[xcol], y=plot["term"], mode="markers",
                                   marker=dict(color=np.where(plot["significant"], "#C8102E", "#8C8C8C"), size=9),
                                   error_x=dict(type="data", symmetric=False, array=plot[hi] - plot[xcol],
                                                arrayminus=plot[xcol] - plot[lo])))
        fig.add_vline(x=1 if xcol == "odds_ratio" else 0, line_dash="dot")
        fig.update_layout(title=f"{xcol} dengan 95% CI (merah = signifikan)", height=max(350, 28 * len(plot)))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(tab.style.format("{:.4f}", subset=[c for c in tab.columns if c != "significant"]),
                     width="stretch")
        with st.expander("Ringkasan statsmodels lengkap"):
            st.text(res.summary().as_text())
    except Exception as exc:  # noqa: BLE001
        st.error(f"Model gagal diestimasi: {exc}")

# --------------------------------------------------------------------------- correlation
with tabs[3]:
    method = st.radio("Metode", ["spearman", "pearson"], horizontal=True)
    cols = NUMERIC + ["churn"]
    corr = df[cols].corr(method=method)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                    title=f"Matriks korelasi ({method})")
    fig.update_layout(height=620)
    st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    x = c1.selectbox("X", NUMERIC, index=NUMERIC.index("nps_score"))
    y = c2.selectbox("Y", NUMERIC, index=NUMERIC.index("num_complaints"))
    show_result(stx.correlation_test(df[x], df[y], method, alpha))

# --------------------------------------------------------------------------- time series
with tabs[4]:
    kpi = load_kpis()
    target = st.selectbox("Series", ["revenue_idr_bn", "active_subscribers", "churn_rate", "arpu_idr", "nps"])
    s = kpi.set_index("month")[target]
    left, right = st.columns(2)
    with left:
        st.markdown("**Level**")
        show_result(stx.adf_test(s, alpha))
    with right:
        st.markdown("**First difference**")
        show_result(stx.adf_test(s.diff().dropna(), alpha))
    st.caption("ADF: H0 = unit root. Series dengan trend biasanya tidak stasioner di level tetapi stasioner setelah differencing → "
               "alasan model forecasting memodelkan growth (Δlog), bukan level.")
    dec = stx.stl_decompose(s, 12)
    fig = go.Figure()
    for i, comp in enumerate(["observed", "trend", "seasonal", "resid"]):
        fig.add_scatter(x=dec.index, y=dec[comp], name=comp, yaxis=f"y{i + 1 if i else ''}")
    fig.update_layout(height=640, title="STL decomposition (period = 12)", showlegend=True,
                      yaxis=dict(domain=[0.78, 1]), yaxis2=dict(domain=[0.53, 0.75]),
                      yaxis3=dict(domain=[0.27, 0.50]), yaxis4=dict(domain=[0, 0.24]))
    st.plotly_chart(fig, width="stretch")
