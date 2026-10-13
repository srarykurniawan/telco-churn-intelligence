import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make app/ importable

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_common import idr, load_customers, load_kpis, num, page_setup, pct, sidebar_filters
from telco.analytics import business_metrics as bm

cfg = page_setup("Business Metrics", "📊", "Churn · ARPU & revenue · Usage · NPS · Campaign performance")
df = sidebar_filters(load_customers(), "bm")
biz = cfg["business"]

tabs = st.tabs(["Churn", "ARPU & Revenue", "Usage", "NPS", "Campaign", "Tren KPI bulanan"])

# --------------------------------------------------------------------------- churn
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Pelanggan", num(len(df)), border=True)
    c[1].metric("Churn rate (90 hari)", pct(bm.churn_rate(df)), border=True)
    c[2].metric("Estimasi churn bulanan", pct(float(bm.monthly_hazard_from_window(bm.churn_rate(df)))), border=True)
    c[3].metric("Churners", num(df["churn"].sum()), border=True)
    st.caption("Churn bulanan dikonversi dari probabilitas 90 hari: h = 1 − (1 − p)^(1/3).")
    dim = st.selectbox("Breakdown churn berdasarkan", ["plan_type", "contract_type", "package", "region",
                                                       "payment_method", "device_type", "has_streaming_addon"])
    cb = bm.churn_by(df, dim)
    left, right = st.columns(2)
    fig = px.bar(cb, x=dim, y="churn_rate", text=cb["churn_rate"].map(pct), title=f"Churn rate per {dim}")
    fig.add_hline(y=bm.churn_rate(df), line_dash="dot", annotation_text="rata-rata")
    fig.update_layout(yaxis_tickformat=".0%")
    left.plotly_chart(fig, width="stretch")
    df_c = df.assign(complaints=df["num_complaints"].clip(upper=4).astype(int).astype(str).replace("4", "4+"))
    cc = bm.churn_by(df_c, "complaints").sort_values("complaints")
    fig = px.line(cc, x="complaints", y="churn_rate", markers=True, title="Churn vs jumlah keluhan")
    fig.update_layout(yaxis_tickformat=".0%")
    right.plotly_chart(fig, width="stretch")
    df_t = df.assign(tenure_band=(df["tenure_months"] // 6 * 6).clip(upper=60))
    tb = df_t.groupby("tenure_band")["churn"].mean().reset_index()
    fig = px.area(tb, x="tenure_band", y="churn", title="Churn per tenure (bulan) — risiko tertinggi di awal lifecycle")
    fig.update_layout(yaxis_tickformat=".0%", xaxis_title="tenure (bulan, bin 6)")
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- ARPU
with tabs[1]:
    rs = bm.revenue_summary(df)
    c = st.columns(4)
    c[0].metric("ARPU", idr(rs["arpu_idr"], short=False), border=True)
    c[1].metric("Revenue bulanan (sampel)", idr(rs["monthly_revenue_idr"]), border=True)
    c[2].metric("Revenue at risk / bulan", idr(rs["revenue_at_risk_monthly_idr"]), border=True)
    c[3].metric("Revenue at risk / tahun", idr(rs["revenue_at_risk_annual_idr"]), pct(rs["revenue_at_risk_pct"]),
                delta_color="inverse", border=True)
    dim = st.selectbox("Breakdown ARPU", ["package", "plan_type", "region", "device_type"], key="arpu_dim")
    ab = bm.arpu_by(df, dim)
    left, right = st.columns(2)
    fig = px.bar(ab, x=dim, y="arpu", text=ab["arpu"].map(lambda v: idr(v, False)), title=f"ARPU per {dim}")
    left.plotly_chart(fig, width="stretch")
    fig = px.pie(ab, names=dim, values="revenue", hole=0.55, title="Kontribusi revenue")
    right.plotly_chart(fig, width="stretch")
    fig = px.histogram(df, x="arpu_idr", color="plan_type", nbins=60, barmode="overlay", opacity=0.7,
                       title="Distribusi ARPU (right-skewed → gunakan median/log untuk analisis)")
    st.plotly_chart(fig, width="stretch")
    clv = bm.customer_lifetime_value(df["arpu_idr"], df["churn"].mean(), biz["gross_margin"], biz["max_lifetime_months"])
    st.info(f"**CLV rata-rata** (ARPU × margin {biz['gross_margin']:.0%} × lifetime 1/h) ≈ **{idr(clv.mean())}** "
            "per pelanggan. Di halaman *Churn & ARPU Models*, CLV dihitung per pelanggan memakai probabilitas churn model.")

# --------------------------------------------------------------------------- usage
with tabs[2]:
    us = bm.usage_summary(df)
    c = st.columns(5)
    c[0].metric("Rata-rata data", f"{us['avg_data_gb']:.1f} GB", border=True)
    c[1].metric("Median data", f"{us['median_data_gb']:.1f} GB", border=True)
    c[2].metric("Voice", f"{us['avg_voice_min']:.0f} menit", border=True)
    c[3].metric("Heavy users (≥50 GB)", pct(us["heavy_user_share"]), border=True)
    c[4].metric("Usage menurun >10%", pct(us["declining_usage_share"]), delta_color="inverse", border=True)
    left, right = st.columns(2)
    ub = bm.usage_by(df, "package")
    fig = px.bar(ub, x="package", y="avg_data_gb", color="package", title="Rata-rata data usage per paket (GB)")
    left.plotly_chart(fig, width="stretch")
    df_u = df.assign(usage_trend=df["usage_change_pct"].clip(-40, 40) // 10 * 10)
    ut = df_u.groupby("usage_trend")["churn"].mean().reset_index()
    fig = px.bar(ut, x="usage_trend", y="churn", title="Churn vs perubahan usage 3 bulan (%) — early warning signal")
    fig.update_layout(yaxis_tickformat=".0%")
    right.plotly_chart(fig, width="stretch")
    samp = df.sample(min(3000, len(df)), random_state=1)
    fig = px.scatter(samp, x="data_usage_gb", y="arpu_idr", color="package", opacity=0.5, log_x=True,
                     title="Data usage vs ARPU (sampel 3.000)")
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- NPS
with tabs[3]:
    n = bm.nps(df["nps_score"])
    c = st.columns(4)
    c[0].metric("NPS", f"{n['nps']:.1f}", border=True)
    c[1].metric("Promoters (9-10)", f"{n['promoters_pct']:.1f}%", border=True)
    c[2].metric("Passives (7-8)", f"{n['passives_pct']:.1f}%", border=True)
    c[3].metric("Detractors (0-6)", f"{n['detractors_pct']:.1f}%", border=True)
    left, right = st.columns(2)
    cat = df.assign(nps_group=bm.nps_category(df["nps_score"]).values)
    cg = bm.churn_by(cat, "nps_group")
    fig = px.bar(cg, x="nps_group", y="churn_rate", color="nps_group", text=cg["churn_rate"].map(pct),
                 color_discrete_map={"Promoter": "#2E8B57", "Passive": "#E39B19", "Detractor": "#C8102E"},
                 title="Churn rate per kategori NPS", category_orders={"nps_group": ["Promoter", "Passive", "Detractor"]})
    fig.update_layout(yaxis_tickformat=".0%", showlegend=False)
    left.plotly_chart(fig, width="stretch")
    nr = bm.nps_by(df, "region")
    fig = go.Figure()
    for col, color in (("promoters_pct", "#2E8B57"), ("passives_pct", "#E39B19"), ("detractors_pct", "#C8102E")):
        fig.add_bar(y=nr["region"], x=nr[col], name=col.split("_")[0].title(), orientation="h", marker_color=color)
    fig.update_layout(barmode="stack", title="Komposisi NPS per region (%)")
    right.plotly_chart(fig, width="stretch")
    st.dataframe(nr.style.format({"nps": "{:.1f}", "promoters_pct": "{:.1f}", "passives_pct": "{:.1f}",
                                  "detractors_pct": "{:.1f}"}), width="stretch", hide_index=True)

# --------------------------------------------------------------------------- campaign
with tabs[4]:
    st.markdown("Kampanye retensi dijalankan sebagai **randomized controlled trial**: grup kontrol (`No Campaign`) "
                "tidak menerima penawaran, sehingga selisih churn = efek kausal kampanye (*uplift*).")
    cp = bm.campaign_performance(df, biz["campaign_cost_idr"], biz["gross_margin"], biz["retention_horizon_months"])
    c = st.columns(4)
    best = cp.iloc[0]
    c[0].metric("Pelanggan ditarget", num(cp["treated"].sum()), border=True)
    c[1].metric("Pelanggan terselamatkan", num(cp["customers_saved"].sum()), border=True)
    c[2].metric("Total biaya", idr(cp["cost_idr"].sum()), border=True)
    c[3].metric("ROI terbaik", f"{best['roi']:.1f}x", best["campaign"], delta_color="off", border=True)
    left, right = st.columns(2)
    fig = go.Figure()
    fig.add_bar(x=cp["campaign"], y=cp["uplift_abs"] * 100, name="Uplift (pp)",
                error_y=dict(type="data", symmetric=False,
                             array=(cp["uplift_ci_high"] - cp["uplift_abs"]) * 100,
                             arrayminus=(cp["uplift_abs"] - cp["uplift_ci_low"]) * 100))
    fig.add_hline(y=0, line_color="gray")
    fig.update_layout(title="Penurunan churn vs kontrol (pp) dengan 95% CI", yaxis_title="percentage points")
    left.plotly_chart(fig, width="stretch")
    fig = px.bar(cp, x="campaign", y="roi", text=cp["roi"].map(lambda v: f"{v:.1f}x"), title="ROI kampanye")
    right.plotly_chart(fig, width="stretch")
    show = cp[["campaign", "treated", "take_up_rate", "churn_treated", "churn_control", "uplift_abs",
               "uplift_rel", "p_value", "significant", "customers_saved", "cost_idr",
               "incremental_gross_profit_idr", "roi", "cost_per_saved_idr"]]
    st.dataframe(show.style.format({
        "take_up_rate": "{:.1%}", "churn_treated": "{:.2%}", "churn_control": "{:.2%}", "uplift_abs": "{:.2%}",
        "uplift_rel": "{:.1%}", "p_value": "{:.4f}", "customers_saved": "{:.0f}", "cost_idr": "Rp {:,.0f}",
        "incremental_gross_profit_idr": "Rp {:,.0f}", "roi": "{:.2f}x", "cost_per_saved_idr": "Rp {:,.0f}"}),
        width="stretch", hide_index=True)
    st.caption(f"Incremental gross profit = saved × ARPU × margin {biz['gross_margin']:.0%} × "
               f"{biz['retention_horizon_months']} bulan. Take-up rate = konversi (penawaran diredeem).")

# --------------------------------------------------------------------------- monthly KPI
with tabs[5]:
    kpi = load_kpis()
    metric = st.selectbox("KPI", ["revenue_idr_bn", "active_subscribers", "churn_rate", "arpu_idr",
                                  "data_traffic_pb", "avg_data_per_user_gb", "nps", "new_subscribers"])
    fig = px.line(kpi, x="month", y=metric, markers=True, title=f"{metric} — bulanan")
    for m in kpi.loc[kpi["is_lebaran"] == 1, "month"]:
        fig.add_vline(x=m, line_dash="dot", line_color="#E39B19")
    st.plotly_chart(fig, width="stretch")
    st.caption("Garis putus-putus oranye = bulan Lebaran (lonjakan traffic & revenue, churn naik bulan setelahnya).")
    kpi_yoy = kpi.set_index("month")[["revenue_idr_bn", "active_subscribers", "arpu_idr"]].pct_change(12).dropna()
    st.dataframe((kpi_yoy.tail(12) * 100).round(2).rename(columns=lambda c: f"{c} YoY %"), width="stretch")
