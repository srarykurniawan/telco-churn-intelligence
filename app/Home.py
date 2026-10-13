"""Telco Churn & Revenue Intelligence — Executive overview (Streamlit entry point)."""

import json

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_common import (get_config, idr, load_customers, load_kpis, num, page_setup, pct, production,
                        registry)
from telco.analytics import business_metrics as bm
from telco.constants import ARPU_MODEL, CHURN_MODEL

cfg = page_setup("Telco Churn & Revenue Intelligence", "📡",
                 "End-to-end data science untuk bisnis subscription telekomunikasi: statistik, "
                 "PyTorch, MLOps, dan metrik bisnis (churn, ARPU, usage, NPS, kampanye).")

df = load_customers()
kpi = load_kpis()
k = bm.executive_kpis(df)
last, prev = kpi.iloc[-1], kpi.iloc[-2]

st.markdown("#### Snapshot pelanggan (sampel analitik)")
c = st.columns(5)
c[0].metric("Pelanggan", num(k["customers"]), border=True)
c[1].metric("Churn 90 hari", pct(k["churn_rate"]), border=True)
c[2].metric("ARPU / bulan", idr(k["arpu_idr"], short=False), border=True)
c[3].metric("NPS", num(k["nps_nps"], 1), border=True)
c[4].metric("Revenue at risk / thn", idr(k["revenue_at_risk_annual_idr"]),
            pct(k["revenue_at_risk_pct"]) + " dari revenue", delta_color="inverse", border=True)

st.markdown(f"#### KPI perusahaan — {last['month']:%B %Y}")
c = st.columns(5)
c[0].metric("Active subscribers", f"{num(last['active_subscribers'] / 1e6, 2)} jt",
            f"{(last['active_subscribers'] / prev['active_subscribers'] - 1) * 100:+.2f}% MoM", border=True)
c[1].metric("Monthly churn", pct(last["churn_rate"], 2),
            f"{(last['churn_rate'] - prev['churn_rate']) * 100:+.2f} pp", delta_color="inverse", border=True)
c[2].metric("ARPU", idr(last["arpu_idr"], short=False),
            f"{(last['arpu_idr'] / prev['arpu_idr'] - 1) * 100:+.1f}% MoM", border=True)
c[3].metric("Revenue", f"Rp {last['revenue_idr_bn']:,.0f} M".replace(",", "."),
            f"{(last['revenue_idr_bn'] / prev['revenue_idr_bn'] - 1) * 100:+.1f}% MoM", border=True)
c[4].metric("Data traffic", f"{last['data_traffic_pb']:,.0f} PB".replace(",", "."),
            f"{(last['data_traffic_pb'] / prev['data_traffic_pb'] - 1) * 100:+.1f}% MoM", border=True)

left, right = st.columns([3, 2])
with left:
    fig = go.Figure()
    fig.add_bar(x=kpi["month"], y=kpi["revenue_idr_bn"], name="Revenue (Rp M)", marker_color="#C8102E", opacity=0.85)
    fig.add_scatter(x=kpi["month"], y=kpi["active_subscribers"] / 1e6, name="Subscribers (jt)", yaxis="y2",
                    line=dict(color="#1F4E79", width=3))
    leb = kpi[kpi["is_lebaran"] == 1]
    fig.add_scatter(x=leb["month"], y=leb["revenue_idr_bn"], mode="markers", name="Lebaran",
                    marker=dict(symbol="star", size=12, color="#E39B19"))
    fig.update_layout(title="Revenue & subscriber 60 bulan terakhir", height=380,
                      yaxis=dict(title="Rp miliar"), yaxis2=dict(title="juta", overlaying="y", side="right"))
    st.plotly_chart(fig, width="stretch")
with right:
    cc =bm.churn_by(df, "contract_type")
    fig = px.bar(cc, x="churn_rate", y="contract_type", orientation="h", text=cc["churn_rate"].map(pct),
                 title="Churn rate per tipe kontrak", labels={"churn_rate": "Churn rate", "contract_type": ""})
    fig.update_layout(height=380, xaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

st.markdown("#### Status model (Production)")
cols = st.columns(3)
churn_pipe, ce = production(CHURN_MODEL)
_, ae = production(ARPU_MODEL)
cols[0].metric("Churn model (PyTorch MLP)", f"v{ce['version']}", f"ROC-AUC {ce['metrics']['test_roc_auc']:.3f}",
               delta_color="off", border=True)
cols[1].metric("ARPU regressor (PyTorch MLP)", f"v{ae['version']}", f"R² {ae['metrics']['test_r2']:.3f}",
               delta_color="off", border=True)
report_path = get_config().path("monitoring_dir") / "latest_report.json"
if report_path.exists():
    rep = json.loads(report_path.read_text())
    sev = rep["decision"]["severity"]
    cols[2].metric("Monitoring terakhir", sev.upper(), f"{rep['n_drifted_features']} fitur drift · {rep['batch'][:22]}",
                   delta_color="off", border=True)
else:
    cols[2].metric("Monitoring terakhir", "Belum ada", "jalankan di halaman MLOps", delta_color="off", border=True)

with st.expander("🧭 Peta aplikasi & cara membaca", expanded=False):
    st.markdown("""
| Halaman | Isi | Kompetensi |
|---|---|---|
| **Business Metrics** | Churn, ARPU, revenue at risk, usage, NPS, performa kampanye (uplift, ROI) | Subscription business metrics |
| **Statistical Analysis** | Hypothesis testing, A/B test & sample size, OLS/Logit regression, korelasi, ADF | Statistik inferensial |
| **Churn & ARPU Models** | PyTorch MLP vs baseline, ROC/PR/lift/kalibrasi, threshold berbasis profit, scoring | Klasifikasi, regresi, evaluasi |
| **Segmentation** | K-Means + silhouette, PCA, profil & aksi per segmen | Clustering |
| **Forecasting** | LSTM PyTorch vs Holt-Winters vs seasonal naive, STL decomposition | Time series |
| **MLOps** | Model registry, data versioning, drift monitoring (PSI), retraining trigger, champion/challenger | MLOps |
""")
    st.caption(f"Total registered model versions: {len(registry().list_versions())}. "
               "Data sintetis realistis (pasar Indonesia) — ganti dengan data internal lewat skema yang sama.")
