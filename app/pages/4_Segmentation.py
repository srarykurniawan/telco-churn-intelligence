import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make app/ importable

import pandas as pd
import plotly.express as px
import streamlit as st

from app_common import idr, load_customers, page_setup, pct, production, read_artifact_csv
from telco.constants import SEGMENT_MODEL
from telco.models.segmentation import SEGMENT_ACTIONS

cfg = page_setup("Customer Segmentation", "👥", "K-Means clustering perilaku pelanggan · pemilihan k via silhouette · profil & aksi")
seg, entry = production(SEGMENT_MODEL)
df = load_customers()
df = df.assign(segment=seg.predict_names(df))

c = st.columns(4)
c[0].metric("Jumlah segmen (k)", seg.k, border=True)
c[1].metric("Silhouette score", f"{entry['metrics']['silhouette']:.3f}", border=True)
c[2].metric("Fitur", len(seg.features), border=True)
c[3].metric("Model version", f"v{entry['version']}", border=True)

left, right = st.columns([2, 3])
sil = pd.DataFrame({"k": list(seg.silhouette_scores), "silhouette": list(seg.silhouette_scores.values())})
fig = px.line(sil, x="k", y="silhouette", markers=True, title="Silhouette per k (lebih tinggi = cluster lebih terpisah)")
fig.add_vline(x=seg.k, line_dash="dot", line_color="#C8102E")
left.plotly_chart(fig, width="stretch")
pca = read_artifact_csv(entry, "pca_sample.csv")
fig = px.scatter(pca, x="pc1", y="pc2", color="segment", opacity=0.55, title="Proyeksi PCA 2D (sampel 3.000)")
right.plotly_chart(fig, width="stretch")

summary = (df.groupby("segment")
           .agg(customers=("customer_id", "size"), churn_rate=("churn", "mean"), arpu=("arpu_idr", "mean"),
                data_gb=("data_usage_gb", "mean"), tenure=("tenure_months", "mean"),
                complaints=("num_complaints", "mean"), nps=("nps_score", "mean"))
           .reset_index())
summary["share"] = summary["customers"] / summary["customers"].sum()
summary["revenue_share"] = (df.groupby("segment")["arpu_idr"].sum() / df["arpu_idr"].sum()).reindex(summary["segment"]).values

left, right = st.columns(2)
fig = px.scatter(summary, x="arpu", y="churn_rate", size="customers", color="segment", text="segment",
                 title="Value vs risk per segmen (ukuran = jumlah pelanggan)")
fig.update_traces(textposition="top center")
fig.update_layout(yaxis_tickformat=".0%", showlegend=False)
left.plotly_chart(fig, width="stretch")
z = summary.set_index("segment")[["arpu", "data_gb", "tenure", "complaints", "nps", "churn_rate"]]
z = (z - z.mean()) / z.std(ddof=0)
fig = px.imshow(z, text_auto=".1f", color_continuous_scale="RdBu_r", zmin=-2, zmax=2, aspect="auto",
                title="Profil segmen (z-score relatif antar segmen)")
right.plotly_chart(fig, width="stretch")

st.subheader("Profil & rekomendasi aksi")
for _, row in summary.sort_values("customers", ascending=False).iterrows():
    with st.container(border=True):
        a, b = st.columns([1, 2])
        a.markdown(f"**{row['segment']}**  \n{pct(row['share'])} pelanggan · {pct(row['revenue_share'])} revenue")
        a.markdown(f"Churn **{pct(row['churn_rate'])}** · ARPU **{idr(row['arpu'], False)}** · NPS {row['nps']:.1f}")
        b.markdown(f"🎯 {SEGMENT_ACTIONS.get(row['segment'], 'Pertahankan engagement dengan penawaran personal.')}")
        b.caption(f"Data {row['data_gb']:.1f} GB · tenure {row['tenure']:.0f} bln · keluhan {row['complaints']:.2f}")

with st.expander("Segmen × plan type"):
    st.dataframe(pd.crosstab(df["segment"], df["plan_type"], normalize="index").style.format("{:.1%}"), width="stretch")
