import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make app/ importable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_common import load_kpis, page_setup, read_artifact_csv, registry
from telco.analytics.statistics import stl_decompose
from telco.constants import forecast_model_name
from telco.models.forecaster import LSTMForecaster

cfg = page_setup("KPI Forecasting", "📈", "LSTM (PyTorch) untuk revenue & subscriber · benchmark vs Holt-Winters & seasonal naive")
targets = cfg["forecast"]["targets"]
labels = {"revenue_idr_bn": "Revenue (Rp miliar)", "active_subscribers": "Active subscribers"}
target = st.radio("Target", targets, format_func=lambda t: labels.get(t, t), horizontal=True)
entry = registry().get_production(forecast_model_name(target))
if entry is None:
    st.warning("Model forecasting belum dilatih.")
    st.stop()

kpi = load_kpis()
horizon = st.sidebar.slider("Horizon (bulan)", 3, 18, int(cfg["forecast"]["horizon"]))


@st.cache_resource(show_spinner=False)
def _load(path: str, version: int) -> LSTMForecaster:
    return LSTMForecaster.load(path)


model = _load(str(registry().artifact_dir(entry) / "lstm.pt"), entry["version"])
fc = model.predict(horizon)

m = entry["metrics"]
c = st.columns(4)
c[0].metric("Backtest MAPE — LSTM", f"{m['backtest_lstm_mape']:.2f}%", border=True)
c[1].metric("Holt-Winters", f"{m.get('backtest_holt_winters_mape', float('nan')):.2f}%", border=True)
c[2].metric("Seasonal naive", f"{m['backtest_seasonal_naive_mape']:.2f}%", border=True)
c[3].metric(f"Forecast {fc['month'].iloc[-1]:%b %Y}", f"{fc['forecast'].iloc[-1]:,.0f}".replace(",", "."), border=True)

fig = go.Figure()
fig.add_scatter(x=kpi["month"], y=kpi[target], name="Aktual", line=dict(color="#1F4E79", width=2.5))
fig.add_scatter(x=pd.concat([fc["month"], fc["month"][::-1]]), y=pd.concat([fc["upper"], fc["lower"][::-1]]),
                fill="toself", fillcolor="rgba(200,16,46,0.15)", line=dict(width=0), name="95% interval")
fig.add_scatter(x=fc["month"], y=fc["forecast"], name="Forecast LSTM", line=dict(color="#C8102E", width=3, dash="dash"))
for mth in kpi.loc[kpi["is_lebaran"] == 1, "month"]:
    fig.add_vline(x=mth, line_dash="dot", line_color="#E39B19", opacity=0.5)
fig.update_layout(title=f"{labels.get(target, target)} — histori & forecast {horizon} bulan", height=460)
st.plotly_chart(fig, width="stretch")
st.caption("Interval melebar ∝ √h dari deviasi residual growth bulanan. Garis oranye = Lebaran (fitur kalender LSTM).")

left, right = st.columns(2)
bt = read_artifact_csv(entry, "backtest.csv", ("month",))
fig = go.Figure()
fig.add_scatter(x=bt["month"], y=bt["actual"], name="Aktual", line=dict(color="black", width=3))
for col, color in (("lstm", "#C8102E"), ("holt_winters", "#1F4E79"), ("seasonal_naive", "#2E8B57")):
    if col in bt:
        fig.add_scatter(x=bt["month"], y=bt[col], name=col, line=dict(color=color, dash="dot"))
fig.update_layout(title="Backtest holdout (model dilatih tanpa periode ini)")
left.plotly_chart(fig, width="stretch")
rows = []
for meth in ("lstm", "holt_winters", "seasonal_naive"):
    if f"backtest_{meth}_mape" in m:
        rows.append({"method": meth, "MAPE %": m[f"backtest_{meth}_mape"], "RMSE": m[f"backtest_{meth}_rmse"],
                     "MAE": m[f"backtest_{meth}_mae"]})
right.dataframe(pd.DataFrame(rows).style.format({"MAPE %": "{:.2f}", "RMSE": "{:,.2f}", "MAE": "{:,.2f}"})
            .highlight_min(subset=["MAPE %"], color="#fde2e4"), width="stretch", hide_index=True)
right.caption("Selalu bandingkan model deep learning dengan baseline sederhana. Dengan 60 titik data, "
          "seasonal naive/Holt-Winters sering kompetitif — pilih model berdasarkan backtest, bukan kompleksitas.")
right.dataframe(fc.assign(month=fc["month"].dt.strftime("%b %Y")).style.format({"forecast": "{:,.2f}", "lower": "{:,.2f}",
                                                                            "upper": "{:,.2f}"}),
            width="stretch", hide_index=True)

with st.expander("STL decomposition & training loss"):
    dec = stl_decompose(kpi.set_index("month")[target], 12)
    fig = go.Figure()
    fig.add_scatter(x=dec.index, y=dec["trend"], name="trend")
    fig.add_scatter(x=dec.index, y=dec["seasonal"], name="seasonal", yaxis="y2")
    fig.update_layout(yaxis2=dict(overlaying="y", side="right"), title="Trend & seasonal component")
    st.plotly_chart(fig, width="stretch")
    if model.loss_history:
        st.line_chart(pd.DataFrame({"MSE loss": model.loss_history}))
