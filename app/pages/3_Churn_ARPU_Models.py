import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make app/ importable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_common import (RISK_COLORS, idr, load_customers, page_setup, pct, production, read_artifact_csv)
from telco.analytics.business_metrics import customer_lifetime_value
from telco.constants import ARPU_MODEL, CHURN_MODEL, SEGMENT_MODEL
from telco.data.validation import DataValidationError
from telco.evaluation import metrics as em
from telco.pipelines.predict import recommend_action, score_customers

cfg = page_setup("Churn & ARPU Models", "🎯",
                 "PyTorch MLP vs baseline · evaluasi lengkap · threshold berbasis profit · prediksi individu & batch")
biz = cfg["business"]
churn_pipe, ce = production(CHURN_MODEL)
arpu_pipe, ae = production(ARPU_MODEL)
segmenter, _ = production(SEGMENT_MODEL)

tabs = st.tabs(["Evaluasi Churn Model", "Threshold & Profit", "Feature Importance", "ARPU Regression",
                "Prediksi Pelanggan", "Batch Scoring"])

preds = read_artifact_csv(ce, "test_predictions.csv")
y, p, pb = preds["y_true"].to_numpy(), preds["p_model"].to_numpy(), preds["p_baseline"].to_numpy()

# --------------------------------------------------------------------------- evaluation
with tabs[0]:
    st.caption(f"Model `{CHURN_MODEL}` v{ce['version']} · data `{ce['data_version']}` · "
               f"holdout test n = {len(y):,} · threshold bisnis = {churn_pipe.threshold:.2f}")
    m_model = em.classification_metrics(y, p, churn_pipe.threshold)
    m_base = em.classification_metrics(y, pb, churn_pipe.threshold)
    comp = pd.DataFrame({"PyTorch MLP": m_model, "Logistic Regression (baseline)": m_base}).T
    st.dataframe(comp[["roc_auc", "pr_auc", "ks", "f1", "precision", "recall", "accuracy", "brier", "log_loss"]]
                 .style.format("{:.4f}").highlight_max(axis=0, subset=["roc_auc", "pr_auc", "ks", "f1"], color="#fde2e4"),
                 width="stretch")
    st.caption("Baseline penting: jika MLP tidak jauh lebih baik dari logistic regression, relasi fitur–target "
               "kemungkinan besar near-linear pada skala logit; model yang lebih sederhana lebih mudah dijelaskan.")
    c1, c2 = st.columns(2)
    fig = go.Figure()
    for name, score, color in (("PyTorch MLP", p, "#C8102E"), ("LogReg", pb, "#1F4E79")):
        r = em.roc_points(y, score)
        fig.add_scatter(x=r["fpr"], y=r["tpr"], name=f"{name} (AUC {em.classification_metrics(y, score)['roc_auc']:.3f})",
                        line=dict(color=color))
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dot", color="gray"), name="random")
    fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
    c1.plotly_chart(fig, width="stretch")
    fig = go.Figure()
    for name, score, color in (("PyTorch MLP", p, "#C8102E"), ("LogReg", pb, "#1F4E79")):
        r = em.pr_points(y, score)
        fig.add_scatter(x=r["recall"], y=r["precision"], name=name, line=dict(color=color))
    fig.add_hline(y=y.mean(), line_dash="dot", annotation_text="base rate")
    fig.update_layout(title="Precision-Recall curve", xaxis_title="Recall", yaxis_title="Precision")
    c2.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    lt = em.lift_table(y, p)
    fig = go.Figure()
    fig.add_bar(x=lt["decile"], y=lt["lift"], name="Lift per decile")
    fig.add_scatter(x=lt["decile"], y=lt["cum_capture"] * 100, name="Cumulative capture %", yaxis="y2",
                    line=dict(color="#1F4E79"))
    fig.update_layout(title="Lift & gains (decile 1 = risiko tertinggi)", xaxis_title="decile",
                      yaxis2=dict(overlaying="y", side="right", range=[0, 105], title="%"))
    c1.plotly_chart(fig, width="stretch")
    cal = em.calibration_table(y, p)
    fig = px.line(cal, x="mean_predicted", y="observed_rate", markers=True, title="Calibration (reliability) plot")
    fig.add_scatter(x=[0, cal["mean_predicted"].max()], y=[0, cal["mean_predicted"].max()],
                    line=dict(dash="dot", color="gray"), name="perfect")
    c2.plotly_chart(fig, width="stretch")
    top = lt.iloc[:2]
    st.info(f"Menargetkan **20% pelanggan berisiko teratas** menangkap **{top['churners'].sum() / lt['churners'].sum():.0%}** "
            f"dari seluruh churner (lift {top['churners'].sum() / top['customers'].sum() / y.mean():.1f}x vs acak).")
    cm = np.array([[m_model["tn"], m_model["fp"]], [m_model["fn"], m_model["tp"]]])
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.imshow(cm, text_auto=True, x=["Pred retain", "Pred churn"], y=["Actual retain", "Actual churn"],
                              color_continuous_scale="Reds", title="Confusion matrix @ threshold bisnis"), width="stretch")
    hist = read_artifact_csv(ce, "training_history.csv")
    if hist is not None:
        fig = go.Figure()
        fig.add_scatter(x=hist["epoch"], y=hist["train_loss"], name="train BCE loss")
        fig.add_scatter(x=hist["epoch"], y=hist["val_loss"], name="val BCE loss")
        fig.add_scatter(x=hist["epoch"], y=hist["val_score"], name="val ROC-AUC", yaxis="y2", line=dict(dash="dot"))
        fig.update_layout(title="Kurva training (early stopping)", yaxis2=dict(overlaying="y", side="right"))
        c2.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- threshold & profit
with tabs[1]:
    st.markdown("Threshold dipilih untuk **memaksimalkan expected profit** kampanye retensi, bukan F1/accuracy.")
    c = st.columns(3)
    cost = c[0].number_input("Biaya per pelanggan dihubungi (Rp)", 0, 500_000, int(biz["retention_offer_cost_idr"]), 5_000)
    success = c[1].slider("Success rate retensi", 0.05, 0.8, float(biz["retention_success_rate"]), 0.05)
    horizon = c[2].slider("Horizon nilai (bulan)", 3, 36, int(biz["retention_horizon_months"]))
    value = preds["arpu_idr"].to_numpy() * biz["gross_margin"] * horizon
    curve = em.profit_curve(y, p, value, cost, success)
    best = curve.loc[curve["profit"].idxmax()]
    thr = st.slider("Threshold", 0.02, 0.95, float(round(best["threshold"], 2)), 0.01)
    row = curve.iloc[(curve["threshold"] - thr).abs().idxmin()]
    c = st.columns(4)
    c[0].metric("Threshold optimal", f"{best['threshold']:.2f}", border=True)
    c[1].metric("Profit @ threshold", idr(row["profit"]), f"optimal {idr(best['profit'])}", delta_color="off", border=True)
    c[2].metric("Pelanggan dihubungi", f"{int(row['contacted']):,}", pct(row["contacted"] / len(y)), delta_color="off", border=True)
    c[3].metric("Recall churners", pct(row["tp"] / max(y.sum(), 1)), border=True)
    fig = go.Figure()
    fig.add_scatter(x=curve["threshold"], y=curve["profit"] / 1e6, name="Profit (Rp jt)", line=dict(color="#C8102E", width=3))
    fig.add_scatter(x=curve["threshold"], y=curve["gain"] / 1e6, name="Gain", line=dict(dash="dot"))
    fig.add_scatter(x=curve["threshold"], y=curve["cost"] / 1e6, name="Cost", line=dict(dash="dot"))
    fig.add_vline(x=thr, line_color="#1F4E79")
    fig.update_layout(title=f"Profit curve pada holdout ({len(y):,} pelanggan)", xaxis_title="threshold", yaxis_title="Rp juta")
    st.plotly_chart(fig, width="stretch")
    mm = em.classification_metrics(y, p, thr)
    st.caption(f"@ {thr:.2f}: precision {mm['precision']:.3f} · recall {mm['recall']:.3f} · F1 {mm['f1']:.3f}")

# --------------------------------------------------------------------------- importance
with tabs[2]:
    imp = read_artifact_csv(ce, "feature_importance.csv")
    fig = px.bar(imp.sort_values("importance"), x="importance", y="feature", orientation="h", error_x="std",
                 title="Permutation importance (penurunan ROC-AUC saat fitur diacak)")
    fig.update_layout(height=560)
    st.plotly_chart(fig, width="stretch")
    st.caption("Model-agnostic: dihitung pada kolom mentah, sehingga fitur kategorikal di-one-hot tetap dinilai sebagai satu fitur.")

# --------------------------------------------------------------------------- ARPU
with tabs[3]:
    ap = read_artifact_csv(ae, "test_predictions.csv")
    m = ae["metrics"]
    c = st.columns(4)
    c[0].metric("R² (MLP)", f"{m['test_r2']:.4f}", f"LinReg {m['baseline_linreg_r2']:.4f}", delta_color="off", border=True)
    c[1].metric("RMSE", idr(m["test_rmse"], False), f"LinReg {idr(m['baseline_linreg_rmse'], False)}", delta_color="off", border=True)
    c[2].metric("MAE", idr(m["test_mae"], False), border=True)
    c[3].metric("MAPE", f"{m['test_mape']:.2f}%", f"LinReg {m['baseline_linreg_mape']:.2f}%", delta_color="off", border=True)
    s = ap.sample(min(2500, len(ap)), random_state=0)
    left, right = st.columns(2)
    fig = px.scatter(s, x="y_true", y="y_pred", color="package", opacity=0.6, title="Actual vs predicted ARPU (MLP)")
    mx = s[["y_true", "y_pred"]].max().max()
    fig.add_scatter(x=[0, mx], y=[0, mx], line=dict(dash="dot", color="gray"), name="ideal")
    left.plotly_chart(fig, width="stretch")
    res = ap.assign(residual_mlp=ap["y_pred"] - ap["y_true"], residual_linreg=ap["y_pred_baseline"] - ap["y_true"])
    fig = go.Figure()
    fig.add_histogram(x=res["residual_mlp"], name="MLP", opacity=0.7, nbinsx=80)
    fig.add_histogram(x=res["residual_linreg"], name="Linear", opacity=0.5, nbinsx=80)
    fig.update_layout(barmode="overlay", title="Distribusi residual (Rp)")
    right.plotly_chart(fig, width="stretch")
    st.caption("MLP menangkap non-linearitas (overage kuota & add-on) yang tidak bisa ditangkap regresi linear pada log(ARPU).")

# --------------------------------------------------------------------------- single prediction
with tabs[4]:
    df = load_customers()
    st.markdown("Isi profil pelanggan, atau muat pelanggan acak dari data.")
    if st.button("🎲 Muat pelanggan acak"):
        st.session_state["cust"] = df.sample(1).iloc[0].to_dict()
    base = st.session_state.get("cust", df.iloc[0].to_dict())
    with st.form("cust_form"):
        c = st.columns(4)
        vals = {}
        vals["region"] = c[0].selectbox("Region", sorted(df.region.unique()), index=sorted(df.region.unique()).index(base["region"]))
        vals["plan_type"] = c[1].selectbox("Plan", ["Prepaid", "Postpaid"], index=["Prepaid", "Postpaid"].index(base["plan_type"]))
        vals["package"] = c[2].selectbox("Package", ["Basic", "Standard", "Premium", "Unlimited"],
                                         index=["Basic", "Standard", "Premium", "Unlimited"].index(base["package"]))
        contracts = ["No Contract", "Month-to-Month", "12 Months", "24 Months"]
        vals["contract_type"] = c[3].selectbox("Contract", contracts, index=contracts.index(base["contract_type"]))
        pays = sorted(df.payment_method.unique())
        vals["payment_method"] = c[0].selectbox("Payment", pays, index=pays.index(base["payment_method"]))
        vals["device_type"] = c[1].selectbox("Device", ["4G", "5G"], index=["4G", "5G"].index(base["device_type"]))
        vals["has_streaming_addon"] = c[2].selectbox("Streaming add-on", ["No", "Yes"], index=["No", "Yes"].index(base["has_streaming_addon"]))
        camps = ["No Campaign", "Discount", "Bonus Data", "Loyalty Points"]
        vals["campaign_type"] = c[3].selectbox("Campaign", camps, index=camps.index(base["campaign_type"]))
        vals["tenure_months"] = c[0].number_input("Tenure (bulan)", 1, 240, int(base["tenure_months"]))
        vals["age"] = c[1].number_input("Usia", 17, 90, int(base["age"]))
        vals["monthly_charge_idr"] = c[2].number_input("Harga paket (Rp)", 0, 2_000_000, int(base["monthly_charge_idr"]), 5000)
        vals["arpu_idr"] = c[3].number_input("ARPU aktual (Rp)", 0, 5_000_000, int(base["arpu_idr"]), 5000)
        vals["data_usage_gb"] = c[0].number_input("Data usage (GB)", 0.0, 1000.0, float(base["data_usage_gb"]))
        vals["voice_minutes"] = c[1].number_input("Voice (menit)", 0, 10000, int(base["voice_minutes"]))
        vals["sms_count"] = c[2].number_input("SMS", 0, 5000, int(base["sms_count"]))
        vals["network_quality_score"] = c[3].slider("Network quality", 1.0, 10.0, float(base["network_quality_score"]), 0.1)
        vals["avg_call_drop_rate"] = c[0].number_input("Call drop rate (%)", 0.0, 50.0, float(base["avg_call_drop_rate"]))
        vals["num_complaints"] = c[1].number_input("Jumlah keluhan", 0, 50, int(base["num_complaints"]))
        vals["days_since_last_topup"] = c[2].number_input("Hari sejak top-up/bayar", 0, 365, int(base["days_since_last_topup"]))
        vals["usage_change_pct"] = c[3].slider("Perubahan usage 3 bln (%)", -80.0, 80.0, float(base["usage_change_pct"]))
        vals["nps_score"] = c[0].slider("NPS score (0-10)", 0, 10, int(base["nps_score"]))
        submitted = st.form_submit_button("Prediksi", type="primary")
    row = pd.DataFrame([{"customer_id": "MANUAL", **vals}])
    prob = float(churn_pipe.predict(row)[0])
    pred_arpu = float(arpu_pipe.predict(row)[0])
    clv = float(customer_lifetime_value([vals["arpu_idr"]], [prob], biz["gross_margin"], biz["max_lifetime_months"])[0])
    seg = segmenter.predict_names(row)[0] if segmenter is not None else "N/A"
    thr = churn_pipe.threshold
    tier = "High" if prob >= thr else "Medium" if prob >= 0.6 * thr else "Low"
    c = st.columns(5)
    c[0].metric("Probabilitas churn 90 hari", pct(prob), border=True)
    c[1].markdown(f"<div style='padding-top:1.6rem'><span style='background:{RISK_COLORS[tier]};color:white;"
                  f"padding:6px 16px;border-radius:999px;font-weight:700'>Risk: {tier}</span></div>", unsafe_allow_html=True)
    c[2].metric("Prediksi ARPU", idr(pred_arpu, False), f"aktual {idr(vals['arpu_idr'], False)}", delta_color="off", border=True)
    c[3].metric("CLV", idr(clv), border=True)
    c[4].metric("Segmen", seg, border=True)
    st.success(f"**Rekomendasi:** {recommend_action(prob, thr, clv, churn_pipe.meta.get('clv_p75', clv), seg)}")
    # local explanation: occlusion — replace each feature with the population median/mode
    ref = df.sample(2000, random_state=0)
    contrib = []
    for f in cfg["features"]["numeric"] + cfg["features"]["categorical"]:
        alt = row.copy()
        alt[f] = ref[f].median() if f in cfg["features"]["numeric"] else ref[f].mode()[0]
        contrib.append({"feature": f, "impact": prob - float(churn_pipe.predict(alt)[0])})
    contrib = pd.DataFrame(contrib).assign(abs=lambda d: d["impact"].abs()).nlargest(10, "abs").sort_values("impact")
    fig = px.bar(contrib, x="impact", y="feature", orientation="h",
                 color=np.where(contrib["impact"] > 0, "menaikkan risiko", "menurunkan risiko"),
                 color_discrete_map={"menaikkan risiko": "#C8102E", "menurunkan risiko": "#2E8B57"},
                 title="Driver risiko pelanggan ini (vs profil tipikal)")
    fig.update_layout(xaxis_tickformat=".1%", legend_title=None)
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- batch scoring
with tabs[5]:
    st.markdown("Upload CSV dengan skema yang sama (kolom `churn` opsional). Data divalidasi sebelum di-scoring.")
    template = load_customers().drop(columns=["churn", "campaign_accepted"]).head(20)
    st.download_button("⬇️ Download template CSV", template.to_csv(index=False), "scoring_template.csv", "text/csv")
    up = st.file_uploader("Upload CSV", type=["csv"])
    use_sample = st.checkbox("Atau gunakan 1.000 pelanggan sampel")
    data = None
    if up is not None:
        try:
            data = pd.read_csv(up)
        except Exception as exc:  # noqa: BLE001
            st.error(f"File tidak bisa dibaca: {exc}")
    elif use_sample:
        data = load_customers().sample(1000, random_state=7).drop(columns=["churn"])
    if data is not None:
        try:
            models = {CHURN_MODEL: (churn_pipe, ce), ARPU_MODEL: (arpu_pipe, ae)}
            if segmenter is not None:
                models[SEGMENT_MODEL] = (segmenter, None)
            scored = score_customers(data, cfg, models)
        except (DataValidationError, KeyError) as exc:
            st.error(f"Validasi gagal: {exc}")
        else:
            c = st.columns(4)
            c[0].metric("Di-scoring", f"{len(scored):,}", border=True)
            c[1].metric("High risk", f"{(scored.risk_tier == 'High').sum():,}", border=True)
            c[2].metric("Revenue at risk (high)", idr(scored.loc[scored.risk_tier == "High", "arpu_idr"].sum() * 12) + "/thn", border=True)
            c[3].metric("Rata-rata probabilitas", pct(scored.churn_probability.mean()), border=True)
            left, right = st.columns(2)
            left.plotly_chart(px.histogram(scored, x="churn_probability", color="risk_tier", nbins=40,
                                        color_discrete_map=RISK_COLORS, title="Distribusi skor"), width="stretch")
            right.plotly_chart(px.bar(scored.groupby("recommended_action").size().reset_index(name="n"), x="n",
                                  y="recommended_action", orientation="h", title="Next best action"), width="stretch")
            cols = ["customer_id", "churn_probability", "risk_tier", "predicted_arpu_idr", "clv_idr", "segment",
                    "recommended_action"]
            st.dataframe(scored.sort_values("churn_probability", ascending=False)[cols].head(200), width="stretch", hide_index=True)
            st.download_button("⬇️ Download hasil scoring", scored.to_csv(index=False), "scored_customers.csv", "text/csv",
                               type="primary")
