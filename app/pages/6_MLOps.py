import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make app/ importable

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from app_common import STATUS_COLORS, clear_caches, get_config, page_setup, registry
from telco.constants import CHURN_MODEL
from telco.data.generator import DRIFT_SCENARIOS
from telco.data.versioning import DataVersionManager
from telco.pipelines.monitor import generate_batch, list_batches, load_history, run_monitoring

cfg = page_setup("MLOps", "⚙️", "Model registry · data versioning · drift monitoring · retraining trigger · champion/challenger")
reg = registry()
mdir = cfg.path("monitoring_dir")

tabs = st.tabs(["Model Registry", "Data Versions", "Monitoring & Drift", "Retraining", "Arsitektur"])

KEY_METRIC = {"churn_model": "test_roc_auc", "arpu_regressor": "test_r2", "customer_segmentation": "silhouette"}

# --------------------------------------------------------------------------- registry
with tabs[0]:
    models = reg.list_models()
    name = st.selectbox("Model", models, index=models.index(CHURN_MODEL) if CHURN_MODEL in models else 0)
    versions = reg.list_versions(name)
    km = KEY_METRIC.get(name, "backtest_lstm_mape")
    table = pd.DataFrame([{"version": v["version"], "stage": v["stage"], "created_at": v["created_at"],
                           km: v["metrics"].get(km), "data_version": v["data_version"],
                           "git": v.get("git_commit"), "description": v["description"]} for v in versions])
    st.dataframe(table.sort_values("version", ascending=False).style.apply(
        lambda r: ["background-color: #e8f5e9" if r["stage"] == "Production" else "" for _ in r], axis=1),
        width="stretch", hide_index=True)
    if len(table) > 1:
        st.plotly_chart(px.line(table, x="version", y=km, markers=True, title=f"{km} per versi"), width="stretch")
    c1, c2, c3 = st.columns([1, 1, 2])
    ver = c1.selectbox("Versi", table["version"].tolist()[::-1])
    stage = c2.selectbox("Pindahkan ke stage", ["Production", "Staging", "Archived"])
    c3.write("")
    if c3.button("Terapkan transisi (promote / rollback)", type="primary"):
        reg.transition(name, int(ver), stage, reason="manual via Streamlit UI")
        clear_caches()
        st.session_state["flash"] = f"{name} v{ver} → {stage}"
        st.rerun()
    if msg := st.session_state.pop("flash", None):
        st.success(f"✔ Transisi berhasil: {msg}")
    sel = reg.get(name, int(ver))
    with st.expander(f"Model card & lineage — v{ver}"):
        card = reg.artifact_dir(sel) / "model_card.md"
        if card.exists():
            st.markdown(card.read_text())
        st.json({"params": sel["params"], "history": sel.get("history", [])}, expanded=False)

# --------------------------------------------------------------------------- data versions
with tabs[1]:
    dv = DataVersionManager(cfg.path("data_manifest")).list()
    if dv:
        dvt = pd.DataFrame(dv)[["dataset", "version_id", "rows", "target_rate", "sha256", "created_at", "description"]]
        dvt["sha256"] = dvt["sha256"].str[:12] + "…"
        st.dataframe(dvt, width="stretch", hide_index=True)
    st.caption("Setiap dataset di-hash (SHA-256) — konten sama → versi sama (idempoten). Registry menyimpan `data_version` "
               "sehingga setiap model dapat ditelusuri ke data latihnya. `dvc.yaml` disediakan untuk integrasi DVC + remote storage.")

# --------------------------------------------------------------------------- monitoring
with tabs[2]:
    st.markdown("Simulasikan batch pelanggan baru (label churn tiba setelah 90 hari) lalu bandingkan dengan "
                "distribusi data latih model Production.")
    c1, c2, c3 = st.columns([2, 1, 1])
    scenario = c1.select_slider("Skenario drift", list(DRIFT_SCENARIOS), value="mild",
                                help="none = populasi sama · mild = kualitas jaringan turun, harga naik 5% · "
                                     "severe = gangguan jaringan besar, harga naik 20%, shift ke prepaid")
    n = c2.number_input("Ukuran batch", 1000, 20000, int(cfg["monitoring"]["default_batch_size"]), 1000)
    c3.write("")
    if c3.button("▶️ Generate & monitor", type="primary"):
        with st.spinner("Menjalankan monitoring…"):
            path = generate_batch(cfg, scenario, int(n))
            run_monitoring(cfg, path, log=lambda m: None)
        st.rerun()
    rep_path = mdir / "latest_report.json"
    if rep_path.exists():
        rep = json.loads(rep_path.read_text())
        drift = pd.read_csv(mdir / "latest_drift.csv")
        dec = rep["decision"]
        sev_icon = {"ok": "🟢", "warning": "🟡", "high": "🟠", "critical": "🔴"}.get(dec["severity"], "⚪")
        st.markdown(f"#### {sev_icon} Batch `{rep['batch']}` — severity **{dec['severity'].upper()}** · "
                    f"retrain: **{'YA' if dec['retrain'] else 'TIDAK'}**")
        c = st.columns(5)
        c[0].metric("Fitur drift (PSI ≥ 0.2)", rep["n_drifted_features"], border=True)
        c[1].metric("Fitur warning", rep["n_warning_features"], border=True)
        c[2].metric("PSI skor prediksi", f"{rep['score_psi']:.3f}", border=True)
        if rep.get("current_auc") is not None:
            c[3].metric("AUC batch baru", f"{rep['current_auc']:.3f}", f"{rep['current_auc'] - rep['reference_auc']:+.3f} vs ref",
                        border=True)
        if rep.get("current_churn_rate") is not None:
            c[4].metric("Churn batch baru", f"{rep['current_churn_rate']:.1%}",
                        f"{(rep['current_churn_rate'] - (rep.get('reference_churn_rate') or 0)) * 100:+.1f} pp",
                        delta_color="inverse", border=True)
        fig = px.bar(drift.sort_values("psi"), x="psi", y="feature", color="status", orientation="h",
                     color_discrete_map=STATUS_COLORS, title="Population Stability Index per fitur")
        fig.add_vline(x=cfg["monitoring"]["psi_warning"], line_dash="dot", line_color="#E39B19")
        fig.add_vline(x=cfg["monitoring"]["psi_threshold"], line_dash="dot", line_color="#C8102E")
        fig.update_layout(height=560)
        st.plotly_chart(fig, width="stretch")
        checks = pd.DataFrame([{"check": k, **v} for k, v in dec["checks"].items()])
        st.dataframe(checks, width="stretch", hide_index=True)
        with st.expander("Tabel drift lengkap (PSI, KS / Chi² p-value)"):
            st.dataframe(drift.style.format({"psi": "{:.4f}", "statistic": "{:.4f}", "p_value": "{:.2e}",
                                             "ref_mean": "{:.2f}", "cur_mean": "{:.2f}"}), width="stretch", hide_index=True)
            st.caption("Status berbasis PSI. Uji KS/Chi² pada sampel besar hampir selalu signifikan, sehingga hanya informatif.")
    else:
        st.info("Belum ada laporan monitoring. Klik **Generate & monitor**.")
    hist = load_history(cfg)
    if not hist.empty:
        hist["timestamp"] = pd.to_datetime(hist["timestamp"])
        fig = px.line(hist, x="timestamp", y=["score_psi", "current_auc"], markers=True, title="Riwayat monitoring")
        st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- retraining
with tabs[3]:
    pol = cfg["monitoring"]
    st.markdown(f"""
**Kebijakan retraining trigger** (`config/config.yaml → monitoring`): retrain jika salah satu terpenuhi **dan** tersedia
≥ {pol['min_new_samples']:,} sampel berlabel baru:
1. **Data drift** — ≥ {pol['min_drifted_features']} fitur dengan PSI ≥ {pol['psi_threshold']}
2. **Prediction drift** — PSI distribusi skor ≥ {pol['psi_threshold']}
3. **Performance decay** — ROC-AUC turun ≥ {pol['auc_drop_threshold']}
4. **Staleness** — umur model ≥ {pol['max_model_age_days']} hari

Model baru (*challenger*) dibandingkan head-to-head dengan *champion* pada data baru yang belum pernah dilihat champion;
dipromosikan ke Production hanya jika AUC ≥ champion − {pol['promotion_tolerance']}.
""")
    batches = list_batches(cfg)
    if not batches:
        st.info("Belum ada batch baru. Jalankan monitoring terlebih dahulu.")
    else:
        rep = json.loads((mdir / "latest_report.json").read_text()) if (mdir / "latest_report.json").exists() else None
        batch = st.selectbox("Batch untuk retraining", batches, format_func=lambda p: p.name)
        triggered = bool(rep and rep["batch"] == batch.name and rep["decision"]["retrain"])
        if triggered:
            st.warning("🔔 Retraining trigger AKTIF untuk batch ini.")
        else:
            st.info("Trigger tidak aktif untuk batch ini — retraining tetap bisa dipaksa (manual).")
        if st.button("🔁 Jalankan retraining (churn, ARPU, segmentasi)", type="primary"):
            from telco.pipelines.train import run_training

            with st.status("Retraining…", expanded=True) as status:
                summary = run_training(get_config(), extra_data=[batch], skip_forecast=True,
                                       log=lambda m: status.write(m))
                status.update(label="Retraining selesai", state="complete")
            clear_caches()
            promo = summary["churn"]["promotion"]
            (st.success if promo["promoted"] else st.warning)(
                f"Churn model v{summary['churn']['version']}: {'DIPROMOSIKAN ke Production' if promo['promoted'] else 'tetap di Staging'} — "
                f"{promo.get('reason')}")
            st.json(promo)
        st.caption("Catatan: di Streamlit Community Cloud filesystem bersifat sementara — hasil retraining dari UI hilang saat app "
                   "restart. Untuk produksi, retraining terjadwal berjalan di GitHub Actions (`.github/workflows/retrain.yml`) "
                   "dan meng-commit registry terbaru → app ter-redeploy otomatis.")

# --------------------------------------------------------------------------- architecture
with tabs[4]:
    st.graphviz_chart("""
digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f7f7f9", fontname="Helvetica", fontsize=11];
  data [label="Raw data\\n(customers, KPI)"]; val [label="Validation\\n(schema, ranges)"];
  dv [label="Data versioning\\n(SHA-256 manifest)"]; fe [label="Feature eng.\\n+ preprocessor"];
  train [label="Training\\nPyTorch MLP / LSTM\\nK-Means", fillcolor="#fde2e4"];
  eval [label="Evaluation\\nAUC, lift, profit,\\nbaselines"]; reg [label="Model registry\\nStaging → Production", fillcolor="#e8f5e9"];
  app [label="Streamlit app\\n& batch scoring"]; mon [label="Monitoring\\nPSI / KS / AUC"];
  trig [label="Retrain trigger\\npolicy", fillcolor="#fff4d6"]; cc [label="Champion /\\nchallenger"];
  data -> val -> dv -> fe -> train -> eval -> reg -> app;
  app -> mon [label="new labelled batch"]; mon -> trig; trig -> train [label="retrain"]; train -> cc -> reg;
}
""")
