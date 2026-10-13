# 📡 Telco Churn & Revenue Intelligence

Project **end-to-end data science & machine learning** untuk bisnis subscription telekomunikasi — dari data, statistik,
model **PyTorch**, sampai **MLOps** (versioning, model registry, monitoring, retraining trigger) dan aplikasi
**Streamlit** yang siap deploy.

| Requirement | Implementasi di project ini |
|---|---|
| **Statistics & ML foundation** | Hypothesis testing (Welch t-test, Mann-Whitney, χ², ANOVA, Kruskal-Wallis, korelasi, bootstrap CI) dengan effect size · A/B test (two-proportion z-test, CI, sample size/power) · **Regression** (OLS HC3, MLP regressor ARPU) · **Classification** (MLP churn vs logistic baseline) · **Clustering** (K-Means + silhouette, PCA) · **Time series** (ADF, STL, LSTM vs Holt-Winters vs seasonal naive) · **Evaluation metrics** (ROC-AUC, PR-AUC, KS, F1, Brier, log-loss, lift/gains, kalibrasi, RMSE/MAE/MAPE/R², profit curve) |
| **Framework TensorFlow / PyTorch** | **PyTorch**: MLP tabular (BatchNorm, Dropout, AdamW, LR scheduler, early stopping) untuk churn & ARPU, **LSTM** untuk forecasting revenue & subscriber |
| **MLOps** | **Data versioning** (SHA-256 manifest + `dvc.yaml`) · **Model registry** (versi, stage Staging/Production/Archived, lineage ke data & git commit, model card, rollback) · **Monitoring** (PSI, KS, χ², prediction drift, AUC decay) · **Retraining trigger** (kebijakan berbasis drift/performa/umur model) · **Champion/challenger** · CI + retraining terjadwal di GitHub Actions |
| **Subscription business metrics** | **Churn** (90 hari → hazard bulanan), **ARPU**, revenue at risk, **CLV**, **Usage** (data/voice/SMS, heavy users, tren usage), **NPS** (promoter/passive/detractor), **Campaign metrics** (take-up/konversi, uplift kausal RCT + CI, customers saved, ROI, cost per saved) |

---

## 🗂️ Struktur repository

```
telco-churn-intelligence/
├── app/                              # Streamlit (multipage)
│   ├── Home.py                       # entry point: executive overview
│   ├── app_common.py                 # bootstrap, cache, formatting, tema
│   └── pages/
│       ├── 1_Business_Metrics.py     # churn, ARPU, usage, NPS, campaign, KPI bulanan
│       ├── 2_Statistical_Analysis.py # hypothesis test, A/B, regresi, korelasi, ADF/STL
│       ├── 3_Churn_ARPU_Models.py    # evaluasi, threshold profit, importance, prediksi & batch scoring
│       ├── 4_Segmentation.py         # K-Means, silhouette, PCA, aksi per segmen
│       ├── 5_Forecasting.py          # LSTM vs baseline, interval prediksi
│       └── 6_MLOps.py                # registry, data versions, drift, retraining, arsitektur
├── src/telco/                        # package Python (logika bisnis & ML, terpisah dari UI)
│   ├── config.py · constants.py
│   ├── data/          generator.py (data sintetis realistis) · validation.py · versioning.py
│   ├── features/      engineering.py (fitur turunan + ColumnTransformer)
│   ├── analytics/     statistics.py · business_metrics.py
│   ├── models/        torch_tabular.py (MLP) · pipeline.py · forecaster.py (LSTM) · segmentation.py
│   ├── evaluation/    metrics.py
│   ├── mlops/         registry.py · monitoring.py · retraining.py
│   └── pipelines/     train.py · monitor.py · predict.py   (CLI)
├── config/config.yaml                # SEMUA parameter (fitur, hyperparameter, bisnis, threshold MLOps)
├── data/raw/                         # customers.csv (20k) · monthly_kpi.csv (60 bulan)
├── data/versions/manifest.json       # data versioning
├── data/incoming/                    # batch baru untuk monitoring (di-generate)
├── models/registry/                  # model registry (registry.json + <model>/v<N>/)
├── monitoring/                       # latest_report.json · latest_drift.csv · history.jsonl
├── reports/training_summary.json
├── tests/                            # 32 unit & integration test (pytest, coverage ≈ 89%)
├── .github/workflows/                # ci.yml · retrain.yml (terjadwal)
├── .streamlit/config.toml
├── Dockerfile · Makefile · dvc.yaml · pyproject.toml
└── requirements.txt · requirements-dev.txt
```

## 🧠 Data

Data **sintetis namun realistis** pasar Indonesia (`src/telco/data/generator.py`) — region (Jawa, Sumatera, …),
prepaid/postpaid, paket & harga Rupiah, kuota & overage, kualitas jaringan per region, keluhan, NPS, efek **Lebaran**.
Kampanye retensi dialokasikan **secara acak (RCT)** sehingga uplift dapat diukur secara kausal. Label `churn` =
churn dalam 90 hari. Skenario **drift** (`none` / `mild` / `severe`) mensimulasikan perubahan populasi untuk MLOps.

**Mengganti dengan data internal**: siapkan CSV dengan kolom yang sama (lihat `CUSTOMER_COLUMNS` di generator,
atau download template di halaman *Batch Scoring*), letakkan di `data/raw/customers.csv`, lalu `make train`.
Validasi skema otomatis menolak data yang rusak.

## 🚀 Menjalankan secara lokal

```bash
git clone https://github.com/<username>/telco-churn-intelligence.git
cd telco-churn-intelligence
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

make train          # (opsional) latih ulang semua model  → ±1 menit di CPU
make app            # streamlit run app/Home.py  → http://localhost:8501
```

Tanpa `make` (Windows):

```bash
set PYTHONPATH=src            # PowerShell: $env:PYTHONPATH="src"
python -m telco.pipelines.train
streamlit run app/Home.py
```

> Jika `models/registry` kosong, aplikasi **otomatis** generate data & melatih semua model saat pertama dibuka.

### CLI pipeline

```bash
python -m telco.pipelines.train --regenerate-data                 # data baru + training penuh
python -m telco.pipelines.monitor --scenario severe               # simulasi batch baru + cek drift
python -m telco.pipelines.monitor --scenario severe --auto-retrain  # retrain otomatis jika trigger aktif
python -m telco.pipelines.monitor --batch data/incoming/batch.csv # monitor batch nyata
python -m telco.pipelines.predict --input in.csv --output scored.csv
pytest                                                             # test suite
```

## ☁️ Deploy ke Streamlit Community Cloud

1. Push repository ke GitHub (**termasuk** `data/raw`, `models/registry`, `data/versions` — sudah tidak di-ignore
   agar app langsung siap tanpa training).
2. Buka **share.streamlit.io** → *Create app* → pilih repo & branch.
3. **Main file path**: `app/Home.py`
4. *Advanced settings* → **Python 3.12**.
5. Deploy. `requirements.txt` sudah memakai **PyTorch CPU** (`--extra-index-url https://download.pytorch.org/whl/cpu`)
   agar ukuran instalasi kecil (±200 MB, bukan ±2,5 GB build CUDA).

> **Catatan**: filesystem Streamlit Cloud bersifat sementara. Retraining yang dipicu dari UI hanya bertahan sampai app
> restart. Retraining permanen berjalan lewat GitHub Actions `retrain.yml`, yang meng-commit registry baru →
> Streamlit Cloud otomatis redeploy.

### Docker (Cloud Run, ECS, VPS, dll.)

```bash
docker build -t telco-intelligence .
docker run -p 8501:8501 telco-intelligence
```

### Troubleshooting

| Masalah | Solusi |
|---|---|
| Instalasi torch gagal / terlalu besar di Streamlit Cloud | Pastikan baris `--extra-index-url …/whl/cpu` ada di baris pertama `requirements.txt`; pilih Python 3.12 |
| `ModuleNotFoundError: telco` saat menjalankan CLI | Set `PYTHONPATH=src` atau `pip install -e .` |
| App lambat saat pertama dibuka | Registry kosong → bootstrap training ±1 menit (hanya sekali) |
| Ingin mulai dari nol | `make clean-models && make train` |

## ⚙️ MLOps: alur lengkap

```
data ─► validation ─► data versioning (sha256) ─► feature eng. ─► training (PyTorch) ─► evaluation (vs baseline)
                                                                                  │
                                    ┌────────────── model registry (Staging → Production → Archived) ◄──┘
                                    ▼                                    ▲
                     Streamlit app & batch scoring                champion / challenger
                                    │                                    ▲
                         new labelled batch ─► monitoring (PSI/KS/AUC) ─► retrain trigger
```

**Kebijakan retraining trigger** (`config.yaml → monitoring`) — retrain jika salah satu terpenuhi **dan** tersedia
≥ 1.000 label baru:

| Check | Aturan default |
|---|---|
| Data drift | ≥ 2 fitur dengan PSI ≥ 0,20 |
| Prediction drift | PSI distribusi skor ≥ 0,20 |
| Performance decay | ROC-AUC turun ≥ 0,05 dari AUC saat training |
| Staleness | umur model ≥ 90 hari |

**Champion/challenger**: challenger dievaluasi head-to-head dengan champion pada **data baru yang belum pernah
dilihat champion**; dipromosikan bila AUC ≥ champion − 0,005. Rollback = pindahkan versi lama ke Production
(tersedia di halaman MLOps).

**Data versioning**: setiap dataset di-hash SHA-256 → `customers@vN`. Setiap model menyimpan `data_version` +
git commit (lineage). Registry API dirancang mirip MLflow sehingga bisa diganti ke MLflow server tanpa mengubah pipeline.

## 📈 Hasil (data default, seed 42)

| Model | Metrik | Baseline |
|---|---|---|
| Churn — PyTorch MLP | ROC-AUC ≈ 0,75 · PR-AUC ≈ 0,47 · threshold profit ≈ 0,39 | Logistic regression ≈ 0,76 |
| ARPU — PyTorch MLP | R² ≈ 0,99 · MAPE ≈ 5% | Linear regression R² ≈ 0,84 |
| Forecast subscriber — LSTM | Backtest MAPE ≈ 0,2% | Holt-Winters ≈ 0,5% |
| Forecast revenue — LSTM | Backtest MAPE ≈ 1,3% | Seasonal naive ≈ 1,0% |

**Insight jujur** (penting untuk interview/portofolio):
- Untuk churn, MLP ≈ logistic regression karena relasi fitur–target hampir linear pada skala logit → model yang
  lebih sederhana layak dipilih bila interpretabilitas penting. Untuk ARPU (non-linear: overage kuota, add-on),
  MLP jauh lebih unggul.
- Untuk revenue 60 bulan, baseline seasonal naive bisa mengalahkan LSTM — keputusan model harus berbasis backtest.
- Threshold churn dioptimasi berdasarkan **expected profit** (nilai pelanggan × success rate − biaya kontak), bukan F1.

## 🧪 Testing & CI

`pytest` menjalankan 32 test: generator & validasi, versioning, uji statistik terhadap nilai acuan, metrik bisnis,
model PyTorch (learn + save/load), LSTM, PSI, registry lifecycle, kebijakan trigger, dan **integration test
end-to-end** (train → score → monitor → drift → retrain → champion/challenger). CI juga menjalankan **smoke test
seluruh halaman Streamlit** dengan `streamlit.testing.AppTest`.

## 🔧 Kustomisasi cepat

- Hyperparameter, fitur, asumsi bisnis (margin, biaya kampanye), threshold MLOps → `config/config.yaml`
- Skenario drift baru → `DRIFT_SCENARIOS` di `src/telco/data/generator.py`
- Aksi per segmen → `SEGMENT_ACTIONS` di `src/telco/models/segmentation.py`
