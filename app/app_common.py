"""Shared helpers for all Streamlit pages: bootstrap, caching, formatting, theming."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402
import streamlit as st  # noqa: E402

from telco.config import Config, load_config  # noqa: E402
from telco.constants import CHURN_MODEL, SEGMENT_MODEL  # noqa: E402
from telco.mlops.registry import ModelRegistry  # noqa: E402
from telco.models.pipeline import TabularPipeline  # noqa: E402
from telco.models.segmentation import CustomerSegmenter  # noqa: E402

# --------------------------------------------------------------------------- theme
PALETTE = ["#C8102E", "#1F4E79", "#2E8B57", "#E39B19", "#6A4C93", "#5B8E7D", "#8C8C8C", "#D96C75"]
RISK_COLORS = {"High": "#C8102E", "Medium": "#E39B19", "Low": "#2E8B57"}
STATUS_COLORS = {"Drift": "#C8102E", "Warning": "#E39B19", "OK": "#2E8B57", "N/A": "#8C8C8C"}

pio.templates["telco"] = go.layout.Template(
    layout=go.Layout(colorway=PALETTE, font=dict(family="Inter, Segoe UI, sans-serif", size=13),
                     margin=dict(l=10, r=10, t=50, b=10), hoverlabel=dict(font_size=12),
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
)
pio.templates.default = "plotly_white+telco"

CSS = """
<style>
  .block-container {padding-top: 2rem; padding-bottom: 2rem;}
  [data-testid="stMetricValue"] {font-size: 1.55rem;}
  .tag {display:inline-block;padding:2px 10px;border-radius:999px;font-size:0.8rem;font-weight:600;
        background:rgba(200,16,46,.1);color:#C8102E;margin-right:6px;}
  .muted {color:#6b7280;font-size:0.9rem;}
</style>
"""


# --------------------------------------------------------------------------- config / bootstrap
@st.cache_resource(show_spinner=False)
def get_config() -> Config:
    return load_config()


@st.cache_resource(show_spinner=False)
def _train_lock() -> threading.Lock:
    return threading.Lock()


def registry() -> ModelRegistry:
    return ModelRegistry(get_config().path("registry_dir"))


def is_ready() -> bool:
    cfg = get_config()
    return (cfg.path("raw_data").exists() and cfg.path("timeseries_data").exists()
            and registry().get_production(CHURN_MODEL) is not None)


def ensure_ready() -> None:
    """First boot on a fresh deployment: generate data & train all models once."""
    if is_ready():
        return
    with _train_lock():
        if is_ready():
            return
        from telco.pipelines.train import run_training

        with st.status("Pertama kali dijalankan: generate data & training semua model (±1 menit)…",
                       expanded=True) as status:
            run_training(get_config(), log=lambda m: status.write(m))
            status.update(label="Model siap ✔", state="complete", expanded=False)
        clear_caches()


def clear_caches() -> None:
    st.cache_data.clear()
    load_pipeline.clear()
    load_segmenter.clear()


def page_setup(title: str, icon: str, subtitle: str | None = None) -> Config:
    st.set_page_config(page_title=f"{title} · Telco Intelligence", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    ensure_ready()
    st.title(f"{icon} {title}")
    if subtitle:
        st.caption(subtitle)
    _sidebar_footer()
    return get_config()


def _sidebar_footer() -> None:
    entry = registry().get_production(CHURN_MODEL)
    with st.sidebar:
        st.markdown("---")
        if entry:
            st.caption(f"**Churn model** v{entry['version']} · AUC {entry['metrics'].get('test_roc_auc', 0):.3f}  \n"
                       f"Data: `{entry.get('data_version')}`")
        st.caption("Stack: PyTorch · scikit-learn · statsmodels · Streamlit")


# --------------------------------------------------------------------------- data loaders
@st.cache_data(show_spinner=False)
def _read_csv(path: str, mtime: float, parse_dates: tuple[str, ...] = ()) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=list(parse_dates) or None)


def load_customers() -> pd.DataFrame:
    p = get_config().path("raw_data")
    return _read_csv(str(p), p.stat().st_mtime)


def load_kpis() -> pd.DataFrame:
    p = get_config().path("timeseries_data")
    return _read_csv(str(p), p.stat().st_mtime, ("month",))


def read_artifact_csv(entry: dict, filename: str, parse_dates: tuple[str, ...] = ()) -> pd.DataFrame | None:
    p = registry().artifact_dir(entry) / filename
    if not p.exists():
        return None
    return _read_csv(str(p), p.stat().st_mtime, parse_dates)


@st.cache_resource(show_spinner=False)
def load_pipeline(path: str, version: int) -> TabularPipeline:  # version busts the cache
    return TabularPipeline.load(path)


@st.cache_resource(show_spinner=False)
def load_segmenter(path: str, version: int) -> CustomerSegmenter:
    return CustomerSegmenter.load(Path(path) / "segmenter.joblib")


def production(name: str):
    entry = registry().get_production(name)
    if entry is None:
        return None, None
    path = str(registry().artifact_dir(entry))
    if name == SEGMENT_MODEL:
        return load_segmenter(path, entry["version"]), entry
    return load_pipeline(path, entry["version"]), entry


# --------------------------------------------------------------------------- formatting
def idr(x: float, short: bool = True) -> str:
    """Indonesian Rupiah formatting: Rp 52.300 · Rp 1,2 jt · Rp 3,4 M · Rp 1,1 T."""
    if x is None or pd.isna(x):
        return "–"
    sign = "-" if x < 0 else ""
    x = abs(float(x))
    if short:
        for div, suf in ((1e12, " T"), (1e9, " M"), (1e6, " jt")):
            if x >= div:
                return f"{sign}Rp {x / div:,.2f}{suf}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{sign}Rp {x:,.0f}".replace(",", ".")


def pct(x: float, digits: int = 1) -> str:
    return "–" if x is None or pd.isna(x) else f"{x * 100:.{digits}f}%".replace(".", ",")


def num(x: float, digits: int = 0) -> str:
    return "–" if x is None or pd.isna(x) else f"{x:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def sidebar_filters(df: pd.DataFrame, key: str = "f") -> pd.DataFrame:
    with st.sidebar:
        st.subheader("Filter")
        regions = st.multiselect("Region", sorted(df["region"].unique()), key=f"{key}_region")
        plans = st.multiselect("Plan", sorted(df["plan_type"].unique()), key=f"{key}_plan")
        packages = st.multiselect("Package", ["Basic", "Standard", "Premium", "Unlimited"], key=f"{key}_pkg")
    out = df
    if regions:
        out = out[out["region"].isin(regions)]
    if plans:
        out = out[out["plan_type"].isin(plans)]
    if packages:
        out = out[out["package"].isin(packages)]
    if out.empty:
        st.warning("Tidak ada data untuk kombinasi filter ini.")
        st.stop()
    return out


def show_result(result) -> None:
    """Render a TestResult consistently."""
    c1, c2, c3 = st.columns(3)
    c1.metric("Statistik uji", f"{result.statistic:.4f}")
    c2.metric("p-value", f"{result.p_value:.4g}")
    if result.effect_size is not None:
        c3.metric(f"Effect size ({result.effect_name})", f"{result.effect_size:.3f}", result.effect_magnitude,
                  delta_color="off")
    (st.success if result.significant else st.info)(f"**{result.test}** — H0: {result.null_hypothesis}.  \n"
                                                     f"{result.conclusion}")
