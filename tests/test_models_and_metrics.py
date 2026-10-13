import numpy as np
import pandas as pd
import pytest

from telco.evaluation import metrics as em
from telco.mlops.monitoring import drift_report, psi_categorical, psi_numeric
from telco.models.forecaster import LSTMForecaster, seasonal_naive_forecast
from telco.models.torch_tabular import TorchTabularModel

rng = np.random.default_rng(0)


def _binary_data(n=3000):
    X = rng.normal(size=(n, 5)).astype(np.float32)
    logit = 2 * X[:, 0] - 1.5 * X[:, 1] + X[:, 2] * X[:, 3]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def test_mlp_classifier_learns_and_roundtrips(tmp_path):
    X, y = _binary_data()
    m = TorchTabularModel("binary", hidden_dims=[32, 16], max_epochs=15, patience=4).fit(X[:2000], y[:2000], X[2000:], y[2000:])
    p = m.predict_proba(X[2000:])
    assert em.classification_metrics(y[2000:], p)["roc_auc"] > 0.8
    m.save(tmp_path / "m.pt")
    np.testing.assert_allclose(TorchTabularModel.load(tmp_path / "m.pt").predict_proba(X[2000:]), p, rtol=1e-5)


def test_mlp_regressor():
    X = rng.normal(size=(2000, 3)).astype(np.float32)
    y = 3 * X[:, 0] + X[:, 1] ** 2 + rng.normal(0, 0.1, 2000)
    m = TorchTabularModel("regression", hidden_dims=[32], max_epochs=40, patience=8, lr=3e-3).fit(X[:1500], y[:1500], X[1500:], y[1500:])
    assert em.regression_metrics(y[1500:], m.predict(X[1500:]))["r2"] > 0.85


def test_lstm_forecaster(tmp_path):
    dates = pd.date_range("2021-01-01", periods=48, freq="MS")
    t = np.arange(48)
    y = 100 * (1 + 0.01 * t) * (1 + 0.05 * np.sin(2 * np.pi * t / 12))
    f = LSTMForecaster(window=12, hidden_size=8, epochs=60).fit(dates, y)
    fc = f.predict(6)
    assert len(fc) == 6 and (fc["lower"] <= fc["forecast"]).all() and (fc["forecast"] <= fc["upper"]).all()
    f.save(tmp_path / "f.pt")
    np.testing.assert_allclose(LSTMForecaster.load(tmp_path / "f.pt").predict(6)["forecast"], fc["forecast"], rtol=1e-5)
    assert len(seasonal_naive_forecast(y, 6)) == 6


def test_lstm_rejects_short_series():
    with pytest.raises(ValueError):
        LSTMForecaster(window=12).fit(pd.date_range("2024-01-01", periods=10, freq="MS"), np.ones(10))


def test_classification_metrics_and_lift():
    y = np.array([0] * 90 + [1] * 10)
    p = np.r_[np.linspace(0, 0.4, 90), np.linspace(0.6, 1, 10)]
    m = em.classification_metrics(y, p, 0.5)
    assert m["roc_auc"] == 1.0 and m["recall"] == 1.0 and m["ks"] == 1.0
    lt = em.lift_table(y, p)
    assert lt["lift"].iloc[0] == pytest.approx(10.0)
    assert lt["cum_capture"].iloc[-1] == pytest.approx(1.0)


def test_profit_threshold():
    y = np.array([0] * 80 + [1] * 20)
    p = np.r_[np.full(80, 0.1), np.full(20, 0.9)]
    thr, profit = em.optimal_threshold(y, p, value_if_saved=1000, cost_per_contact=10, success_rate=0.5)
    assert 0.1 < thr <= 0.9 and profit == pytest.approx(20 * 500 - 20 * 10)


def test_psi():
    a = rng.normal(size=5000)
    assert psi_numeric(a, rng.normal(size=5000)) < 0.02
    assert psi_numeric(a, rng.normal(1, 1, 5000)) > 0.2
    assert psi_categorical(["a"] * 50 + ["b"] * 50, ["a"] * 90 + ["b"] * 10) > 0.2
    ref = pd.DataFrame({"x": a, "c": ["a", "b"] * 2500})
    cur = pd.DataFrame({"x": a + 2, "c": ["a", "b"] * 2500})
    rep = drift_report(ref, cur, ["x"], ["c"])
    assert rep.set_index("feature").loc["x", "status"] == "Drift"
    assert rep.set_index("feature").loc["c", "status"] == "OK"


def test_permutation_importance():
    df = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    y = (df["a"] > 0).astype(int).to_numpy()
    imp = em.permutation_importance(lambda d: d["a"].to_numpy(), df, y, ["a", "b"], n_repeats=2)
    assert imp.iloc[0]["feature"] == "a" and imp.set_index("feature").loc["b", "importance"] == 0
