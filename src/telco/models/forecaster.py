"""Monthly KPI forecasting with a PyTorch LSTM, plus classical baselines.

The LSTM models monthly log-growth (Δ log y), which makes it trend-agnostic and
able to extrapolate. Each input step also carries the calendar features of the
*next* month (month-of-year sin/cos, Lebaran & post-Lebaran flags) so the
network learns the Idul Fitri seasonality. Multi-step forecasts are recursive.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from telco.data.generator import LEBARAN_MONTHS


def calendar_features(months: pd.DatetimeIndex) -> np.ndarray:
    periods = pd.DatetimeIndex(months).to_period("M")
    moy = periods.month.to_numpy()
    leb = np.array([p.strftime("%Y-%m") in LEBARAN_MONTHS for p in periods], dtype=float)
    post = np.array([(p - 1).strftime("%Y-%m") in LEBARAN_MONTHS for p in periods], dtype=float)
    return np.column_stack([np.sin(2 * np.pi * moy / 12), np.cos(2 * np.pi * moy / 12), leb, post])


class LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class LSTMForecaster:
    def __init__(self, window: int = 12, hidden_size: int = 32, num_layers: int = 1,
                 epochs: int = 400, lr: float = 5e-3, seed: int = 42):
        self.window = window
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.net: LSTMNet | None = None
        self.mu, self.sigma = 0.0, 1.0
        self.history_dates: pd.DatetimeIndex | None = None
        self.history_values: np.ndarray | None = None
        self.residual_std: float = 0.0
        self.loss_history: list[float] = []

    def _sequences(self, growth_scaled: np.ndarray, cal_next: np.ndarray):
        feats = np.column_stack([growth_scaled, cal_next])
        X, y = [], []
        for i in range(self.window, len(growth_scaled)):
            X.append(feats[i - self.window:i])
            y.append(growth_scaled[i])
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

    def fit(self, dates, values) -> LSTMForecaster:
        torch.manual_seed(self.seed)
        dates = pd.DatetimeIndex(dates)
        values = np.asarray(values, dtype=float)
        if len(values) < self.window + 6:
            raise ValueError(f"Need at least {self.window + 6} observations, got {len(values)}")
        if (values <= 0).any():
            raise ValueError("LSTMForecaster models log-growth; values must be positive.")
        growth = np.diff(np.log(values))                     # growth[k] = Δlog between t=k and t=k+1
        self.mu, self.sigma = float(growth.mean()), float(growth.std() or 1.0)
        g = (growth - self.mu) / self.sigma
        cal_next = calendar_features(dates[1:])               # calendar of the month each growth lands on
        # input step k carries growth[k] and calendar of month k+1 (the next target)
        cal_shift = np.vstack([cal_next[1:], calendar_features(dates[-1:] + pd.offsets.MonthBegin(1))])
        # X[j] = steps j..j+window-1, target = growth[j+window]; the last step carries the
        # calendar of the target month, so the model knows if it is predicting a Lebaran month.
        X, y = self._sequences(g, cal_shift)
        self.net = LSTMNet(X.shape[2], self.hidden_size, self.num_layers)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
        best, best_state = np.inf, None
        self.loss_history = []
        for _ in range(self.epochs):
            self.net.train()
            opt.zero_grad()
            loss = loss_fn(self.net(Xt), yt)
            loss.backward()
            opt.step()
            self.loss_history.append(loss.item())
            if loss.item() < best:
                best, best_state = loss.item(), copy.deepcopy(self.net.state_dict())
        self.net.load_state_dict(best_state)
        self.net.eval()
        with torch.no_grad():
            fitted = self.net(Xt).numpy()
        self.residual_std = float(np.std((fitted - y) * self.sigma))
        self.history_dates, self.history_values = dates, values
        self._g_scaled, self._cal_shift = g, cal_shift
        return self

    def predict(self, horizon: int = 6, z: float = 1.96) -> pd.DataFrame:
        if self.net is None:
            raise RuntimeError("Forecaster is not fitted.")
        last = self.history_dates[-1]
        future = pd.date_range(last + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
        g = list(self._g_scaled)
        cal = list(self._cal_shift)
        level = float(self.history_values[-1])
        rows = []
        with torch.no_grad():
            for h, date in enumerate(future, start=1):
                seq = np.column_stack([np.asarray(g[-self.window:]), np.asarray(cal[-self.window:])])
                g_next = float(self.net(torch.from_numpy(seq[None].astype(np.float32)))[0])
                level *= float(np.exp(g_next * self.sigma + self.mu))
                band = z * self.residual_std * np.sqrt(h)
                rows.append({"month": date, "forecast": level,
                             "lower": level * np.exp(-band), "upper": level * np.exp(band)})
                g.append(g_next)
                cal.append(calendar_features(pd.DatetimeIndex([date + pd.offsets.MonthBegin(1)]))[0])
        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> None:
        torch.save({"state_dict": self.net.state_dict(),
                    "params": {"window": self.window, "hidden_size": self.hidden_size,
                               "num_layers": self.num_layers, "epochs": self.epochs, "lr": self.lr,
                               "seed": self.seed},
                    "mu": self.mu, "sigma": self.sigma, "residual_std": self.residual_std,
                    "dates": [d.isoformat() for d in self.history_dates],
                    "values": self.history_values.tolist(),
                    "g_scaled": list(map(float, self._g_scaled)),
                    "cal_shift": np.asarray(self._cal_shift).tolist(),
                    "loss_history": self.loss_history}, path)

    @classmethod
    def load(cls, path: str | Path) -> LSTMForecaster:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(**ckpt["params"])
        obj.mu, obj.sigma, obj.residual_std = ckpt["mu"], ckpt["sigma"], ckpt["residual_std"]
        obj.history_dates = pd.DatetimeIndex(pd.to_datetime(ckpt["dates"]))
        obj.history_values = np.asarray(ckpt["values"])
        obj._g_scaled = np.asarray(ckpt["g_scaled"])
        obj._cal_shift = np.asarray(ckpt["cal_shift"])
        obj.loss_history = ckpt.get("loss_history", [])
        obj.net = LSTMNet(1 + obj._cal_shift.shape[1], obj.hidden_size, obj.num_layers)
        obj.net.load_state_dict(ckpt["state_dict"])
        obj.net.eval()
        return obj


# --------------------------------------------------------------------------- baselines
def seasonal_naive_forecast(values: np.ndarray, horizon: int, season: int = 12) -> np.ndarray:
    """Same month last year × average YoY growth (seasonal naive with drift)."""
    v = np.asarray(values, dtype=float)
    yoy = v[season:] / v[:-season]
    growth = float(np.mean(yoy[-season:])) if len(yoy) else 1.0
    return np.array([v[len(v) - season + (h % season)] * growth ** (1 + h // season) for h in range(horizon)])


def holt_winters_forecast(values: np.ndarray, horizon: int, season: int = 12) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    model = ExponentialSmoothing(np.asarray(values, dtype=float), trend="add", seasonal="mul",
                                 seasonal_periods=season, initialization_method="estimated").fit()
    return np.asarray(model.forecast(horizon))
