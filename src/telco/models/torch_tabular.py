"""PyTorch MLP for tabular data — binary classification (churn) and regression (ARPU).

Features: BatchNorm + Dropout MLP, AdamW, ReduceLROnPlateau, early stopping on
the validation metric (ROC-AUC or RMSE) with best-weights restore, optional
class weighting, and a portable save/load format.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TorchTabularModel:
    """sklearn-like wrapper around :class:`MLP`. ``task`` ∈ {'binary', 'regression'}."""

    def __init__(self, task: str = "binary", hidden_dims=(128, 64, 32), dropout: float = 0.2,
                 lr: float = 1e-3, weight_decay: float = 1e-5, batch_size: int = 256,
                 max_epochs: int = 50, patience: int = 6, use_class_weight: bool = False,
                 seed: int = 42):
        if task not in {"binary", "regression"}:
            raise ValueError("task must be 'binary' or 'regression'")
        self.task = task
        self.hidden_dims = list(hidden_dims)
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.use_class_weight = use_class_weight
        self.seed = seed
        self.model: MLP | None = None
        self.input_dim: int | None = None
        self.y_mean, self.y_std = 0.0, 1.0
        self.history: list[dict] = []
        self.best_epoch: int = 0

    # ------------------------------------------------------------------ training
    def _val_score(self, X_val: np.ndarray, y_val: np.ndarray) -> tuple[float, float]:
        """Return (loss-like value, score where higher is better)."""
        raw = self._forward_numpy(X_val)
        if self.task == "binary":
            p = 1 / (1 + np.exp(-raw))
            eps = 1e-7
            loss = -np.mean(y_val * np.log(p + eps) + (1 - y_val) * np.log(1 - p + eps))
            score = roc_auc_score(y_val, p) if len(np.unique(y_val)) > 1 else -loss
            return float(loss), float(score)
        pred = raw * self.y_std + self.y_mean
        rmse = float(np.sqrt(np.mean((pred - y_val) ** 2)))
        return rmse, -rmse

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
            verbose: bool = False) -> TorchTabularModel:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)
        self.input_dim = X.shape[1]
        self.model = MLP(self.input_dim, self.hidden_dims, self.dropout)

        if self.task == "binary":
            pos = max(y.sum(), 1.0)
            weight = torch.tensor((len(y) - pos) / pos) if self.use_class_weight else None
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
            y_train = y
        else:
            self.y_mean, self.y_std = float(y.mean()), float(y.std() or 1.0)
            loss_fn = nn.MSELoss()
            y_train = (y - self.y_mean) / self.y_std

        loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y_train.astype(np.float32))),
                            batch_size=self.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(self.seed), drop_last=len(X) > self.batch_size)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

        best_score, best_state, bad_epochs = -np.inf, None, 0
        self.history = []
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            running, seen = 0.0, 0
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
                running += loss.item() * len(xb)
                seen += len(xb)
            val_loss, val_score = self._val_score(X_val, y_val)
            sched.step(val_score)
            self.history.append({"epoch": epoch, "train_loss": running / max(seen, 1),
                                 "val_loss": val_loss, "val_score": val_score,
                                 "lr": opt.param_groups[0]["lr"]})
            if verbose:
                print(self.history[-1])
            if val_score > best_score + 1e-5:
                best_score, best_state, bad_epochs = val_score, copy.deepcopy(self.model.state_dict()), 0
                self.best_epoch = epoch
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    # ------------------------------------------------------------------ inference
    def _forward_numpy(self, X: np.ndarray, batch: int = 8192) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        self.model.eval()
        X = np.asarray(X, dtype=np.float32)
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), batch):
                outs.append(self.model(torch.from_numpy(X[i:i + batch])).numpy())
        return np.concatenate(outs) if outs else np.empty(0, dtype=np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "binary":
            raise RuntimeError("predict_proba only for binary task")
        return 1 / (1 + np.exp(-self._forward_numpy(X).astype(np.float64)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self._forward_numpy(X).astype(np.float64)
        if self.task == "binary":
            return (1 / (1 + np.exp(-raw)) >= 0.5).astype(int)
        return raw * self.y_std + self.y_mean

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        torch.save({
            "state_dict": self.model.state_dict(),
            "config": {"task": self.task, "hidden_dims": self.hidden_dims, "dropout": self.dropout,
                       "lr": self.lr, "weight_decay": self.weight_decay, "batch_size": self.batch_size,
                       "max_epochs": self.max_epochs, "patience": self.patience,
                       "use_class_weight": self.use_class_weight, "seed": self.seed},
            "input_dim": self.input_dim, "y_mean": self.y_mean, "y_std": self.y_std,
            "best_epoch": self.best_epoch,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> TorchTabularModel:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(**ckpt["config"])
        obj.input_dim = ckpt["input_dim"]
        obj.y_mean, obj.y_std = ckpt["y_mean"], ckpt["y_std"]
        obj.best_epoch = ckpt.get("best_epoch", 0)
        obj.model = MLP(obj.input_dim, obj.hidden_dims, obj.dropout)
        obj.model.load_state_dict(ckpt["state_dict"])
        obj.model.eval()
        return obj
