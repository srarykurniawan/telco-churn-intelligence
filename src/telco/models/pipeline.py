"""End-to-end inference pipeline = feature engineering + preprocessor + PyTorch model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from telco.features.engineering import prepare_frame
from telco.models.torch_tabular import TorchTabularModel


class TabularPipeline:
    def __init__(self, preprocessor, model: TorchTabularModel, numeric: list[str],
                 categorical: list[str], target_transform: str | None = None,
                 threshold: float = 0.5, meta: dict | None = None):
        self.preprocessor = preprocessor
        self.model = model
        self.numeric = numeric
        self.categorical = categorical
        self.target_transform = target_transform
        self.threshold = threshold
        self.meta = meta or {}

    @property
    def features(self) -> list[str]:
        return self.numeric + self.categorical

    @property
    def task(self) -> str:
        return self.model.task

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        frame = prepare_frame(df, self.numeric, self.categorical)
        return self.preprocessor.transform(frame).astype(np.float32)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Churn probability for binary task, value in original units for regression."""
        X = self.transform(df)
        if self.task == "binary":
            return self.model.predict_proba(X)
        pred = self.model.predict(X)
        return np.expm1(pred) if self.target_transform == "log1p" else pred

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, directory / "preprocessor.joblib")
        self.model.save(directory / "model.pt")
        meta = {"numeric": self.numeric, "categorical": self.categorical,
                "target_transform": self.target_transform, "threshold": self.threshold,
                "framework": "pytorch", **self.meta}
        (directory / "pipeline.json").write_text(json.dumps(meta, indent=2, default=str))
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> TabularPipeline:
        directory = Path(directory)
        meta = json.loads((directory / "pipeline.json").read_text())
        return cls(
            preprocessor=joblib.load(directory / "preprocessor.joblib"),
            model=TorchTabularModel.load(directory / "model.pt"),
            numeric=meta.pop("numeric"), categorical=meta.pop("categorical"),
            target_transform=meta.pop("target_transform", None),
            threshold=float(meta.pop("threshold", 0.5)), meta=meta,
        )
