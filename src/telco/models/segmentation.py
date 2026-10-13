"""Behavioural customer segmentation with K-Means (k chosen by silhouette score)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEGMENT_ACTIONS = {
    "High-Value Power Users": "Lindungi: program VIP, prioritas jaringan 5G, bundling konten premium.",
    "At-Risk Dissatisfied": "Service recovery: tindak lanjut keluhan < 24 jam, kompensasi kuota, cek kualitas jaringan area.",
    "New Joiners": "Onboarding: welcome pack, edukasi aplikasi, bonus top-up bulan ke-2 & ke-3.",
    "Low-Engagement Budget": "Aktivasi: paket hemat mikro, gamifikasi top-up, cross-sell e-wallet.",
    "Loyal Mainstream": "Upsell: tawarkan upgrade paket / add-on streaming, loyalty points.",
}


class CustomerSegmenter:
    def __init__(self, features: list[str], log_features: list[str] | None = None,
                 k_min: int = 3, k_max: int = 7, sample_size: int = 6000, seed: int = 42):
        self.features = list(features)
        self.log_features = list(log_features or [])
        self.k_min, self.k_max = k_min, k_max
        self.sample_size = sample_size
        self.seed = seed
        self.scaler = StandardScaler()
        self.kmeans: KMeans | None = None
        self.pca: PCA | None = None
        self.silhouette_scores: dict[int, float] = {}
        self.segment_names: dict[int, str] = {}
        self.profile_: pd.DataFrame | None = None

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].apply(pd.to_numeric, errors="coerce").copy()
        X = X.fillna(X.median(numeric_only=True)).fillna(0)
        for c in self.log_features:
            X[c] = np.log1p(X[c].clip(lower=0))
        return X.to_numpy(dtype=float)

    def fit(self, df: pd.DataFrame) -> CustomerSegmenter:
        X = self.scaler.fit_transform(self._matrix(df))
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(X), min(self.sample_size, len(X)), replace=False)
        for k in range(self.k_min, self.k_max + 1):
            km = KMeans(n_clusters=k, n_init=5, random_state=self.seed).fit(X[idx])
            self.silhouette_scores[k] = float(silhouette_score(X[idx], km.labels_))
        best_k = max(self.silhouette_scores, key=self.silhouette_scores.get)
        self.kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=self.seed).fit(X)
        self.pca = PCA(n_components=2, random_state=self.seed).fit(X)
        self._name_segments(df, self.kmeans.labels_)
        return self

    @property
    def k(self) -> int:
        return int(self.kmeans.n_clusters) if self.kmeans else 0

    def _name_segments(self, df: pd.DataFrame, labels: np.ndarray) -> None:
        prof = df[self.features].apply(pd.to_numeric, errors="coerce").assign(_c=labels).groupby("_c").mean()
        z = (prof - prof.mean()) / prof.std(ddof=0).replace(0, 1)
        rules = [
            ("High-Value Power Users", z["arpu_idr"] + 0.5 * z["data_usage_gb"]),
            ("At-Risk Dissatisfied", z["num_complaints"] - z["nps_score"]),
            ("New Joiners", -z["tenure_months"]),
            ("Low-Engagement Budget", -(z["data_usage_gb"] + z["arpu_idr"])),
            ("Loyal Mainstream", z["tenure_months"] + z["nps_score"]),
        ]
        names: dict[int, str] = {}
        for name, score in rules:
            remaining = score.drop(index=list(names), errors="ignore")
            if remaining.empty:
                break
            names[int(remaining.idxmax())] = name
        for c in prof.index:
            names.setdefault(int(c), f"Mainstream {int(c)}")
        self.segment_names = names
        counts = pd.Series(labels).value_counts()
        prof = prof.assign(customers=counts, share=counts / counts.sum())
        prof.index = [names[int(i)] for i in prof.index]
        self.profile_ = prof.sort_values("customers", ascending=False)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.kmeans.predict(self.scaler.transform(self._matrix(df)))

    def predict_names(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([self.segment_names[int(c)] for c in self.predict(df)], dtype=object)

    def project_2d(self, df: pd.DataFrame) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(self._matrix(df)))

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> CustomerSegmenter:
        return joblib.load(path)
