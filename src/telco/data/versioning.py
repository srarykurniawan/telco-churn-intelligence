"""Lightweight, content-addressed data versioning.

Every dataset used for training is hashed (SHA-256) and recorded in a JSON
manifest together with schema and summary statistics. Model registry entries
reference the ``version_id`` → full data lineage. The approach mirrors what
DVC does (content hashing + metadata) without requiring remote storage, and
``dvc.yaml`` in the repo shows how to plug DVC in when needed.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from telco.config import PROJECT_ROOT


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    os.replace(tmp, path)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


class DataVersionManager:
    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)

    def _load(self) -> dict:
        if self.manifest_path.exists():
            with open(self.manifest_path, encoding="utf-8") as fh:
                return json.load(fh)
        return {"datasets": {}}

    def register(self, path: str | Path, name: str, description: str = "",
                 target: str | None = "churn") -> dict:
        """Register a dataset file. Idempotent: same content → same version."""
        path = Path(path)
        digest = file_sha256(path)
        manifest = self._load()
        versions = manifest["datasets"].setdefault(name, [])
        for v in versions:
            if v["sha256"] == digest:
                return v
        df = pd.read_csv(path)
        entry = {
            "version_id": f"{name}@v{len(versions) + 1}",
            "version": len(versions) + 1,
            "sha256": digest,
            "path": _rel(path),
            "rows": int(len(df)),
            "columns": list(df.columns),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "description": description,
        }
        if target and target in df.columns:
            entry["target_rate"] = round(float(pd.to_numeric(df[target], errors="coerce").mean()), 4)
        versions.append(entry)
        _atomic_write_json(self.manifest_path, manifest)
        return entry

    def list(self, name: str | None = None) -> list[dict]:
        manifest = self._load()["datasets"]
        if name:
            return list(manifest.get(name, []))
        return [v | {"dataset": k} for k, vs in manifest.items() for v in vs]

    def latest(self, name: str) -> dict | None:
        versions = self.list(name)
        return versions[-1] if versions else None
