"""File-based model registry with versioning, stages and lineage.

Layout::

    models/registry/
      registry.json                 # index of all models & versions
      <model_name>/v<N>/            # immutable artifact directory per version
        ... artifacts ...
        model_card.md

Stages follow the MLflow convention: ``Staging`` → ``Production`` → ``Archived``.
Only one version per model can be in Production; promoting a new one archives
the previous champion automatically. The API is intentionally similar to
MLflow's so the backend can be swapped for an MLflow tracking server.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

STAGES = ("None", "Staging", "Production", "Archived")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             timeout=3, cwd=Path(__file__).parent)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - git is optional
        return None


class ModelRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.index_path = self.root / "registry.json"

    # ------------------------------------------------------------------ io
    def _load(self) -> dict:
        if self.index_path.exists():
            with open(self.index_path, encoding="utf-8") as fh:
                return json.load(fh)
        return {"models": {}}

    def _save(self, data: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        os.replace(tmp, self.index_path)

    # ------------------------------------------------------------------ write
    def register(self, name: str, artifact_dir: str | Path, metrics: dict, params: dict | None = None,
                 data_version: str | None = None, description: str = "", stage: str = "Staging",
                 tags: dict | None = None) -> dict:
        """Copy ``artifact_dir`` into the registry as a new immutable version."""
        if stage not in STAGES:
            raise ValueError(f"Invalid stage {stage}")
        data = self._load()
        versions = data["models"].setdefault(name, [])
        version = max((v["version"] for v in versions), default=0) + 1
        dest = self.root / name / f"v{version}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(artifact_dir, dest)
        entry = {
            "name": name, "version": version, "stage": "None",
            "created_at": _now(), "updated_at": _now(),
            "metrics": metrics, "params": params or {}, "data_version": data_version,
            "description": description, "tags": tags or {}, "git_commit": _git_commit(),
            "artifact_path": f"{name}/v{version}",
        }
        versions.append(entry)
        self._save(data)
        self._write_model_card(entry)
        if stage != "None":
            entry = self.transition(name, version, stage)
        return entry

    def transition(self, name: str, version: int, stage: str, reason: str = "") -> dict:
        if stage not in STAGES:
            raise ValueError(f"Invalid stage {stage}")
        data = self._load()
        versions = data["models"].get(name)
        if not versions:
            raise KeyError(f"Model '{name}' not found")
        target = next((v for v in versions if v["version"] == version), None)
        if target is None:
            raise KeyError(f"{name} v{version} not found")
        if stage == "Production":
            for v in versions:
                if v["stage"] == "Production" and v["version"] != version:
                    v["stage"] = "Archived"
                    v["updated_at"] = _now()
                    v.setdefault("history", []).append({"at": _now(), "stage": "Archived",
                                                        "reason": f"superseded by v{version}"})
        target["stage"] = stage
        target["updated_at"] = _now()
        target.setdefault("history", []).append({"at": _now(), "stage": stage, "reason": reason})
        self._save(data)
        return target

    def update_metrics(self, name: str, version: int, extra: dict) -> None:
        data = self._load()
        for v in data["models"].get(name, []):
            if v["version"] == version:
                v["metrics"].update(extra)
        self._save(data)

    # ------------------------------------------------------------------ read
    def list_models(self) -> list[str]:
        return list(self._load()["models"])

    def list_versions(self, name: str | None = None) -> list[dict]:
        models = self._load()["models"]
        if name:
            return list(models.get(name, []))
        return [v for vs in models.values() for v in vs]

    def get(self, name: str, version: int) -> dict:
        for v in self.list_versions(name):
            if v["version"] == version:
                return v
        raise KeyError(f"{name} v{version} not found")

    def get_production(self, name: str) -> dict | None:
        prod = [v for v in self.list_versions(name) if v["stage"] == "Production"]
        return prod[-1] if prod else None

    def get_latest(self, name: str, stage: str | None = None) -> dict | None:
        vs = [v for v in self.list_versions(name) if stage is None or v["stage"] == stage]
        return vs[-1] if vs else None

    def artifact_dir(self, entry: dict) -> Path:
        return self.root / entry["artifact_path"]

    def production_dir(self, name: str) -> Path | None:
        entry = self.get_production(name)
        return self.artifact_dir(entry) if entry else None

    # ------------------------------------------------------------------ docs
    def _write_model_card(self, e: dict) -> None:
        metrics = "\n".join(f"| {k} | {v:.4f} |" if isinstance(v, (int, float)) else f"| {k} | {v} |"
                            for k, v in e["metrics"].items())
        params = "\n".join(f"- **{k}**: {v}" for k, v in e["params"].items())
        card = (f"# Model Card — {e['name']} v{e['version']}\n\n"
                f"- Created: {e['created_at']}\n- Data version: `{e['data_version']}`\n"
                f"- Git commit: `{e['git_commit']}`\n- Description: {e['description']}\n\n"
                f"## Metrics\n\n| metric | value |\n|---|---|\n{metrics}\n\n## Parameters\n\n{params}\n")
        (self.root / e["artifact_path"] / "model_card.md").write_text(card, encoding="utf-8")
