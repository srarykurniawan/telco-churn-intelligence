"""Configuration loading with deep-merge overrides and project-relative paths."""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def deep_merge(base: dict, override: Mapping) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Config(dict):
    """A dict with helpers for resolving paths relative to the project root."""

    def path(self, key: str) -> Path:
        try:
            raw = self["paths"][key]
        except KeyError as exc:
            raise KeyError(f"Path '{key}' is not defined in config.paths") from exc
        p = Path(raw)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def section(self, name: str) -> dict[str, Any]:
        return self.get(name, {}) or {}

    @property
    def seed(self) -> int:
        return int(self.section("project").get("seed", 42))


def load_config(path: str | Path | None = None, overrides: Mapping | None = None) -> Config:
    """Load YAML config. Priority: explicit path > $TELCO_CONFIG > default."""
    cfg_path = Path(path or os.environ.get("TELCO_CONFIG") or DEFAULT_CONFIG_PATH)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if overrides:
        data = deep_merge(data, overrides)
    return Config(data)
