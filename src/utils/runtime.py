"""Helpers for reconciling blueprint configs with runtime settings."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

from .config import DEFAULT_SETTINGS_PATH

BLUEPRINT_CONFIG = Path(__file__).resolve().parents[2] / "blueprints" / "phase1" / "config.yaml"


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionaries."""
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_phase1_config() -> Dict[str, Any]:
    """Return the effective configuration for Phase 1 processors."""
    if not BLUEPRINT_CONFIG.exists():
        raise FileNotFoundError("Blueprint config file missing. Verify repository checkout.")

    with BLUEPRINT_CONFIG.open(encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle) or {}

    settings_overrides: Dict[str, Any] = {}
    if DEFAULT_SETTINGS_PATH.exists():
        with DEFAULT_SETTINGS_PATH.open(encoding="utf-8") as handle:
            settings_overrides = yaml.safe_load(handle) or {}

    return _deep_merge(base_config, settings_overrides)
