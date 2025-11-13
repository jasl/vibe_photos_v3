"""Helpers for loading and validating Vibe Photos runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, MutableMapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"


class ConfigurationError(RuntimeError):
    """Raised when the runtime configuration cannot be loaded or is invalid."""


@dataclass(slots=True)
class Settings:
    """Lightweight wrapper around the raw YAML structure."""

    data: Dict[str, Any]

    def get(self, key: str, default: Any | None = None) -> Any:
        """Retrieve a top-level value from the settings."""
        return self.data.get(key, default)

    @property
    def dataset_dir(self) -> Path:
        """Return the dataset directory path."""
        dataset_config: MutableMapping[str, Any] = (
            self.data.get("dataset") if isinstance(self.data.get("dataset"), MutableMapping) else {}
        )
        directory = dataset_config.get("directory", "samples")
        return (REPO_ROOT / directory).resolve()

    @property
    def database_url(self) -> str:
        """Return the SQLAlchemy connection string."""
        db_config = self.data.get("database") or {}
        url = db_config.get("url") or db_config.get("path")
        if not url:
            raise ConfigurationError("Missing database connection details in settings.yaml")

        if isinstance(url, str) and url.startswith("sqlite:///"):
            return url

        if db_config.get("type") == "sqlite":
            return f"sqlite:///{(REPO_ROOT / url).resolve()}"

        return url

    @property
    def logging_dir(self) -> Path:
        """Return the directory used for log files."""
        logging_config = self.data.get("logging") or {}
        directory = logging_config.get("directory", "log")
        return (REPO_ROOT / directory).resolve()

    @property
    def cache_paths(self) -> Dict[str, Path]:
        """Return resolved cache directories configured under preprocessing.paths."""
        preprocessing = self.data.get("preprocessing") or {}
        paths = preprocessing.get("paths") or {}
        resolved: Dict[str, Path] = {}
        for key, relative_path in paths.items():
            path = (REPO_ROOT / relative_path).resolve()
            resolved[key] = path if path.suffix == "" else path.parent
        return resolved


@lru_cache(maxsize=1)
def load_settings(path: str | Path | None = None) -> Settings:
    """Load the YAML settings file and return a typed wrapper."""
    target = Path(path) if path else DEFAULT_SETTINGS_PATH
    if not target.exists():
        raise ConfigurationError(
            f"Settings file not found at {target}. Run ./init_project.sh to scaffold it."
        )

    with target.open(encoding="utf-8") as handle:
        data: Dict[str, Any] = yaml.safe_load(handle) or {}

    return Settings(data=data)


def reload_settings(path: str | Path | None = None) -> Settings:
    """Clear the cached settings and reload from disk."""
    load_settings.cache_clear()
    return load_settings(path=path)
