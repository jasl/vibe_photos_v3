#!/usr/bin/env python3
"""Phase 1 quick-start script for environment bootstrap."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BLUEPRINT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = BLUEPRINT_ROOT / "config.yaml"


def print_step(step_num: int, total_steps: int, message: str) -> None:
    """Display a formatted step banner."""
    print(f"\n[{step_num}/{total_steps}] {message}")
    print("-" * 50)


def check_python_version() -> bool:
    """Ensure Python 3.12 is available."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 12):
        print("❌ Python 3.12 or later is required.")
        print(f"   Detected version: {sys.version.split()[0]}")
        return False

    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def check_uv_tool() -> bool:
    """Verify that the uv package manager is available."""
    uv_path = shutil.which("uv")
    if uv_path is None:
        print("❌ uv is not installed or not on PATH.")
        print("   Install via `pip install uv` or consult https://docs.astral.sh/uv/.")
        return False

    print(f"✅ uv executable located at: {uv_path}")
    return True


def load_config() -> dict:
    """Load the Phase 1 configuration blueprint."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration blueprint missing: {CONFIG_PATH}")

    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def prepare_runtime_directories() -> bool:
    """Create runtime directories defined in the blueprint configuration."""
    try:
        config = load_config()
    except FileNotFoundError as error:
        print(f"❌ {error}")
        return False

    directories = set()

    dataset_dir = REPO_ROOT / config.get("dataset", {}).get("directory", "samples")
    directories.add(dataset_dir)

    dataset_state = config.get("dataset", {}).get("state_file")
    if dataset_state:
        directories.add((REPO_ROOT / dataset_state).parent)

    preprocessing_paths = config.get("preprocessing", {}).get("paths", {})
    for relative_path in preprocessing_paths.values():
        path = REPO_ROOT / relative_path
        directories.add(path if path.suffix == "" else path.parent)

    temporary_dir = config.get("temporary", {}).get("directory")
    if temporary_dir:
        directories.add(REPO_ROOT / temporary_dir)

    logging_dir = config.get("logging", {}).get("directory")
    if logging_dir:
        directories.add(REPO_ROOT / logging_dir)

    for directory in sorted(directories):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"✅ Prepared {len(directories)} runtime directories.")
    return True


def ensure_settings_file() -> bool:
    """Ensure config/settings.yaml exists by copying from the template if necessary."""
    settings_path = REPO_ROOT / "config" / "settings.yaml"
    sample_path = REPO_ROOT / "config" / "settings.yaml.sample"

    if settings_path.exists():
        print("✅ config/settings.yaml already exists.")
        return True

    if not sample_path.exists():
        print("❌ Template file config/settings.yaml.sample is missing.")
        return False

    shutil.copyfile(sample_path, settings_path)
    print("✅ Created config/settings.yaml from template. Update the file before running pipelines.")
    return True


def print_next_steps() -> bool:
    """Display follow-up commands for the contributor."""
    print("✅ Environment bootstrap checklist complete. Next actions:")
    print("   • Create a virtual environment: `uv venv --python 3.12` and activate it.")
    print("   • Install dependencies: `uv sync` (run from the repository root).")
    print("   • Download perception models: `uv run python blueprints/phase1/download_models.py`.")
    print("   • Process the sample dataset: `uv run python blueprints/phase1/process_dataset.py`.")
    print("   • Review `blueprints/phase1/README.md` for API and UI launch instructions.")
    return True


def main() -> int:
    """Execute the quick-start checklist."""
    steps: List[Tuple[str, Callable[[], bool]]] = [
        ("Verify Python 3.12 availability", check_python_version),
        ("Check uv installation", check_uv_tool),
        ("Prepare runtime directories", prepare_runtime_directories),
        ("Ensure config/settings.yaml exists", ensure_settings_file),
        ("Review follow-up commands", print_next_steps),
    ]

    total_steps = len(steps)

    for index, (message, action) in enumerate(steps, start=1):
        print_step(index, total_steps, message)
        if not action():
            print("\nHalting quick-start due to the issue above. Resolve it and rerun the script.")
            return 1

    print("\nAll quick-start steps completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
