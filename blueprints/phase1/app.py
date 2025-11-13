"""Streamlit entry point for the Phase 1 MVP dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"


@st.cache_resource(show_spinner=False)
def load_settings() -> Dict[str, Any]:
    """Load runtime settings for display."""
    if not SETTINGS_PATH.exists():
        raise FileNotFoundError(
            "Missing config/settings.yaml. Run ./init_project.sh to create it from the template."
        )

    with SETTINGS_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def format_supported_formats(formats: List[str]) -> str:
    """Return a human-readable list of supported file extensions."""
    return ", ".join(extension.upper() for extension in formats)


def render_sidebar(settings: Dict[str, Any]) -> None:
    """Render dataset information and quick links inside the sidebar."""
    st.sidebar.header("Project Checklist")
    st.sidebar.markdown("1. Populate **samples/** with test images.")
    st.sidebar.markdown("2. Run **uv sync** to install dependencies.")
    st.sidebar.markdown(
        "3. Execute **uv run python blueprints/phase1/process_dataset.py** before browsing results."
    )
    st.sidebar.markdown("4. Keep `config/settings.yaml` aligned with your environment paths.")

    dataset_settings = settings.get("dataset", {})
    st.sidebar.subheader("Dataset Overview")
    st.sidebar.write("Directory:", dataset_settings.get("directory", "samples"))
    st.sidebar.write(
        "Supported formats:",
        format_supported_formats(dataset_settings.get("supported_formats", [])),
    )
    st.sidebar.write(
        "Incremental mode:", "Enabled" if dataset_settings.get("incremental", True) else "Disabled"
    )


def render_header() -> None:
    """Display the main page header."""
    st.title("Vibe Photos — Phase 1 Dashboard")
    st.caption(
        "Browse processed assets, inspect metadata, and validate the ingestion pipeline output."
    )


def render_status_panels(settings: Dict[str, Any]) -> None:
    """Render placeholder sections for the MVP features."""
    st.subheader("Ingestion Status")
    st.info(
        "Connect the batch processing telemetry here. Write processed counts to a lightweight status JSON and load it for this view."
    )

    st.subheader("Search Preview")
    st.warning(
        "Hook the FastAPI search endpoint once it is available. Display ranked results with thumbnails and metadata snippets."
    )

    st.subheader("Gallery Browser")
    st.info(
        "Render paginated thumbnails from cache/images/thumbnails after dataset processing."
    )

    st.subheader("Configuration Snapshot")
    st.json(settings)


def render_missing_settings_error(error: FileNotFoundError) -> None:
    """Render guidance when the settings file is unavailable."""
    st.error(str(error))
    st.markdown(
        "Run `./init_project.sh` or copy `config/settings.yaml.sample` to `config/settings.yaml` before launching the dashboard."
    )


def main() -> None:
    """Render the Streamlit dashboard."""
    try:
        settings = load_settings()
    except FileNotFoundError as error:
        render_missing_settings_error(error)
        return

    render_sidebar(settings)
    render_header()
    render_status_panels(settings)


if __name__ == "__main__":
    main()
