"""Streamlit entry point for the Phase 1 MVP dashboard."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import yaml

from src.core.database import AssetRepository, get_db_session
from src.core.searcher import AssetSearchService

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
    """Render ingestion metrics, search, and gallery previews."""
    stats = load_ingestion_stats()
    st.subheader("Ingestion Status")
    st.metric("Indexed assets", stats["total_assets"])
    st.metric("Last refresh", stats["refreshed_at"])

    st.subheader("Search Preview")
    render_search_panel()

    st.subheader("Configuration Snapshot")
    st.json(settings)


@st.cache_resource(show_spinner=False)
def get_search_service() -> AssetSearchService:
    """Instantiate the shared search service."""
    return AssetSearchService()


@st.cache_data(ttl=15, show_spinner=False)
def load_ingestion_stats() -> Dict[str, Any]:
    """Return ingestion stats pulled from SQLite."""
    session = get_db_session()
    repo = AssetRepository(session)
    total_assets = repo.total_assets()
    session.close()
    return {"total_assets": total_assets, "refreshed_at": st.session_state.get("last_refresh", "now")}


def render_search_panel() -> None:
    """Render a lightweight search UI backed by the SQLite metadata store."""
    query = st.text_input("Query", placeholder="e.g. recipe cards from 2024")
    if not query:
        st.info("Enter a query to preview ranked results.")
        return

    search_service = get_search_service()
    hits = search_service.search(query)
    if not hits:
        st.warning("No matches found.")
        return

    for hit in hits:
        asset = hit.data
        thumbnail_path = asset.get("thumbnail_path")
        cols = st.columns([1, 3])
        with cols[0]:
            if thumbnail_path and Path(thumbnail_path).exists():
                st.image(thumbnail_path, use_column_width=True)
        with cols[1]:
            st.markdown(f"**{asset['filename']}** — score {hit.score:.2f}")
            if asset["captions"]:
                st.caption(asset["captions"][0]["text"])
            labels = ", ".join(label["label"] for label in asset["labels"])
            if labels:
                st.write("Labels:", labels)


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

    st.session_state["last_refresh"] = datetime.utcnow().isoformat()
    render_sidebar(settings)
    render_header()
    render_status_panels(settings)


if __name__ == "__main__":
    main()
