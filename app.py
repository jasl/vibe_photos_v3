"""Streamlit entry point for the Phase 1 MVP dashboard."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import yaml

from src.core.database import AssetRepository, get_session_factory, serialize_asset
from src.core.searcher import AssetSearchService

REPO_ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"


def _select_asset(asset_id: int) -> None:
    """Store the selected asset ID in the session state."""
    st.session_state["selected_asset_id"] = int(asset_id)


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
    st.sidebar.markdown("3. Execute **uv run python process_dataset.py** before browsing results.")
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
    session_factory = get_session_factory()
    session = session_factory()
    try:
        repo = AssetRepository(session)
        total_assets = repo.total_assets()
    finally:
        session.close()
    return {"total_assets": total_assets, "refreshed_at": st.session_state.get("last_refresh", "now")}


@st.cache_data(ttl=15, show_spinner=False)
def load_gallery_assets(limit: int, offset: int) -> List[Dict[str, Any]]:
    """Return a page of assets for the gallery view."""
    session_factory = get_session_factory()
    session = session_factory()
    try:
        repo = AssetRepository(session)
        assets = repo.list_assets(limit=limit, offset=offset)
        return [serialize_asset(asset) for asset in assets]
    finally:
        session.close()


@st.cache_data(ttl=60, show_spinner=False)
def load_asset_detail(asset_id: int) -> Dict[str, Any] | None:
    """Load a single asset with full metadata."""
    session_factory = get_session_factory()
    session = session_factory()
    try:
        repo = AssetRepository(session)
        asset = repo.get_asset(asset_id)
        return serialize_asset(asset) if asset else None
    finally:
        session.close()


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
                st.image(thumbnail_path, use_container_width=True)
        with cols[1]:
            st.markdown(f"**{asset['filename']}** — score {hit.score:.2f}")
            if asset["captions"]:
                st.caption(asset["captions"][0]["text"])
            labels = ", ".join(label["label"] for label in asset["labels"])
            if labels:
                st.write("Labels:", labels)


def render_gallery_panel() -> None:
    """Render a paginated gallery of processed assets."""
    if "selected_asset_id" not in st.session_state:
        st.session_state["selected_asset_id"] = None

    stats = load_ingestion_stats()
    total_assets = stats["total_assets"]

    if total_assets == 0:
        st.info("No assets found in the database. Run the ingestion pipeline first.")
        return

    st.subheader("Gallery")

    with st.expander("Filters", expanded=False):
        text_query = st.text_input(
            "Search text",
            placeholder="Caption, labels, OCR",
            key="gallery_text_query",
        )
        label_query = st.text_input(
            "Label contains",
            placeholder="e.g. beach",
            key="gallery_label_query",
        )
        only_with_ocr = st.checkbox(
            "Only photos with OCR text",
            key="gallery_only_with_ocr",
            value=False,
        )

    page_size = 24
    max_page = max(1, (total_assets + page_size - 1) // page_size)

    page = st.number_input("Page", min_value=1, max_value=max_page, value=1, step=1)
    offset = (page - 1) * page_size

    assets = load_gallery_assets(limit=page_size, offset=offset)
    if not assets:
        st.info("No assets available for this page.")
        return

    query_lower = (text_query or "").strip().lower()
    label_lower = (label_query or "").strip().lower()

    def _matches_filters(asset: Dict[str, Any]) -> bool:
        if query_lower:
            in_caption = any(query_lower in cap["text"].lower() for cap in asset.get("captions", []))
            in_label = any(query_lower in lbl["label"].lower() for lbl in asset.get("labels", []))
            in_ocr = any(query_lower in blk["text"].lower() for blk in asset.get("ocr", []))
            if not (in_caption or in_label or in_ocr):
                return False

        if label_lower:
            if not any(label_lower in lbl["label"].lower() for lbl in asset.get("labels", [])):
                return False

        if only_with_ocr and not asset.get("ocr"):
            return False

        return True

    filtered_assets = [asset for asset in assets if _matches_filters(asset)]
    if not filtered_assets:
        st.info("No assets match the current filters on this page.")
        return

    st.caption(
        f"Showing page {page} of {max_page} "
        f"({len(filtered_assets)} assets on this page after filters).",
    )

    columns_per_row = 4
    for index in range(0, len(filtered_assets), columns_per_row):
        row_assets = filtered_assets[index : index + columns_per_row]
        cols = st.columns(len(row_assets))
        for col, asset in zip(cols, row_assets):
            with col:
                thumbnail_path = asset.get("thumbnail_path") or asset.get("processed_path")
                if thumbnail_path and Path(thumbnail_path).exists():
                    st.image(thumbnail_path, use_container_width=True)
                st.caption(asset.get("filename", ""))
                st.button(
                    "View details",
                    key=f"view-{asset['id']}",
                    on_click=_select_asset,
                    args=(int(asset["id"]),),
                )

    render_asset_detail_panel()


def render_asset_detail_panel() -> None:
    """Render a detail view for the selected asset."""
    asset_id = st.session_state.get("selected_asset_id")
    if not asset_id:
        return

    asset = load_asset_detail(int(asset_id))
    if asset is None:
        st.info("Selected asset could not be found. It may have been removed.")
        return

    st.markdown("---")
    st.subheader("Photo details")

    col_image, col_meta = st.columns([2, 3])
    with col_image:
        image_path = asset.get("processed_path") or asset.get("thumbnail_path")
        if image_path and Path(image_path).exists():
            st.image(image_path, use_container_width=True)
        else:
            st.warning("Image file not found on disk.")

    with col_meta:
        st.markdown(f"**ID:** {asset.get('id')}")
        st.markdown(f"**Filename:** {asset.get('filename', '')}")
        st.markdown(f"**Original path:** `{asset.get('original_path', '')}`")
        st.markdown(f"**Processed path:** `{asset.get('processed_path') or ''}`")
        st.markdown(f"**Thumbnail path:** `{asset.get('thumbnail_path') or ''}`")

        if asset.get("captions"):
            st.markdown("**Captions**")
            for caption in asset["captions"]:
                source = caption.get("source", "unknown")
                st.write(f"- {caption['text']} ({source})")

        if asset.get("labels"):
            st.markdown("**Labels**")
            for label in asset["labels"]:
                st.write(f"- {label['label']} ({label['confidence']:.2f})")

        if asset.get("ocr"):
            st.markdown("**OCR text blocks**")
            for block in asset["ocr"]:
                language = block.get("language") or "unknown"
                st.write(f"- [{language}] {block['text']}")

    if st.button("Clear selection", key="clear-selected-asset"):
        st.session_state["selected_asset_id"] = None


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
    view = st.sidebar.radio("View", ["Overview", "Gallery"], index=0)
    if view == "Overview":
        render_status_panels(settings)
    else:
        render_gallery_panel()


if __name__ == "__main__":
    main()
