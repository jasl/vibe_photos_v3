"""Typer CLI exposing ingestion and search helpers."""

from __future__ import annotations

import asyncio

import typer

TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise typer.BadParameter("Expected a boolean value such as 'true', 'false', 'yes', or 'no'.")

from src.core.database import get_session_factory, init_db
from src.core.detector import SigLIPBLIPDetector
from src.core.ocr import PaddleOCREngine
from src.core.preprocessor import ImagePreprocessor
from src.core.processor import BatchProcessor
from src.core.searcher import AssetSearchService
from src.utils.runtime import load_phase1_config

app = typer.Typer(help="Vibe Photos command-line interface")
cli = app


def _build_processor(config: dict) -> BatchProcessor:
    init_db()
    session_factory = get_session_factory()
    preprocessor = ImagePreprocessor(config["preprocessing"])
    detector = SigLIPBLIPDetector(
        model=config["detection"]["model"],
        device=config["detection"].get("device", "auto"),
    )
    ocr_engine = PaddleOCREngine(config["ocr"]) if config.get("ocr", {}).get("enabled", True) else None

    return BatchProcessor(
        session_factory=session_factory,
        detector=detector,
        preprocessor=preprocessor,
        ocr_engine=ocr_engine,
        config=config,
    )


@app.command()
def ingest(
    incremental: bool = typer.Option(
        True,
        "--incremental",
        parser=_parse_bool,
        help="Skip files already processed via perceptual hash (pass '--incremental false' to reprocess everything).",
        show_default=True,
    ),
) -> None:
    """Process the dataset defined in config/settings.yaml."""
    config = load_phase1_config()
    processor = _build_processor(config)

    typer.echo("Starting ingestion…")
    asyncio.run(processor.process_dataset(incremental=incremental))
    stats = processor.get_statistics()
    typer.echo(f"Processed: {stats['successful']} success, {stats['duplicates']} duplicates, {stats['failed']} failed.")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, help="Maximum results"),
) -> None:
    """Execute a metadata search and print the results."""
    init_db()
    service = AssetSearchService()
    hits = service.search(query=query, limit=limit)

    if not hits:
        typer.echo("No matches found.")
        raise typer.Exit(code=0)

    for hit in hits:
        asset = hit.data
        typer.echo(
            f"[{hit.score:.2f}] #{asset['id']} {asset['filename']} – "
            f"caption: {asset['captions'][0]['text'] if asset['captions'] else 'N/A'}"
        )


@app.command(name="rebuild-index")
def rebuild_index() -> None:
    """Placeholder command that recreates database tables."""
    typer.echo("Rebuilding SQLite schema…")
    init_db()
    typer.echo("Done.")
