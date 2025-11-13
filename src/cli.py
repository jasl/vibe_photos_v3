"""Typer CLI exposing ingestion and search helpers."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from src.core.database import get_db_session, init_db
from src.core.detector import SigLIPBLIPDetector
from src.core.ocr import PaddleOCREngine
from src.core.preprocessor import ImagePreprocessor
from src.core.processor import BatchProcessor
from src.core.searcher import AssetSearchService
from src.utils.runtime import load_phase1_config

cli = typer.Typer(help="Vibe Photos command-line interface")


def _build_processor(config: dict) -> BatchProcessor:
    init_db()
    session = get_db_session()
    preprocessor = ImagePreprocessor(config["preprocessing"])
    detector = SigLIPBLIPDetector(
        model=config["detection"]["model"],
        device=config["detection"].get("device", "auto"),
    )
    ocr_engine = PaddleOCREngine(config["ocr"]) if config.get("ocr", {}).get("enabled", True) else None

    return BatchProcessor(
        db_session=session,
        detector=detector,
        preprocessor=preprocessor,
        ocr_engine=ocr_engine,
        config=config,
    )


@cli.command()
def ingest(incremental: bool = typer.Option(True, help="Skip files already processed via perceptual hash.")) -> None:
    """Process the dataset defined in config/settings.yaml."""
    config = load_phase1_config()
    processor = _build_processor(config)

    typer.echo("Starting ingestion…")
    asyncio.run(processor.process_dataset(incremental=incremental))
    stats = processor.get_statistics()
    typer.echo(f"Processed: {stats['successful']} success, {stats['duplicates']} duplicates, {stats['failed']} failed.")


@cli.command()
def search(query: str = typer.Argument(..., help="Search query"), limit: int = typer.Option(10, help="Maximum results")) -> None:
    """Execute a metadata search and print the results."""
    init_db()
    service = AssetSearchService()
    hits = service.search(query=query, limit=limit)

    if not hits:
        typer.echo("No matches found.")
        raise typer.Exit(code=0)

    for hit in hits:
        asset = hit.data
        typer.echo(f"[{hit.score:.2f}] #{asset['id']} {asset['filename']} – caption: {asset['captions'][0]['text'] if asset['captions'] else 'N/A'}")


@cli.command(name="rebuild-index")
def rebuild_index() -> None:
    """Placeholder command that recreates database tables."""
    typer.echo("Rebuilding SQLite schema…")
    init_db()
    typer.echo("Done.")


def app():
    """Entry hook for `uv run vibe`."""
    cli()
