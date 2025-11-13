"""Typer-based CLI for Phase 1 workflows."""

from __future__ import annotations

import asyncio

import typer

from src.core.database import get_session_factory, init_db
from src.core.searcher import AssetSearchService
from src.utils.runtime import load_phase1_config

app = typer.Typer(help="Vibe Photos Phase 1 command-line interface.")


@app.command("ingest")
def ingest_command() -> None:
    """Run batch dataset ingestion."""
    from process_dataset import main as process_dataset_main

    exit_code = asyncio.run(process_dataset_main())
    raise typer.Exit(code=exit_code)


@app.command("search")
def search_command(query: str, limit: int = typer.Option(10, "--limit", "-n", min=1, max=100)) -> None:
    """Run a metadata search against the local SQLite store."""
    load_phase1_config()
    init_db()
    session_factory = get_session_factory()
    service = AssetSearchService(session_factory=session_factory)
    hits = service.search(query=query, limit=limit)

    if not hits:
        typer.echo("No matches found.")
        raise typer.Exit(code=0)

    for hit in hits:
        filename = hit.data.get("filename", "")
        typer.echo(f"{hit.asset_id}\t{hit.score:.2f}\t{filename}")


@app.command("rebuild-index")
def rebuild_index_command() -> None:
    """Placeholder for future embedding index rebuild."""
    typer.echo("Rebuild index is not implemented for the Phase 1 PoC. Search uses metadata only.")
