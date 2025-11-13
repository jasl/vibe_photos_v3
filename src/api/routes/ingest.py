"""Ingestion routes that accept uploads and trigger processing."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile

from src.api.schemas.models import ImportResponse
from src.api.services.dependencies import get_batch_processor
from src.core.processor import BatchProcessor

router = APIRouter(prefix="/import", tags=["ingest"])


@router.post("/", response_model=ImportResponse)
async def import_file(file: UploadFile, processor: BatchProcessor = Depends(get_batch_processor)):
    """Accept an uploaded file, persist it to the dataset directory, and process it."""
    dataset_dir = Path(processor.config["dataset"]["directory"]).resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    destination = dataset_dir / file.filename
    with destination.open("wb") as handle:
        content = await file.read()
        handle.write(content)

    await processor.process_file(destination)
    return ImportResponse(status="ok", data={"path": str(destination)})
