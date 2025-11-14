"""Ingestion routes that accept uploads and trigger processing."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from src.api.schemas.models import ImportResponse
from src.api.services.dependencies import get_batch_processor
from src.core.processor import BatchProcessor

router = APIRouter(prefix="/import", tags=["ingest"])


@router.post("/", response_model=ImportResponse)
async def import_file(file: UploadFile, processor: BatchProcessor = Depends(get_batch_processor)):
    """Accept an uploaded file, persist it to the dataset directory, and process it."""
    dataset_dir = Path(processor.config["dataset"]["directory"]).resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    destination = _derive_destination_path(dataset_dir, file.filename)
    with destination.open("wb") as handle:
        content = await file.read()
        handle.write(content)

    await processor.process_file(destination)
    return ImportResponse(status="ok", data={"path": str(destination)})


def _derive_destination_path(dataset_dir: Path, original_filename: str | None) -> Path:
    """Resolve a safe destination path inside the dataset directory for an upload."""
    sanitized_name = _sanitize_filename(original_filename)
    destination = (dataset_dir / sanitized_name).resolve()
    dataset_dir_resolved = dataset_dir.resolve()
    if dataset_dir_resolved not in destination.parents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid upload filename.",
        )
    return destination


def _sanitize_filename(original_filename: str | None) -> str:
    """Normalize the uploaded filename and guard against traversal characters."""
    raw_name = (original_filename or "").strip()
    candidate = Path(raw_name).name
    suffix = Path(raw_name).suffix
    if not candidate or candidate in {".", ".."}:
        candidate = f"upload-{uuid4().hex}{suffix}"

    invalid_separators = {"/", "\\"}
    if any(sep in candidate for sep in invalid_separators):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename contains invalid path separators.",
        )

    return candidate
