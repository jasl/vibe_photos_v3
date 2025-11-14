"""Filesystem-backed ingestion task queue."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from uuid import uuid4

from src.utils.logging import get_logger


@dataclass(slots=True)
class IngestionTask:
    """A single ingestion task pulled from the filesystem queue."""

    image_path: Path
    incremental: bool
    payload_path: Path


class FileTaskQueue:
    """Minimal filesystem queue so workers survive restarts."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.pending_dir = self.root / "pending"
        self.inflight_dir = self.root / "inflight"
        self.failed_dir = self.root / "failed"
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.inflight_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)

    def enqueue(self, image_path: Path, *, incremental: bool = True) -> Path:
        """Write a task file describing the image to be processed."""
        payload = {
            "image_path": str(image_path),
            "incremental": bool(incremental),
            "created_at": time.time(),
        }
        task_name = f"task-{int(time.time() * 1000)}-{uuid4().hex}.json"
        task_path = self.pending_dir / task_name
        with task_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        self.logger.debug("Enqueued ingestion task", extra={"task": str(task_path)})
        return task_path

    def pull(self) -> Optional[IngestionTask]:
        """Fetch the next available ingestion task (if any)."""
        candidates = sorted(self.pending_dir.glob("*.json"))
        if not candidates:
            return None
        task_path = candidates[0]
        inflight_path = self.inflight_dir / task_path.name
        task_path.rename(inflight_path)

        with inflight_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        image_path = Path(payload["image_path"])
        incremental = bool(payload.get("incremental", True))
        return IngestionTask(image_path=image_path, incremental=incremental, payload_path=inflight_path)

    def acknowledge(self, task: IngestionTask) -> None:
        """Remove a processed task from the queue."""
        if task.payload_path.exists():
            task.payload_path.unlink()
        self.logger.debug("Acknowledged ingestion task", extra={"task": str(task.payload_path)})

    def requeue(self, task: IngestionTask, *, retry: bool = True) -> None:
        """Move a failed task back to pending or into the failed bucket."""
        destination = self.pending_dir if retry else self.failed_dir
        destination_path = destination / task.payload_path.name
        if task.payload_path.exists():
            task.payload_path.rename(destination_path)
        self.logger.warning(
            "Requeued ingestion task",
            extra={"task": str(destination_path), "retry": retry},
        )
