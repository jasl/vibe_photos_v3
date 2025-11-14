"""Long-running ingestion worker that consumes queued tasks."""

from __future__ import annotations

import asyncio
from pathlib import Path

from src.core.processor import BatchProcessor
from src.services.task_queue import FileTaskQueue, IngestionTask
from src.utils.logging import get_logger


class IngestionService:
    """Background worker that keeps models in memory and drains the task queue."""

    def __init__(
        self,
        processor: BatchProcessor,
        queue: FileTaskQueue,
        *,
        incremental: bool = True,
        poll_interval: float = 2.0,
    ) -> None:
        self.processor = processor
        self.queue = queue
        self.incremental = incremental
        self.poll_interval = poll_interval
        self.logger = get_logger(__name__)
        self._running = False

    async def run(self) -> None:
        """Start the worker loop until cancelled."""
        self._running = True
        await self.processor.startup()
        self.logger.info("Ingestion service started")
        try:
            while self._running:
                task = self.queue.pull()
                if task is None:
                    await asyncio.sleep(self.poll_interval)
                    continue
                await self._handle_task(task)
        except asyncio.CancelledError:  # pragma: no cover - cooperative cancellation
            raise
        finally:
            await self.processor.shutdown()
            self.logger.info("Ingestion service stopped")

    async def shutdown(self) -> None:
        """Signal the loop to stop after the current iteration."""
        self._running = False

    async def _handle_task(self, task: IngestionTask) -> None:
        try:
            await self.processor.process_file(
                task.image_path,
                incremental=self.incremental and task.incremental,
                shutdown_executor=False,
            )
        except Exception:  # noqa: BLE001
            self.logger.exception(
                "Failed to process ingestion task", extra={"path": str(task.image_path)}
            )
            self.queue.requeue(task)
        else:
            self.queue.acknowledge(task)
            self.logger.info("Processed queued asset", extra={"path": str(task.image_path)})


async def run_ingestion_service(
    processor: BatchProcessor,
    queue_dir: Path,
    *,
    incremental: bool = True,
    poll_interval: float = 2.0,
) -> None:
    """Helper to bootstrap the ingestion service from CLI scripts."""
    queue = FileTaskQueue(queue_dir)
    service = IngestionService(
        processor=processor,
        queue=queue,
        incremental=incremental,
        poll_interval=poll_interval,
    )
    await service.run()
