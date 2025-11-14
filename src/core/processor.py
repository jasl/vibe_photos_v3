"""Batch processor orchestrating preprocessing, detection, OCR, and persistence."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from sqlalchemy.orm import Session

from src.core.database import AssetData, AssetRepository
from src.core.detector import SigLIPBLIPDetector
from src.core.ocr import PaddleOCREngine
from src.core.preprocessor import ImagePreprocessor, PreprocessedImage
from src.utils.logging import get_logger
from src.models.siglip import LabelScore
from src.services.cache import CacheWriter, CachedArtifact


@dataclass(slots=True)
class _PreparedAsset:
    """Container for preprocessed assets awaiting model inference."""

    image_path: Path
    preprocessed: PreprocessedImage
    preprocess_time: float


@dataclass(slots=True)
class _DetectionOutcome:
    """Result bundle combining detection metadata and timings."""

    prepared: _PreparedAsset
    caption: str | None
    labels: List[LabelScore]
    detection_time: float


class BatchProcessor:
    """Coordinate dataset processing for Phase 1."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        detector: SigLIPBLIPDetector,
        preprocessor: ImagePreprocessor,
        config: Dict[str, Any],
        ocr_engine: PaddleOCREngine | None = None,
        *,
        cache_writer: CacheWriter | None = None,
        persist_to_db: bool = True,
    ) -> None:
        self.session_factory = session_factory
        self.detector = detector
        self.ocr_engine = ocr_engine
        self.preprocessor = preprocessor
        self.config = config
        self.logger = get_logger(__name__)
        self.cache_writer = cache_writer
        self.persist_to_db = persist_to_db
        self.total_images: int | None = None

        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "duplicates": 0,
            "failed": 0,
        }
        batching = config.get("batch_processing", {})
        self.max_workers = max(1, int(batching.get("max_workers", 1)))
        self.batch_size = max(1, int(batching.get("batch_size", self.max_workers)))
        self._stats_lock = asyncio.Lock()
        self._executor: ThreadPoolExecutor | None = None
        self._phash_index: Dict[str, Dict[str, int]] = {}

    async def process_dataset(self, incremental: bool = True) -> None:
        """Process every supported image in the dataset directory."""
        dataset_dir = Path(self.config["dataset"]["directory"]).resolve()
        dataset_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Starting dataset scan", extra={"dataset_dir": str(dataset_dir)})

        self.total_images = self._count_images(dataset_dir)

        if self.total_images == 0:
            self.logger.warning("No image files found during scan", extra={"dataset_dir": str(dataset_dir)})
            return

        await self.startup()

        batch: List[Path] = []
        try:
            for image_path in self._iter_images(dataset_dir):
                batch.append(image_path)
                if len(batch) >= self.batch_size:
                    await self._process_batch(list(batch), incremental)
                    batch.clear()

            if batch:
                await self._process_batch(list(batch), incremental)
        finally:
            await self.shutdown()
            self.total_images = None

    async def process_file(
        self, image_path: Path, *, incremental: bool = True, shutdown_executor: bool = True
    ) -> None:
        """Process a single image file (used by API uploads)."""
        await self._initialize_executor()
        await self._load_duplicate_index()
        try:
            await self._process_batch([image_path], incremental=incremental)
        finally:
            if shutdown_executor:
                await self._shutdown_executor()

    def get_statistics(self) -> Dict[str, int]:
        """Return a copy of the current stats."""
        return dict(self.stats)

    async def startup(self) -> None:
        """Prepare executor resources for long-running workloads."""
        await self._initialize_executor()
        await self._load_duplicate_index()

    async def shutdown(self) -> None:
        """Release executor resources after processing."""
        await self._shutdown_executor()

    def _iter_images(self, dataset_dir: Path) -> Iterable[Path]:
        """Yield image files matching the supported extensions."""
        supported = self.config["dataset"].get("supported_formats") or [".jpg", ".jpeg", ".png"]
        for extension in supported:
            yield from dataset_dir.glob(f"**/*{extension}")
            yield from dataset_dir.glob(f"**/*{extension.upper()}")

    async def _process_batch(self, image_paths: Sequence[Path], incremental: bool) -> None:
        prepared = await self._prepare_batch(image_paths, incremental)
        if not prepared:
            return

        detections = await self._run_detection(prepared)
        if not detections:
            return

        await self._finalize_batch(detections)

    async def _prepare_batch(self, image_paths: Sequence[Path], incremental: bool) -> List[_PreparedAsset]:
        tasks = [self._prepare_single(path, incremental) for path in image_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        prepared: List[_PreparedAsset] = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.exception("Unhandled exception while preparing asset: %s", result)
                continue
            if result is not None:
                prepared.append(result)
        return prepared

    async def _prepare_single(self, image_path: Path, incremental: bool) -> Optional[_PreparedAsset]:
        await self._increment_stat("total_processed")
        start_time = perf_counter()
        try:
            preprocessed = await self._run_in_executor(self.preprocessor.preprocess, image_path)
        except Exception as error:  # noqa: BLE001
            await self._increment_stat("failed")
            self.logger.exception(
                "Failed to preprocess %s: %s: %s",
                str(image_path),
                type(error).__name__,
                error,
                extra={"path": str(image_path), "error": str(error)},
            )
            return None

        preprocess_time = perf_counter() - start_time

        if incremental and preprocessed.phash:
            duplicate_count = await self._handle_duplicate(preprocessed.phash)
            if duplicate_count is not None:
                await self._increment_stat("duplicates")
                self.logger.info(
                    "Skipping duplicate",
                    extra={
                        "path": str(image_path),
                        "phash": preprocessed.phash,
                        "duplicate_count": duplicate_count,
                    },
                )
                return None

        return _PreparedAsset(image_path=image_path, preprocessed=preprocessed, preprocess_time=preprocess_time)

    async def _run_detection(self, prepared: Sequence[_PreparedAsset]) -> List[_DetectionOutcome]:
        batch_inputs = [
            (item.preprocessed.processed_path, item.preprocessed.processed_image)
            for item in prepared
        ]
        start_time = perf_counter()
        try:
            results = await self._run_in_executor(self.detector.analyze_batch, batch_inputs)
        except Exception as error:  # noqa: BLE001
            for item in prepared:
                await self._increment_stat("failed")
                self.logger.exception(
                    "Failed to analyze %s: %s: %s",
                    str(item.image_path),
                    type(error).__name__,
                    error,
                    extra={"path": str(item.image_path), "error": str(error)},
                )
            return []

        total_detection_time = perf_counter() - start_time
        if len(results) != len(prepared):
            self.logger.warning(
                "Detection results size mismatch",
                extra={"expected": len(prepared), "received": len(results)},
            )
        per_item_detection = total_detection_time / max(1, len(results) or len(prepared))
        outcomes: List[_DetectionOutcome] = []
        for prepared_item, detection in zip(prepared, results):
            outcomes.append(
                _DetectionOutcome(
                    prepared=prepared_item,
                    caption=detection.caption,
                    labels=detection.labels,
                    detection_time=per_item_detection,
                )
            )
        return outcomes

    async def _finalize_batch(self, outcomes: Sequence[_DetectionOutcome]) -> None:
        for outcome in outcomes:
            prepared = outcome.prepared
            detection = outcome

            ocr_blocks: List[dict] = []
            ocr_time = 0.0
            if self.ocr_engine:
                ocr_start = perf_counter()
                try:
                    ocr_result = await self._run_in_executor(
                        self.ocr_engine.extract_text,
                        prepared.preprocessed.processed_path,
                    )
                except Exception as error:  # noqa: BLE001
                    await self._increment_stat("failed")
                    self.logger.exception(
                        "OCR failed for %s: %s: %s",
                        str(prepared.image_path),
                        type(error).__name__,
                        error,
                        extra={"path": str(prepared.image_path), "error": str(error)},
                    )
                    continue

                ocr_blocks = [
                    {"text": block.text, "language": block.language, "bbox": block.bbox}
                    for block in ocr_result
                ]
                ocr_time = perf_counter() - ocr_start

            asset_data = AssetData(
                original_path=str(prepared.preprocessed.original_path),
                filename=prepared.image_path.name,
                processed_path=str(prepared.preprocessed.processed_path),
                thumbnail_path=str(prepared.preprocessed.thumbnail_path)
                if prepared.preprocessed.thumbnail_path
                else None,
                phash=prepared.preprocessed.phash,
                file_size=prepared.preprocessed.file_size,
                width=prepared.preprocessed.width,
                height=prepared.preprocessed.height,
                caption=detection.caption,
                labels=[(label.label, label.confidence) for label in detection.labels],
                ocr_blocks=ocr_blocks,
            )

            cache_error = await self._write_cache(asset_data, stage_timings, prepared.image_path)
            if cache_error:
                await self._increment_stat("failed")
                continue

            asset_id: int | None = None
            if self.persist_to_db:
                try:
                    asset_id = await self._run_in_executor(self._persist_asset, asset_data)
                except Exception as error:  # noqa: BLE001
                    await self._increment_stat("failed")
                    self.logger.exception(
                        "Failed to persist %s: %s: %s",
                        str(prepared.image_path),
                        type(error).__name__,
                        error,
                        extra={"path": str(prepared.image_path), "error": str(error)},
                    )
                    continue

            duplicate_key = prepared.preprocessed.phash
            if duplicate_key and asset_id is not None:
                self._phash_index[duplicate_key] = {
                    "asset_id": asset_id,
                    "duplicate_count": 0,
                }

            successful_index = await self._increment_stat("successful")

            stage_timings = {
                "preprocess": prepared.preprocess_time,
                "detection": detection.detection_time,
                "ocr": ocr_time,
            }

            if self.total_images:
                self.logger.info(
                    "[%d/%d] %s",
                    successful_index,
                    self.total_images,
                    prepared.image_path.name,
                    extra={"path": str(prepared.image_path), "timings": stage_timings},
                )

            self.logger.info(
                "Processed image",
                extra={
                    "path": str(prepared.image_path),
                    "caption": detection.caption or "",
                    "timings": stage_timings,
                },
            )

    async def _increment_stat(self, key: str, value: int = 1) -> int:
        """Thread-safe counter updates."""
        async with self._stats_lock:
            self.stats[key] += value
            return self.stats[key]

    async def _write_cache(
        self, asset_data: AssetData, stage_timings: Dict[str, float], image_path: Path
    ) -> bool:
        """Write cache artifacts when configured. Returns True if an error occurred."""
        if self.cache_writer is None:
            return False

        try:
            artifact = CachedArtifact(
                asset=asset_data,
                timings=stage_timings,
                source_path=str(image_path),
            )
            await self._run_in_executor(self.cache_writer.write, artifact)
            return False
        except Exception as error:  # noqa: BLE001
            self.logger.exception(
                "Failed to write cache", extra={"path": str(image_path), "error": str(error)}
            )
            return True

    def _count_images(self, dataset_dir: Path) -> int:
        return sum(1 for _ in self._iter_images(dataset_dir))

    async def _initialize_executor(self) -> None:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="processor")

    async def _shutdown_executor(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    async def _run_in_executor(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        await self._initialize_executor()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: func(*args, **kwargs))

    async def _load_duplicate_index(self) -> None:
        if self._phash_index:
            return

        def _load() -> Dict[str, Dict[str, int]]:
            session = self.session_factory()
            try:
                repository = AssetRepository(session)
                return repository.phash_index()
            finally:
                session.close()

        self._phash_index = await self._run_in_executor(_load)

    async def _handle_duplicate(self, phash: str) -> Optional[int]:
        entry = self._phash_index.get(phash)
        if entry is None:
            return None

        def _update() -> None:
            session = self.session_factory()
            try:
                repository = AssetRepository(session)
                repository.increment_duplicate_count(entry["asset_id"])
                entry["duplicate_count"] += 1
            finally:
                session.close()

        await self._run_in_executor(_update)
        return entry["duplicate_count"]

    def _persist_asset(self, asset_data: AssetData) -> None:
        """Persist asset metadata inside a dedicated session."""
        session = self.session_factory()
        try:
            repository = AssetRepository(session)
            asset = repository.create_asset(asset_data)
            return asset.id
        finally:
            session.close()
