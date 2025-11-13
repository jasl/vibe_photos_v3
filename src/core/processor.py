"""Batch processor orchestrating preprocessing, detection, OCR, and persistence."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.core.database import AssetData, AssetRepository
from src.core.detector import SigLIPBLIPDetector
from src.core.ocr import PaddleOCREngine
from src.core.preprocessor import ImagePreprocessor
from src.utils.logging import get_logger


class BatchProcessor:
    """Coordinate dataset processing for Phase 1."""

    def __init__(
        self,
        db_session,
        detector: SigLIPBLIPDetector,
        preprocessor: ImagePreprocessor,
        config: Dict[str, Any],
        ocr_engine: PaddleOCREngine | None = None,
    ) -> None:
        self.session = db_session
        self.detector = detector
        self.ocr_engine = ocr_engine
        self.preprocessor = preprocessor
        self.config = config
        self.repository = AssetRepository(db_session)
        self.logger = get_logger(__name__)

        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "duplicates": 0,
            "failed": 0,
        }

    async def process_dataset(self, incremental: bool = True) -> None:
        """Process every supported image in the dataset directory."""
        dataset_dir = Path(self.config["dataset"]["directory"]).resolve()
        dataset_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Starting dataset scan", extra={"dataset_dir": str(dataset_dir)})
        for image_path in self._iter_images(dataset_dir):
            await self._process_path(image_path, incremental=incremental)

    async def process_file(self, image_path: Path, incremental: bool = True) -> None:
        """Process a single image file (used by API uploads)."""
        await self._process_path(image_path, incremental=incremental)

    def get_statistics(self) -> Dict[str, int]:
        """Return a copy of the current stats."""
        return dict(self.stats)

    def _iter_images(self, dataset_dir: Path) -> Iterable[Path]:
        """Yield image files matching the supported extensions."""
        supported = self.config["dataset"].get("supported_formats") or [".jpg", ".jpeg", ".png"]
        for extension in supported:
            yield from dataset_dir.glob(f"**/*{extension}")
            yield from dataset_dir.glob(f"**/*{extension.upper()}")

    async def _process_path(self, image_path: Path, incremental: bool) -> None:
        """Process a single file with preprocessing + detector."""
        self.stats["total_processed"] += 1

        try:
            result = await asyncio.to_thread(self.preprocessor.preprocess, image_path)
            if incremental and result.phash and self.repository.find_by_phash(result.phash):
                self.logger.info(
                    "Skipping duplicate",
                    extra={"path": str(image_path), "phash": result.phash},
                )
                self.stats["duplicates"] += 1
                return

            detection = await asyncio.to_thread(self.detector.analyze, result.processed_path)

            ocr_blocks = []
            if self.ocr_engine:
                ocr_result = await asyncio.to_thread(self.ocr_engine.extract_text, result.processed_path)
                ocr_blocks = [
                    {"text": block.text, "language": block.language, "bbox": block.bbox}
                    for block in ocr_result
                ]

            asset_data = AssetData(
                original_path=str(result.original_path),
                filename=image_path.name,
                processed_path=str(result.processed_path),
                thumbnail_path=str(result.thumbnail_path) if result.thumbnail_path else None,
                phash=result.phash,
                file_size=result.file_size,
                width=result.width,
                height=result.height,
                caption=detection.caption,
                labels=[(label.label, label.confidence) for label in detection.labels],
                ocr_blocks=ocr_blocks,
            )

            await asyncio.to_thread(self.repository.create_asset, asset_data)
            self.stats["successful"] += 1
            self.logger.info(
                "Processed image",
                extra={"path": str(image_path), "caption": detection.caption or ""},
            )
        except Exception as error:  # noqa: BLE001
            self.stats["failed"] += 1
            self.logger.error(
                "Failed to process image",
                extra={"path": str(image_path), "error": str(error)},
            )
