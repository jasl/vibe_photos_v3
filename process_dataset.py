#!/usr/bin/env python3
"""Phase 1 dataset processing script with incremental support."""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import ModuleType

from src.core.detector import build_detector
from src.core.ocr import PaddleOCREngine
from src.core.preprocessor import ImagePreprocessor
from src.core.processor import BatchProcessor
from src.core.database import get_session_factory, init_db
from src.services.cache import CacheImporter, CacheWriter
from src.services.ingestion_service import run_ingestion_service
from src.services.task_queue import FileTaskQueue
from src.utils.runtime import load_phase1_config


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def setup_logging(config: dict) -> logging.Logger:
    """Configure console and rotating file logging."""
    log_config = config.get("logging", {})

    log_dir = Path(log_config.get("directory", "log"))
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / "process_dataset.log"
    log_format = log_config.get("format", "[%(asctime)s] %(levelname)s - %(message)s")
    log_level = log_config.get("level", "INFO")

    rotation_config = log_config.get("rotation", {})
    max_bytes = rotation_config.get("max_bytes", 10_485_760)
    backup_count = rotation_config.get("backup_count", 5)

    handlers = [
        logging.StreamHandler(),
        RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count),
    ]

    logging.basicConfig(level=getattr(logging, log_level), format=log_format, handlers=handlers)
    return logging.getLogger(__name__)


def load_config() -> dict:
    """Load merged blueprint + runtime configuration."""
    return load_phase1_config()


async def main(
    *,
    dry_run: bool = False,
    stub_models: bool = False,
    service: bool = False,
    import_cache: bool = False,
    enqueue_only: bool = False,
    skip_db: bool = False,
) -> int:
    """Execute the dataset processing workflow."""
    config = load_config()
    logger = setup_logging(config)
    logger.info("Configuration loaded successfully.")

    dataset_dir = Path(config["dataset"]["directory"])
    image_count = 0
    if dry_run:
        logger.info("Dry-run mode enabled; skipping dataset validation and processing.")
    else:
        if not dataset_dir.exists():
            logger.error("Dataset directory not found: %s", dataset_dir)
            logger.info("Place sample images inside '%s' before running the script.", dataset_dir)
            return 1

        for fmt in config["dataset"]["supported_formats"]:
            image_count += len(list(dataset_dir.glob(f"**/*{fmt}")))
            image_count += len(list(dataset_dir.glob(f"**/*{fmt.upper()}")))

        logger.info("Dataset directory: %s", dataset_dir)
        logger.info("Detected %d image files.", image_count)

        if image_count == 0:
            logger.warning("No image files detected. Aborting run.")
            return 1

    for path_value in config["preprocessing"]["paths"].values():
        resolved_path = Path(path_value)
        target = resolved_path if resolved_path.suffix == "" else resolved_path.parent
        target.mkdir(parents=True, exist_ok=True)

    queue_dir = Path(config["preprocessing"]["paths"].get("ingestion_queue", "cache/ingestion_queue"))
    queue_dir.mkdir(parents=True, exist_ok=True)

    if import_cache:
        logger.info("Importing cached detections into the database.")
        init_db()
        session_factory = get_session_factory()
        importer = CacheImporter(
            cache_dir=Path(config["preprocessing"]["paths"]["detections"]),
            session_factory=session_factory,
            logger=logger,
        )
        imported = importer.import_all()
        logger.info("Imported %d cached artifacts.", imported)
        return 0

    if dry_run and stub_models:
        _install_dry_run_stubs(logger)

    preprocessor = ImagePreprocessor(config["preprocessing"])
    detector = build_detector(config, logger=logger)
    ocr_engine = PaddleOCREngine(config["ocr"]) if config.get("ocr", {}).get("enabled", True) else None

    init_db()
    session_factory = get_session_factory()

    if dry_run:
        logger.info("Running initialization warmups.")
        if ocr_engine is not None and ocr_engine.enabled:
            ocr_engine.warmup()
            logger.info("OCR engine warmup completed.")
        logger.info("Dry-run checks completed successfully.")
        return 0

    cache_writer = CacheWriter(Path(config["preprocessing"]["paths"]["detections"]))
    persist_to_db = bool(config["dataset"].get("persist_to_db", True)) and not skip_db

    batch_processor = BatchProcessor(
        session_factory=session_factory,
        detector=detector,
        ocr_engine=ocr_engine,
        preprocessor=preprocessor,
        config=config,
        cache_writer=cache_writer,
        persist_to_db=persist_to_db,
    )

    if service:
        logger.info("Starting ingestion service; press Ctrl+C to stop.")
        try:
            await run_ingestion_service(
                processor=batch_processor,
                queue_dir=queue_dir,
                incremental=config["dataset"]["incremental"],
            )
        except KeyboardInterrupt:
            logger.info("Ingestion service interrupted by user.")
        return 0

    if enqueue_only:
        queue = FileTaskQueue(queue_dir)
        enqueued = _enqueue_dataset(queue, dataset_dir, config["dataset"].get("supported_formats", []))
        logger.info("Enqueued %d images for the ingestion service.", enqueued)
        return 0

    start_time = datetime.now()
    logger.info("Starting dataset processing...")

    try:
        await batch_processor.process_dataset(incremental=config["dataset"]["incremental"])

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("Processing completed in %.1f seconds.", elapsed)

        stats = batch_processor.get_statistics()
        logger.info("Processing summary:")
        logger.info("  - Total processed: %s", stats["total_processed"])
        logger.info("  - Successful: %s", stats["successful"])
        logger.info("  - Skipped (duplicates): %s", stats["duplicates"])
        logger.info("  - Failed: %s", stats["failed"])

        if stats["failed"] > 0:
            logger.warning("%d images failed to process. Check logs for details.", stats["failed"])

    except Exception as error:  # noqa: BLE001
        logger.error("Error while processing dataset: %s", error, exc_info=True)
        return 1

    logger.info("✅ Dataset processing finished.")
    logger.info("Next steps:")
    logger.info("  1. Start the API: uv run uvicorn src.api.main:app --reload")
    logger.info("  2. Start the UI: streamlit run app.py")
    logger.info("  3. Open http://localhost:8501 to explore the gallery")

    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Process the Vibe Photos dataset with Phase 1 pipeline.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip dataset scanning and run environment/model warmups only.",
    )
    parser.add_argument(
        "--real-models",
        action="store_true",
        help="Use actual model weights during dry-run instead of light stubs.",
    )
    parser.add_argument(
        "--service",
        action="store_true",
        help="Start the long-lived ingestion service and consume queued tasks.",
    )
    parser.add_argument(
        "--import-cache",
        action="store_true",
        help="Import cached detections into the database and exit.",
    )
    parser.add_argument(
        "--enqueue-only",
        action="store_true",
        help="Queue dataset images for the ingestion service without processing them immediately.",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip database writes during processing (cache only).",
    )
    return parser


def _enqueue_dataset(queue: FileTaskQueue, dataset_dir: Path, supported_formats: list[str]) -> int:
    enqueued = 0
    formats = supported_formats or [".jpg", ".jpeg", ".png"]
    for extension in formats:
        for image_path in dataset_dir.glob(f"**/*{extension}"):
            queue.enqueue(image_path)
            enqueued += 1
        for image_path in dataset_dir.glob(f"**/*{extension.upper()}"):
            queue.enqueue(image_path)
            enqueued += 1
    return enqueued


def _install_dry_run_stubs(logger: logging.Logger) -> None:
    """Install lightweight stubs to keep dry-run fast."""

    logger.info("Dry-run stub mode enabled; installing lightweight model stubs.")

    class DummyPipeline:
        def __call__(self, *args, **kwargs):
            return []

    def fake_pipeline(task, model, device=None, model_kwargs=None, torch_dtype=None):  # noqa: ARG001
        return DummyPipeline()

    class DummyProcessor:
        @classmethod
        def from_pretrained(cls, model_name, use_fast=True):  # noqa: ARG002
            instance = cls()
            instance.model_name = model_name
            return instance

        def __call__(self, images, return_tensors="pt"):
            return {"pixel_values": images}

        def decode(self, _output_ids, skip_special_tokens=True):  # noqa: ARG002
            return "stub-caption"

    class DummyModel:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            instance = cls()
            instance.model_name = model_name
            instance.kwargs = kwargs
            return instance

        def eval(self):
            return self

        def generate(self, **_kwargs):
            return [[0]]

    transformers_stub = ModuleType("transformers")
    transformers_stub.pipeline = fake_pipeline
    transformers_stub.BlipProcessor = DummyProcessor
    transformers_stub.BlipForConditionalGeneration = DummyModel
    previous_transformers = sys.modules.get("transformers")
    if previous_transformers is not None:
        logger.debug("Overriding previously imported transformers module for dry-run.")
    sys.modules["transformers"] = transformers_stub

    class DummyPaddleOCR:
        def __init__(self, *args, **kwargs):  # noqa: D401, ARG002
            """Initialize a stub PaddleOCR engine."""

        def ocr(self, *_args, **_kwargs):
            return []

    paddle_stub = ModuleType("paddleocr")
    paddle_stub.PaddleOCR = DummyPaddleOCR
    previous_paddle = sys.modules.get("paddleocr")
    if previous_paddle is not None:
        logger.debug("Overriding previously imported paddleocr module for dry-run.")
    sys.modules["paddleocr"] = paddle_stub


if __name__ == "__main__":
    Path("log").mkdir(exist_ok=True)
    arguments = _build_arg_parser().parse_args()
    stub_models = arguments.dry_run and not arguments.real_models
    sys.exit(
        asyncio.run(
            main(
                dry_run=arguments.dry_run,
                stub_models=stub_models,
                service=arguments.service,
                import_cache=arguments.import_cache,
                enqueue_only=arguments.enqueue_only,
                skip_db=arguments.skip_db,
            )
        )
    )
