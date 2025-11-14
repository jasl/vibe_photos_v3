#!/usr/bin/env python3
"""Phase 1 dataset processing script with incremental support."""

import asyncio
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.core.detector import SigLIPBLIPDetector
from src.core.ocr import PaddleOCREngine
from src.core.preprocessor import ImagePreprocessor
from src.core.processor import BatchProcessor
from src.core.database import get_session_factory, init_db
from src.models.blip import BlipCaptioner
from src.models.siglip import SiglipClassifier
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


def _build_detector(config: dict, logger: logging.Logger) -> SigLIPBLIPDetector:
    """Construct a SigLIP+BLIP detector with injected transformers components.

    This ensures that `transformers` is imported and the Hugging Face objects are
    created exactly once per process, avoiding repeated lazy imports in workers.
    """
    detection_config = config.get("detection", {})
    model_name = detection_config.get("model", "google/siglip2-base-patch16-224")
    device = detection_config.get("device", "auto")
    device_map = detection_config.get("device_map", "auto")
    precision = detection_config.get("precision", "auto")

    def _normalize_device(value: str | int | None) -> str | int | None:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        lowered = value.lower()
        if lowered in {"auto", "default"}:
            return None
        return value

    device_arg = _normalize_device(device)
    device_map_arg = None if not device_map else device_map
    if isinstance(precision, str) and precision.lower() not in {"auto", "default"}:
        torch_dtype = precision
    else:
        torch_dtype = None

    try:
        import torch  # noqa: WPS433 - optional dependency resolved lazily
    except Exception:  # noqa: BLE001 - torch may be missing in lightweight environments
        torch = None  # type: ignore[assignment]

    dtype_obj = getattr(torch, torch_dtype, None) if torch is not None and torch_dtype else None

    try:
        from transformers import (
            BlipForConditionalGeneration,
            BlipProcessor,
            pipeline as hf_pipeline,
        )
    except ImportError as error:  # pragma: no cover - exercised only without transformers
        logger.error("transformers is required for SigLIP/BLIP support but is not installed: %s", error)
        raise

    logger.info("Loading SigLIP zero-shot pipeline: %s", model_name)
    siglip_model_kwargs: dict[str, object] = {"use_safetensors": True}
    if device_map_arg:
        siglip_model_kwargs["device_map"] = device_map_arg
    if dtype_obj is not None:
        siglip_model_kwargs["torch_dtype"] = dtype_obj

    pipeline_device = None if device_map_arg else device_arg

    siglip_pipeline = hf_pipeline(
        "zero-shot-image-classification",
        model=model_name,
        device=pipeline_device,
        model_kwargs=siglip_model_kwargs,
        torch_dtype=dtype_obj,
    )
    classifier = SiglipClassifier(
        model_name=model_name,
        pipeline=siglip_pipeline,
        device=pipeline_device,
        device_map=device_map_arg,
        torch_dtype=torch_dtype,
    )

    logger.info(
        "SigLIP zero-shot pipeline loaded",
        extra={"model_name": model_name, "device": device},
    )

    blip_model_name = "Salesforce/blip-image-captioning-base"
    logger.info("Loading BLIP captioning components: %s", blip_model_name)
    blip_processor = BlipProcessor.from_pretrained(blip_model_name)
    blip_model_kwargs: dict[str, object] = {"use_safetensors": True}
    if device_map_arg:
        blip_model_kwargs["device_map"] = device_map_arg
    if dtype_obj is not None:
        blip_model_kwargs["torch_dtype"] = dtype_obj

    blip_model = BlipForConditionalGeneration.from_pretrained(
        blip_model_name,
        **blip_model_kwargs,
    )
    captioner = BlipCaptioner(
        model_name=blip_model_name,
        processor=blip_processor,
        model=blip_model,
        device_map=device_map_arg,
        torch_dtype=torch_dtype,
    )

    logger.info(
        "BLIP captioning components loaded",
        extra={"model_name": blip_model_name},
    )

    detector = SigLIPBLIPDetector(
        model=model_name,
        device=device,
        classifier=classifier,
        captioner=captioner,
    )

    if detection_config.get("warmup", True):
        classifier.ensure_loaded()

    return detector


async def main() -> int:
    """Execute the dataset processing workflow."""
    config = load_config()
    logger = setup_logging(config)
    logger.info("Configuration loaded successfully.")

    dataset_dir = Path(config["dataset"]["directory"])
    if not dataset_dir.exists():
        logger.error("Dataset directory not found: %s", dataset_dir)
        logger.info("Place sample images inside '%s' before running the script.", dataset_dir)
        return 1

    image_count = 0
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

    preprocessor = ImagePreprocessor(config["preprocessing"])
    detector = _build_detector(config, logger)
    ocr_engine = PaddleOCREngine(config["ocr"]) if config.get("ocr", {}).get("enabled", True) else None

    init_db()
    session_factory = get_session_factory()

    batch_processor = BatchProcessor(
        session_factory=session_factory,
        detector=detector,
        ocr_engine=ocr_engine,
        preprocessor=preprocessor,
        config=config,
    )

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


if __name__ == "__main__":
    Path("log").mkdir(exist_ok=True)
    sys.exit(asyncio.run(main()))
