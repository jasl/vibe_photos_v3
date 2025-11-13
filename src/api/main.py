"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import assets, health, ingest, search
from src.api.services.dependencies import RuntimeResources
from src.core.database import get_session_factory, init_db
from src.core.detector import SigLIPBLIPDetector
from src.core.ocr import PaddleOCREngine
from src.core.preprocessor import ImagePreprocessor
from src.models.blip import BlipCaptioner
from src.models.siglip import SiglipClassifier
from src.utils.logging import get_logger
from src.utils.runtime import load_phase1_config


logger = get_logger(__name__)


def _bootstrap_runtime() -> RuntimeResources:
    """Instantiate shared services for the API process."""
    config = load_phase1_config()
    init_db()
    session_factory = get_session_factory()

    preprocessor = ImagePreprocessor(config["preprocessing"])

    detection_config = config.get("detection", {})
    model_name = detection_config.get("model", "google/siglip2-base-patch16-224")
    device = detection_config.get("device", "auto")

    try:
        from transformers import (
            BlipForConditionalGeneration,
            BlipProcessor,
            pipeline as hf_pipeline,
        )
    except ImportError as error:  # pragma: no cover - exercised only without transformers
        logger.error(
            "transformers is required for SigLIP/BLIP support but is not installed",
            extra={"error": str(error)},
        )
        raise

    logger.info(
        "Loading SigLIP zero-shot pipeline for API",
        extra={"model_name": model_name},
    )
    siglip_pipeline = hf_pipeline("zero-shot-image-classification", model=model_name)
    classifier = SiglipClassifier(model_name=model_name, pipeline=siglip_pipeline)

    logger.info(
        "SigLIP zero-shot pipeline loaded for API",
        extra={"model_name": model_name, "device": device},
    )

    blip_model_name = "Salesforce/blip-image-captioning-base"
    logger.info(
        "Loading BLIP captioning components for API",
        extra={"model_name": blip_model_name},
    )
    blip_processor = BlipProcessor.from_pretrained(blip_model_name)
    blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_name)
    captioner = BlipCaptioner(model_name=blip_model_name, processor=blip_processor, model=blip_model)

    logger.info(
        "BLIP captioning components loaded for API",
        extra={"model_name": blip_model_name},
    )

    detector = SigLIPBLIPDetector(
        model=model_name,
        device=device,
        classifier=classifier,
        captioner=captioner,
    )
    ocr_engine = PaddleOCREngine(config["ocr"]) if config.get("ocr", {}).get("enabled", True) else None

    return RuntimeResources(
        config=config,
        detector=detector,
        preprocessor=preprocessor,
        ocr_engine=ocr_engine,
        session_factory=session_factory,
    )


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Vibe Photos API", version="0.1.0")
    app.state.runtime = _bootstrap_runtime()

    app.include_router(health.router)
    app.include_router(assets.router)
    app.include_router(ingest.router)
    app.include_router(search.router)
    return app


app = create_app()


def run() -> None:
    """Convenience entrypoint for `uv run vibe-server`."""
    import uvicorn

    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
