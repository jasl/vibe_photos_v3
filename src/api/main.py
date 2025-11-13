"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import health, ingest, search
from src.api.services.dependencies import RuntimeState
from src.core.database import get_db_session, init_db
from src.core.detector import SigLIPBLIPDetector
from src.core.ocr import PaddleOCREngine
from src.core.preprocessor import ImagePreprocessor
from src.core.processor import BatchProcessor
from src.core.searcher import AssetSearchService
from src.utils.runtime import load_phase1_config


def _bootstrap_runtime() -> RuntimeState:
    """Instantiate shared services for the API process."""
    config = load_phase1_config()
    init_db()
    session = get_db_session()

    preprocessor = ImagePreprocessor(config["preprocessing"])
    detector = SigLIPBLIPDetector(
        model=config["detection"]["model"],
        device=config["detection"].get("device", "auto"),
    )
    ocr_engine = PaddleOCREngine(config["ocr"]) if config.get("ocr", {}).get("enabled", True) else None

    processor = BatchProcessor(
        db_session=session,
        detector=detector,
        preprocessor=preprocessor,
        ocr_engine=ocr_engine,
        config=config,
    )

    search_service = AssetSearchService()
    return RuntimeState(processor=processor, search_service=search_service)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Vibe Photos API", version="0.1.0")
    app.state.runtime = _bootstrap_runtime()

    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(search.router)
    return app


app = create_app()


def run() -> None:
    """Convenience entrypoint for `uv run vibe-server`."""
    import uvicorn

    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
