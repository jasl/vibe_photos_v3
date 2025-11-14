"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import assets, health, ingest, search
from src.api.services.dependencies import RuntimeResources
from src.core.database import get_session_factory, init_db
from src.core.detector import build_detector
from src.core.ocr import PaddleOCREngine
from src.core.preprocessor import ImagePreprocessor
from src.utils.logging import get_logger
from src.utils.runtime import load_phase1_config


logger = get_logger(__name__)


def _bootstrap_runtime() -> RuntimeResources:
    """Instantiate shared services for the API process."""
    config = load_phase1_config()
    init_db()
    session_factory = get_session_factory()

    preprocessor = ImagePreprocessor(config["preprocessing"])

    detector = build_detector(config, logger=logger)
    ocr_engine = PaddleOCREngine(config["ocr"]) if config.get("ocr", {}).get("enabled", True) else None

    return RuntimeResources(
        config=config,
        detector=detector,
        preprocessor=preprocessor,
        ocr_engine=ocr_engine,
        session_factory=session_factory,
    )


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Initialize shared runtime resources on startup."""
    app.state.runtime = _bootstrap_runtime()
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Vibe Photos API", version="0.1.0", lifespan=_lifespan)

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
