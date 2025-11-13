"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import assets, health, ingest, search
from src.api.services.dependencies import RuntimeResources
from src.core.database import get_session_factory, init_db
from src.core.detector import SigLIPBLIPDetector
from src.core.ocr import PaddleOCREngine
from src.core.preprocessor import ImagePreprocessor
from src.utils.runtime import load_phase1_config


def _bootstrap_runtime() -> RuntimeResources:
    """Instantiate shared services for the API process."""
    config = load_phase1_config()
    init_db()
    session_factory = get_session_factory()

    preprocessor = ImagePreprocessor(config["preprocessing"])
    detector = SigLIPBLIPDetector(
        model=config["detection"]["model"],
        device=config["detection"].get("device", "auto"),
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
