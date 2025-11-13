"""FastAPI dependency helpers for shared services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from fastapi import Depends, Request
from sqlalchemy.orm import sessionmaker

from src.core.detector import SigLIPBLIPDetector
from src.core.ocr import PaddleOCREngine
from src.core.preprocessor import ImagePreprocessor
from src.core.processor import BatchProcessor
from src.core.searcher import AssetSearchService


@dataclass
class RuntimeResources:
    """Shared state stored on the FastAPI app."""

    config: Dict[str, Any]
    detector: SigLIPBLIPDetector
    preprocessor: ImagePreprocessor
    ocr_engine: PaddleOCREngine | None
    session_factory: sessionmaker


def get_runtime_resources(request: Request) -> RuntimeResources:
    """Return the runtime state stored on the FastAPI app."""
    state = getattr(request.app.state, "runtime", None)
    if state is None:
        raise RuntimeError("Runtime state not initialized. Ensure create_app() is used.")
    return state


def get_search_service(resources: RuntimeResources = Depends(get_runtime_resources)) -> AssetSearchService:
    """Provide the shared search service."""
    return AssetSearchService(session_factory=resources.session_factory)


def get_batch_processor(resources: RuntimeResources = Depends(get_runtime_resources)) -> BatchProcessor:
    """Provide the shared batch processor."""
    return BatchProcessor(
        session_factory=resources.session_factory,
        detector=resources.detector,
        preprocessor=resources.preprocessor,
        ocr_engine=resources.ocr_engine,
        config=resources.config,
    )
