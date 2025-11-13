"""FastAPI dependency helpers for shared services."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Request

from src.core.processor import BatchProcessor
from src.core.searcher import AssetSearchService


@dataclass
class RuntimeState:
    """Shared state stored on the FastAPI app."""

    processor: BatchProcessor
    search_service: AssetSearchService


def get_runtime_state(request: Request) -> RuntimeState:
    """Return the runtime state stored on the FastAPI app."""
    state = getattr(request.app.state, "runtime", None)
    if state is None:
        raise RuntimeError("Runtime state not initialized. Ensure create_app() is used.")
    return state


def get_search_service(state: RuntimeState = Depends(get_runtime_state)) -> AssetSearchService:
    """Provide the shared search service."""
    return state.search_service


def get_batch_processor(state: RuntimeState = Depends(get_runtime_state)) -> BatchProcessor:
    """Provide the shared batch processor."""
    return state.processor
