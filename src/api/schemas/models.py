"""Pydantic models shared across FastAPI routes."""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


class ResponseEnvelope(BaseModel):
    """Standard response structure with status and optional payload."""

    status: Literal["ok", "error"]
    data: Any | None = None
    message: Optional[str] = None


class SearchHitModel(BaseModel):
    """Serializer for search hits."""

    asset_id: int
    score: float = Field(description="Heuristic score derived from metadata matches")
    data: dict


class SearchResponse(ResponseEnvelope):
    """Search results envelope."""

    data: List[SearchHitModel]


class ImportResponse(ResponseEnvelope):
    """Response returned after /import uploads."""

    data: dict


def ok_response(data: Any) -> ResponseEnvelope:
    """Return a success envelope."""
    return ResponseEnvelope(status="ok", data=data)


def error_response(message: str) -> ResponseEnvelope:
    """Return an error envelope."""
    return ResponseEnvelope(status="error", message=message)
