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


class AssetLabel(BaseModel):
    """Label associated with an asset."""

    label: str
    confidence: float


class AssetCaption(BaseModel):
    """Caption text attached to an asset."""

    text: str
    source: str


class AssetOCRBlock(BaseModel):
    """OCR text block for an asset."""

    text: str
    language: Optional[str] = None
    bbox: Optional[List[List[float]]] = None


class AssetModel(BaseModel):
    """Serialized asset metadata."""

    id: int
    filename: str
    original_path: str
    processed_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    labels: List[AssetLabel] = []
    captions: List[AssetCaption] = []
    ocr: List[AssetOCRBlock] = []


class AssetResponse(ResponseEnvelope):
    """Envelope for a single asset."""

    data: AssetModel


class AssetListResponse(ResponseEnvelope):
    """Envelope for a list of assets."""

    data: List[AssetModel]


class ImportResponse(ResponseEnvelope):
    """Response returned after /import uploads."""

    data: dict


def ok_response(data: Any) -> ResponseEnvelope:
    """Return a success envelope."""
    return ResponseEnvelope(status="ok", data=data)


def error_response(message: str) -> ResponseEnvelope:
    """Return an error envelope."""
    return ResponseEnvelope(status="error", message=message)
