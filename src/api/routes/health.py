"""Health probe endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas.models import ok_response

router = APIRouter(tags=["health"])


@router.get("/health")
async def healthcheck():
    """Simple health endpoint for readiness probes."""
    return ok_response({"service": "vibe-photos", "status": "healthy"})
