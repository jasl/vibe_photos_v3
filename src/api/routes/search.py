"""Search API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.api.schemas.models import SearchHitModel, SearchResponse
from src.api.services.dependencies import get_search_service
from src.core.searcher import AssetSearchService

router = APIRouter(prefix="/search", tags=["search"])


@router.get("/", response_model=SearchResponse)
async def search_assets(
    query: str = Query(..., description="Natural language query"),
    limit: int = Query(20, ge=1, le=100),
    service: AssetSearchService = Depends(get_search_service),
):
    """Return ranked assets that match the query."""
    hits = service.search(query=query, limit=limit)
    payload = [SearchHitModel(asset_id=hit.asset_id, score=hit.score, data=hit.data) for hit in hits]
    return SearchResponse(status="ok", data=payload)
