"""Asset metadata routes backed by SQLite."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from src.api.schemas.models import AssetListResponse, AssetModel, AssetResponse
from src.api.services.dependencies import get_db_session
from src.core.database import AssetRepository, serialize_asset

router = APIRouter(prefix="/assets", tags=["assets"])


@router.get("/", response_model=AssetListResponse)
def list_assets(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_db_session),
):
    """Return a page of recently ingested assets."""
    repository = AssetRepository(session)
    assets = repository.list_assets(limit=limit, offset=offset)
    payload = [AssetModel(**serialize_asset(asset)) for asset in assets]
    return AssetListResponse(status="ok", data=payload)


@router.get("/{asset_id}", response_model=AssetResponse)
def get_asset(asset_id: int, session: Session = Depends(get_db_session)):
    """Return metadata for a single asset."""
    repository = AssetRepository(session)
    asset = repository.get_asset(asset_id)
    if asset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")

    return AssetResponse(status="ok", data=AssetModel(**serialize_asset(asset)))

