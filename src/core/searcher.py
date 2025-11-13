"""SQLite-backed metadata search service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.core import database


@dataclass(slots=True)
class SearchHit:
    """Search result containing asset metadata."""

    asset_id: int
    score: float
    data: dict


class AssetSearchService:
    """Perform lightweight metadata searches against SQLite."""

    def __init__(self, session=None) -> None:
        self.session = session or database.get_db_session()
        self.repository = database.AssetRepository(self.session)

    def search(self, query: str, limit: int = 20) -> List[SearchHit]:
        """Return ranked assets matching the provided query."""
        if not query:
            return []

        assets = self.repository.search_assets(query=query, limit=limit)
        hits: List[SearchHit] = []
        for asset in assets:
            matches_caption = any(query.lower() in cap.text.lower() for cap in asset.captions)
            matches_label = any(query.lower() in label.label.lower() for label in asset.labels)
            matches_ocr = any(query.lower() in block.text.lower() for block in asset.text_blocks)
            score = sum([matches_caption, matches_label, matches_ocr])
            hits.append(SearchHit(asset_id=asset.id, score=float(score), data=database.serialize_asset(asset)))

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits
