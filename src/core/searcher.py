"""SQLite-backed metadata search service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from sqlalchemy.orm import Session, sessionmaker

from src.core import database


@dataclass(slots=True)
class SearchHit:
    """Search result containing asset metadata."""

    asset_id: int
    score: float
    data: dict


class AssetSearchService:
    """Perform lightweight metadata searches against SQLite."""

    def __init__(self, session_factory: sessionmaker | Callable[[], Session] | None = None) -> None:
        self.session_factory: sessionmaker | Callable[[], Session] = session_factory or database.get_session_factory()

    def search(self, query: str, limit: int = 20) -> List[SearchHit]:
        """Return ranked assets matching the provided query."""
        if not query:
            return []

        hits: List[SearchHit] = []
        session = self.session_factory()
        try:
            repository = database.AssetRepository(session)
            assets = repository.search_assets(query=query, limit=limit)

            lowered = query.lower()
            for asset in assets:
                matches_caption = any(lowered in cap.text.lower() for cap in asset.captions)
                matches_label = any(lowered in label.label.lower() for label in asset.labels)
                matches_ocr = any(lowered in block.text.lower() for block in asset.text_blocks)
                score = sum([matches_caption, matches_label, matches_ocr])
                hits.append(
                    SearchHit(asset_id=asset.id, score=float(score), data=database.serialize_asset(asset))
                )
        finally:
            session.close()

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits
