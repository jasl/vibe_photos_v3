"""Reusable cache utilities for ingestion artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from src.core.database import AssetData
from src.utils.logging import get_logger


@dataclass(slots=True)
class CachedArtifact:
    """Serializable representation of an ingested asset."""

    asset: AssetData
    timings: dict[str, float]
    source_path: str
    version: str = "v1"
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )


class CacheWriter:
    """Persist detection/OCR outputs to a reusable cache directory."""

    def __init__(self, cache_dir: Path | str, *, version: str = "v1") -> None:
        self.cache_dir = Path(cache_dir)
        self.version = version
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)

    def write(self, artifact: CachedArtifact) -> Path:
        """Write a cached artifact to disk and return the written path."""
        payload = self._serialize(artifact)
        cache_name = self._build_filename(artifact)
        cache_path = self.cache_dir / cache_name
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        self.logger.debug(
            "Cached artifact", extra={"cache_path": str(cache_path), "source": artifact.source_path}
        )
        return cache_path

    def iter_cache_files(self) -> Iterable[Path]:
        """Yield every cache file stored for this writer."""
        yield from sorted(self.cache_dir.glob("*.json"))

    def _build_filename(self, artifact: CachedArtifact) -> str:
        digest = hashlib.sha256(artifact.source_path.encode("utf-8")).hexdigest()
        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        return f"{digest}_{timestamp}.json"

    def _serialize(self, artifact: CachedArtifact) -> dict[str, Any]:
        asset_dict = _serialize_asset_data(artifact.asset)
        return {
            "version": artifact.version,
            "created_at": artifact.created_at,
            "source_path": artifact.source_path,
            "timings": dict(artifact.timings),
            "asset": asset_dict,
        }


class CacheImporter:
    """Import cached artifacts back into the primary database."""

    def __init__(self, cache_dir: Path | str, session_factory, *, logger=None) -> None:
        self.cache_dir = Path(cache_dir)
        self.session_factory = session_factory
        self.logger = logger or get_logger(__name__)

    def import_all(self) -> int:
        """Import all cache files found beneath the cache directory."""
        imported = 0
        for cache_file in sorted(self.cache_dir.glob("*.json")):
            imported += self.import_file(cache_file)
        return imported

    def import_file(self, cache_file: Path) -> int:
        """Import a single cache file. Returns 1 when an asset was created."""
        if not cache_file.exists():
            return 0

        with cache_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        asset_payload = payload.get("asset")
        if not asset_payload:
            self.logger.warning("Cache file missing asset payload", extra={"path": str(cache_file)})
            return 0

        asset_data = _deserialize_asset_data(asset_payload)

        session = self.session_factory()
        try:
            from src.core.database import AssetRepository

            repository = AssetRepository(session)
            existing = repository.find_by_original_path(asset_data.original_path)
            if existing is not None:
                self.logger.debug(
                    "Skipping cached asset; already exists",
                    extra={"path": asset_data.original_path},
                )
                return 0

            repository.create_asset(asset_data)
            self.logger.info(
                "Imported cached asset", extra={"cache_file": str(cache_file), "path": asset_data.original_path}
            )
            return 1
        finally:
            session.close()


def _serialize_asset_data(asset: AssetData) -> dict[str, Any]:
    payload = asdict(asset)
    labels = payload.get("labels") or []
    payload["labels"] = [
        {"label": label, "confidence": confidence} for label, confidence in labels
    ]
    return payload


def _deserialize_asset_data(payload: dict[str, Any]) -> AssetData:
    labels_payload: Iterable[dict[str, Any]] | None = payload.get("labels")
    labels: Optional[list[tuple[str, float]]] = None
    if labels_payload:
        labels = []
        for item in labels_payload:
            label = item.get("label")
            confidence = float(item.get("confidence", 0.0))
            if label:
                labels.append((label, confidence))
    return AssetData(
        original_path=payload.get("original_path", ""),
        filename=payload.get("filename", ""),
        processed_path=payload.get("processed_path"),
        thumbnail_path=payload.get("thumbnail_path"),
        phash=payload.get("phash"),
        file_size=payload.get("file_size"),
        width=payload.get("width"),
        height=payload.get("height"),
        caption=payload.get("caption"),
        caption_source=payload.get("caption_source", "blip"),
        labels=labels,
        ocr_blocks=payload.get("ocr_blocks"),
        duplicate_count=int(payload.get("duplicate_count", 0)),
    )
