"""SQLite persistence layer for the Phase 1 prototype."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.sql import func

from src.utils.config import Settings, load_settings

ENGINE: Engine | None = None
SessionFactory: sessionmaker | None = None


class Base(DeclarativeBase):
    """Shared declarative base."""


class Asset(Base):
    """Primary asset record representing an ingested image."""

    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    original_path: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    processed_path: Mapped[Optional[str]] = mapped_column(String(1024))
    thumbnail_path: Mapped[Optional[str]] = mapped_column(String(1024))
    phash: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    file_size: Mapped[Optional[int]] = mapped_column(Integer)
    width: Mapped[Optional[int]] = mapped_column(Integer)
    height: Mapped[Optional[int]] = mapped_column(Integer)
    captured_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(32), default="processed")
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    embedding_json: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    labels: Mapped[List["Label"]] = relationship(back_populates="asset", cascade="all, delete-orphan")
    captions: Mapped[List["Caption"]] = relationship(back_populates="asset", cascade="all, delete-orphan")
    text_blocks: Mapped[List["TextBlock"]] = relationship(back_populates="asset", cascade="all, delete-orphan")


class Label(Base):
    """Classification labels linked to an asset."""

    __tablename__ = "labels"

    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), nullable=False, index=True)
    label: Mapped[str] = mapped_column(String(256), nullable=False)
    confidence: Mapped[float] = mapped_column(default=0.0)

    asset: Mapped[Asset] = relationship(back_populates="labels")


class Caption(Base):
    """Caption text derived from BLIP or manual annotations."""

    __tablename__ = "captions"

    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), nullable=False, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(64), default="blip")

    asset: Mapped[Asset] = relationship(back_populates="captions")


class TextBlock(Base):
    """OCR text blocks for an asset."""

    __tablename__ = "text_blocks"

    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), nullable=False, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[Optional[str]] = mapped_column(String(16))
    bbox: Mapped[Optional[str]] = mapped_column(Text)

    asset: Mapped[Asset] = relationship(back_populates="text_blocks")


class JobRun(Base):
    """Records ingestion jobs for observability."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_type: Mapped[str] = mapped_column(String(64), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(32), default="running")
    metadata_json: Mapped[Optional[str]] = mapped_column(Text)


def _ensure_engine(settings: Settings | None = None) -> Engine:
    """Create (or reuse) the SQLAlchemy engine."""
    global ENGINE  # noqa: PLW0603
    global SessionFactory  # noqa: PLW0603

    if ENGINE is not None:
        return ENGINE

    settings = settings or load_settings()
    database_url = settings.database_url
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}

    if database_url.startswith("sqlite:///"):
        db_path = Path(database_url.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    ENGINE = create_engine(database_url, connect_args=connect_args, future=True)
    SessionFactory = sessionmaker(bind=ENGINE, autoflush=False, autocommit=False, expire_on_commit=False)
    return ENGINE


def init_db() -> None:
    """Create tables if they do not exist."""
    engine = _ensure_engine()
    Base.metadata.create_all(bind=engine)


def get_db_session():
    """Return a SQLAlchemy session bound to the engine."""
    factory = get_session_factory()
    return factory()


def get_session_factory():
    """Expose the configured session factory, initializing it if needed."""
    if SessionFactory is None:
        _ensure_engine()

    assert SessionFactory is not None
    return SessionFactory


@dataclass(slots=True)
class AssetData:
    """Typed representation for creating or updating an asset."""

    original_path: str
    filename: str
    processed_path: str | None
    thumbnail_path: str | None
    phash: str | None
    file_size: int | None
    width: int | None
    height: int | None
    caption: str | None = None
    caption_source: str = "blip"
    labels: List[tuple[str, float]] | None = None
    ocr_blocks: List[dict] | None = None


class AssetRepository:
    """Encapsulates common queries for ingestion and search."""

    def __init__(self, session) -> None:
        self.session = session

    def find_by_phash(self, phash: str) -> Asset | None:
        """Return the asset matching the given perceptual hash."""
        stmt = select(Asset).where(Asset.phash == phash)
        return self.session.scalar(stmt)

    def get_asset(self, asset_id: int) -> Asset | None:
        """Return a single asset by primary key."""
        return self.session.get(Asset, asset_id)

    def list_assets(self, limit: int = 20, offset: int = 0) -> List[Asset]:
        """Return a page of recently ingested assets."""
        stmt = select(Asset).order_by(Asset.id.desc()).offset(offset).limit(limit)
        return list(self.session.scalars(stmt))

    def create_asset(self, data: AssetData) -> Asset:
        """Persist a new asset and its associated metadata."""
        asset = Asset(
            filename=data.filename,
            original_path=data.original_path,
            processed_path=data.processed_path,
            thumbnail_path=data.thumbnail_path,
            phash=data.phash,
            file_size=data.file_size,
            width=data.width,
            height=data.height,
        )

        if data.caption:
            asset.captions.append(Caption(text=data.caption, source=data.caption_source))

        for label, confidence in data.labels or []:
            asset.labels.append(Label(label=label, confidence=confidence))

        for block in data.ocr_blocks or []:
            asset.text_blocks.append(
                TextBlock(
                    text=block.get("text", ""),
                    language=block.get("language"),
                    bbox=json.dumps(block.get("bbox")),
                )
            )

        self.session.add(asset)
        self.session.commit()
        self.session.refresh(asset)
        return asset

    def search_assets(self, query: str, limit: int = 20) -> List[Asset]:
        """Naive LIKE-based search across captions, labels, and OCR text."""
        wildcard = f"%{query.lower()}%"
        stmt = (
            select(Asset)
            .outerjoin(Caption)
            .outerjoin(Label)
            .outerjoin(TextBlock)
            .where(
                (func.lower(Caption.text).like(wildcard))
                | (func.lower(Label.label).like(wildcard))
                | (func.lower(TextBlock.text).like(wildcard))
            )
            .distinct()
            .limit(limit)
        )
        return list(self.session.scalars(stmt))

    def total_assets(self) -> int:
        """Return total number of assets stored."""
        stmt = select(func.count(Asset.id))
        return int(self.session.scalar(stmt) or 0)


def serialize_asset(asset: Asset) -> dict:
    """Convert an asset ORM object into a JSON-friendly dict."""
    return {
        "id": asset.id,
        "filename": asset.filename,
        "original_path": asset.original_path,
        "processed_path": asset.processed_path,
        "thumbnail_path": asset.thumbnail_path,
        "labels": [{"label": lbl.label, "confidence": lbl.confidence} for lbl in asset.labels],
        "captions": [{"text": cap.text, "source": cap.source} for cap in asset.captions],
        "ocr": [{"text": blk.text, "language": blk.language} for blk in asset.text_blocks],
    }
