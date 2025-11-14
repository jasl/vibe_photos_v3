from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.database import AssetData, AssetRepository, Base
from src.services.cache import CacheImporter, CacheWriter, CachedArtifact


def _build_session_factory(db_path: Path):
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def test_cache_writer_persists_artifact(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    writer = CacheWriter(cache_dir)

    asset_data = AssetData(
        original_path="/images/sample.jpg",
        filename="sample.jpg",
        processed_path="/processed/sample.jpg",
        thumbnail_path=None,
        phash="abc123",
        file_size=1024,
        width=800,
        height=600,
        caption="test",
        labels=[("object", 0.9)],
        ocr_blocks=[],
    )
    artifact = CachedArtifact(asset=asset_data, timings={"preprocess": 0.1}, source_path=asset_data.original_path)

    cache_file = writer.write(artifact)
    assert cache_file.exists()
    content = cache_file.read_text(encoding="utf-8")
    assert "sample.jpg" in content
    assert "object" in content


def test_cache_importer_round_trip(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    writer = CacheWriter(cache_dir)
    asset_data = AssetData(
        original_path="/images/another.jpg",
        filename="another.jpg",
        processed_path="/processed/another.jpg",
        thumbnail_path=None,
        phash="def456",
        file_size=2048,
        width=1024,
        height=768,
        caption="caption",
        labels=[("cat", 0.8)],
        ocr_blocks=[{"text": "hello"}],
    )
    artifact = CachedArtifact(asset=asset_data, timings={"preprocess": 0.2}, source_path=asset_data.original_path)
    writer.write(artifact)

    session_factory = _build_session_factory(tmp_path / "db.sqlite")
    importer = CacheImporter(cache_dir, session_factory)

    imported = importer.import_all()
    assert imported == 1

    session = session_factory()
    try:
        repository = AssetRepository(session)
        stored = repository.find_by_original_path(asset_data.original_path)
        assert stored is not None
        assert stored.filename == asset_data.filename
        assert stored.phash == asset_data.phash
    finally:
        session.close()

    imported_again = importer.import_all()
    assert imported_again == 0
