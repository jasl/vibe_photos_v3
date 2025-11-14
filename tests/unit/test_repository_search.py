from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.database import AssetData, AssetRepository, Base


def test_repository_search_returns_results():
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    repo = AssetRepository(session)
    asset = repo.create_asset(
        AssetData(
            original_path="/tmp/example.jpg",
            filename="example.jpg",
            processed_path="/tmp/processed.jpg",
            thumbnail_path="/tmp/thumb.jpg",
            phash="abcd",
            file_size=1024,
            width=400,
            height=300,
            caption="Recipe card with instructions",
            labels=[("recipe", 0.9)],
            ocr_blocks=[{"text": "Step 1: Mix ingredients"}],
        )
    )

    matches = repo.search_assets("recipe", limit=5)
    assert asset in matches
