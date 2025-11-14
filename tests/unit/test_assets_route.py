from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.routes import assets as assets_route
from src.api.services import dependencies
from src.core.database import AssetData, AssetRepository, Base


def create_app_with_asset():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    repository = AssetRepository(session)
    asset = repository.create_asset(
        AssetData(
            original_path="/tmp/example.jpg",
            filename="example.jpg",
            processed_path="/tmp/processed.jpg",
            thumbnail_path="/tmp/thumb.jpg",
            phash="abcd",
            file_size=1024,
            width=400,
            height=300,
            caption="Sample caption",
            labels=[("recipe", 0.9)],
            ocr_blocks=[{"text": "Sample text"}],
        )
    )

    app = FastAPI()
    app.include_router(assets_route.router)

    def get_test_session():
        yield session

    app.dependency_overrides[dependencies.get_db_session] = get_test_session
    return app, asset


def test_list_assets_returns_results():
    app, asset = create_app_with_asset()
    client = TestClient(app)

    response = client.get("/assets/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert len(payload["data"]) == 1
    assert payload["data"][0]["id"] == asset.id


def test_get_asset_returns_single_asset():
    app, asset = create_app_with_asset()
    client = TestClient(app)

    response = client.get(f"/assets/{asset.id}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["data"]["id"] == asset.id
    assert payload["data"]["filename"] == "example.jpg"


def test_get_asset_returns_404_for_missing_asset():
    app, _ = create_app_with_asset()
    client = TestClient(app)

    response = client.get("/assets/9999")
    assert response.status_code == 404
