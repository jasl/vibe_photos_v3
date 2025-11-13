from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.api.routes import ingest


def test_derive_destination_normalizes_traversal(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    destination = ingest._derive_destination_path(dataset_dir, "../../config/settings.yaml")

    assert destination.parent == dataset_dir.resolve()
    assert destination.name == "settings.yaml"


def test_derive_destination_rejects_windows_separators(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    with pytest.raises(HTTPException):
        ingest._derive_destination_path(dataset_dir, "..\\..\\evil.txt")


def test_derive_destination_generates_uuid_when_missing_filename(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    monkeypatch.setattr(
        ingest,
        "uuid4",
        lambda: SimpleNamespace(hex="deadbeefcafebabe"),
    )

    destination = ingest._derive_destination_path(dataset_dir, "")

    assert destination.parent == dataset_dir.resolve()
    assert destination.name == "upload-deadbeefcafebabe"
