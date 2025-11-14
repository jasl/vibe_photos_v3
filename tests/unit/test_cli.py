from typer.testing import CliRunner

from src import cli


runner = CliRunner()


def test_ingest_command_invokes_process(monkeypatch):
    called = {"run": False, "incremental": None}

    class DummyProcessor:
        async def process_dataset(self, incremental: bool = True) -> None:
            called["run"] = True
            called["incremental"] = incremental

        def get_statistics(self) -> dict:
            return {"successful": 1, "duplicates": 0, "failed": 0}

    def fake_build_processor(config: dict) -> DummyProcessor:
        return DummyProcessor()

    monkeypatch.setattr(cli, "_build_processor", fake_build_processor)

    result = runner.invoke(cli.cli, ["ingest"])
    assert result.exit_code == 0
    assert called["run"] is True


def test_search_command_handles_no_results(monkeypatch):
    class DummyService:
        def search(self, query: str, limit: int = 10):
            return []

    monkeypatch.setattr(cli, "init_db", lambda: None)
    monkeypatch.setattr(cli, "AssetSearchService", lambda: DummyService())

    result = runner.invoke(cli.cli, ["search", "nothing"])
    assert result.exit_code == 0
    assert "No matches found." in result.stdout


def test_build_processor_uses_detector_factory(monkeypatch):
    captured = {}
    config = {
        "preprocessing": {"paths": {"processed": "cache/processed"}},
        "ocr": {"enabled": False},
        "detection": {"model": "siglip-test"},
    }

    monkeypatch.setattr(cli, "init_db", lambda: None)
    monkeypatch.setattr(cli, "get_session_factory", lambda: "session-factory")

    class DummyPreprocessor:
        def __init__(self, settings):
            captured["preprocessing"] = settings

    detector_sentinel = object()

    def fake_build_detector(cfg):
        captured["detector_config"] = cfg
        return detector_sentinel

    class DummyBatchProcessor:
        def __init__(self, **kwargs):
            captured["batch_kwargs"] = kwargs

    monkeypatch.setattr(cli, "ImagePreprocessor", DummyPreprocessor)
    monkeypatch.setattr(cli, "build_detector", fake_build_detector)
    monkeypatch.setattr(cli, "BatchProcessor", DummyBatchProcessor)

    processor = cli._build_processor(config)

    assert captured["preprocessing"] == config["preprocessing"]
    assert captured["detector_config"] is config
    assert captured["batch_kwargs"]["detector"] is detector_sentinel
    assert captured["batch_kwargs"]["session_factory"] == "session-factory"
    assert captured["batch_kwargs"]["ocr_engine"] is None
    assert isinstance(processor, DummyBatchProcessor)
