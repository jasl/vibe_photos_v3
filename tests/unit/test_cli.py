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
