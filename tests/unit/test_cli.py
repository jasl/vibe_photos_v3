from typer.testing import CliRunner

from src import cli


runner = CliRunner()


def test_ingest_command_invokes_process(monkeypatch):
    called = {"run": False}

    async def fake_main():
        called["run"] = True
        return 0

    monkeypatch.setattr("process_dataset.main", fake_main)

    result = runner.invoke(cli.app, ["ingest"])
    assert result.exit_code == 0
    assert called["run"]


def test_search_command_handles_no_results(monkeypatch):
    class DummyService:
        def search(self, query: str, limit: int = 10):
            return []

    monkeypatch.setattr(cli, "load_phase1_config", lambda: {})
    monkeypatch.setattr(cli, "init_db", lambda: None)
    monkeypatch.setattr(cli, "get_session_factory", lambda: None)
    monkeypatch.setattr(cli, "AssetSearchService", lambda session_factory: DummyService())

    result = runner.invoke(cli.app, ["search", "nothing"])
    assert result.exit_code == 0
    assert "No matches found." in result.stdout
