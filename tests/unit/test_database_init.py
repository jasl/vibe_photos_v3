import sqlite3
from types import SimpleNamespace

from src.core import database


def test_init_db_creates_tables_on_fresh_sqlite(tmp_path, monkeypatch):
    db_path = tmp_path / "fresh.db"
    settings = SimpleNamespace(database_url=f"sqlite:///{db_path}")

    monkeypatch.setattr(database, "ENGINE", None)
    monkeypatch.setattr(database, "SessionFactory", None)
    monkeypatch.setattr(database, "load_settings", lambda: settings)

    database.init_db()

    assert db_path.exists()

    connection = sqlite3.connect(db_path)
    try:
        columns = connection.execute("PRAGMA table_info(assets)").fetchall()
    finally:
        connection.close()

    column_names = {column[1] for column in columns}
    assert {"id", "filename", "duplicate_count"}.issubset(column_names)
