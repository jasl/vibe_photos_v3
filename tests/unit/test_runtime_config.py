import yaml

from src.utils import runtime


def test_load_phase1_config_merges_settings(tmp_path, monkeypatch):
    blueprint = tmp_path / "blueprint.yaml"
    blueprint.write_text(
        yaml.safe_dump({"dataset": {"directory": "samples"}, "logging": {"level": "INFO"}}),
        encoding="utf-8",
    )

    overrides = tmp_path / "settings.yaml"
    overrides.write_text(
        yaml.safe_dump({"dataset": {"directory": "custom"}, "logging": {"level": "DEBUG"}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(runtime, "BLUEPRINT_CONFIG", blueprint, raising=False)
    monkeypatch.setattr(runtime, "DEFAULT_SETTINGS_PATH", overrides, raising=False)

    config = runtime.load_phase1_config()
    assert config["dataset"]["directory"] == "custom"
    assert config["logging"]["level"] == "DEBUG"
