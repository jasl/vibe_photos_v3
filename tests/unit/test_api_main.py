from src.api import main


def test_bootstrap_runtime_uses_detector_factory(monkeypatch):
    config = {
        "preprocessing": {"paths": {"processed": "cache/processed"}},
        "detection": {"model": "siglip-test"},
        "ocr": {"enabled": False},
    }
    captured = {}

    monkeypatch.setattr(main, "load_phase1_config", lambda: config)
    monkeypatch.setattr(main, "init_db", lambda: None)
    monkeypatch.setattr(main, "get_session_factory", lambda: lambda: "session")

    class DummyPreprocessor:
        def __init__(self, settings):
            captured["preprocessing"] = settings

    detector_sentinel = object()

    def fake_build_detector(cfg, *, logger=None):
        captured["detector_config"] = cfg
        captured["logger"] = logger
        return detector_sentinel

    monkeypatch.setattr(main, "ImagePreprocessor", DummyPreprocessor)
    monkeypatch.setattr(main, "build_detector", fake_build_detector)

    runtime = main._bootstrap_runtime()

    assert captured["preprocessing"] == config["preprocessing"]
    assert captured["detector_config"] is config
    assert captured["logger"] is main.logger
    assert runtime.detector is detector_sentinel
    assert runtime.session_factory() == "session"

