import sys
import warnings
from types import ModuleType

import pytest

from src.core.detector import build_detector
from src.core.ocr import PaddleOCREngine


@pytest.fixture
def stub_transformers(monkeypatch):
    captured = {}

    class DummyPipeline:
        def __call__(self, *args, **kwargs):
            return []

    def fake_pipeline(task, model, device=None, model_kwargs=None, torch_dtype=None):
        captured.update(
            {
                "task": task,
                "model": model,
                "device": device,
                "model_kwargs": model_kwargs,
                "torch_dtype": torch_dtype,
            }
        )
        return DummyPipeline()

    class DummyProcessor:
        def __init__(self):
            self.model_name = None

        @classmethod
        def from_pretrained(cls, model_name, use_fast=True):
            instance = cls()
            instance.model_name = model_name
            return instance

        def __call__(self, images, return_tensors="pt"):
            return {"pixel_values": images}

        def decode(self, _output_ids, skip_special_tokens=True):
            return "stub-caption"

    class DummyModel:
        def __init__(self):
            self._eval = False

        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            instance = cls()
            instance.model_name = model_name
            instance.kwargs = kwargs
            return instance

        def eval(self):
            self._eval = True
            return self

        def generate(self, **_kwargs):
            return [[0]]

    fake_module = ModuleType("transformers")
    fake_module.pipeline = fake_pipeline
    fake_module.BlipProcessor = DummyProcessor
    fake_module.BlipForConditionalGeneration = DummyModel

    monkeypatch.setitem(sys.modules, "transformers", fake_module)
    return captured


def test_build_detector_warmup_is_warning_clean(stub_transformers):
    config = {"detection": {"model": "siglip-test"}}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        detector = build_detector(config)

    assert detector.classifier is not None
    assert stub_transformers["task"] == "zero-shot-image-classification"
    assert not [item for item in caught if issubclass(item.category, DeprecationWarning)]


def test_paddleocr_warmup_is_warning_clean(monkeypatch):
    class DummyEngine:
        def __init__(self, *args, **kwargs):
            pass

        def ocr(self, *_args, **_kwargs):
            return []

    fake_module = ModuleType("paddleocr")
    fake_module.PaddleOCR = DummyEngine
    monkeypatch.setitem(sys.modules, "paddleocr", fake_module)

    engine = PaddleOCREngine({"enabled": True, "languages": ["en"]})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        engine.warmup()

    assert not [item for item in caught if issubclass(item.category, DeprecationWarning)]
