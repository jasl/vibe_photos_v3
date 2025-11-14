import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from PIL import Image

from src.core.detector import SigLIPBLIPDetector, build_detector
from src.models.siglip import LabelScore, SiglipClassifier


def test_siglip_classifier_uses_injected_pipeline(tmp_path):
    calls = {"count": 0}

    class DummyPipeline:
        def __call__(self, image, candidate_labels, top_k=5):
            calls["count"] += 1
            return [
                {"label": candidate_labels[0], "score": 0.9},
                {"label": candidate_labels[1], "score": 0.1},
            ]

    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(image_path)

    pipeline = DummyPipeline()
    classifier = SiglipClassifier(model_name="dummy-model", pipeline=pipeline)

    labels = ["cat", "dog"]
    results = classifier.classify(image_path=image_path, candidate_labels=labels, top_k=2)

    assert calls["count"] == 1
    assert len(results) == 2
    assert isinstance(results[0], LabelScore)
    assert results[0].label == "cat"
    assert results[0].confidence == 0.9


def test_siglip_classifier_classify_batch_supports_image_keyword():
    class DummyPipeline:
        def __init__(self):
            self.last_kwargs = {}

        def __call__(self, image, candidate_labels, top_k=5):
            self.last_kwargs = {
                "image": image,
                "candidate_labels": list(candidate_labels),
                "top_k": top_k,
            }
            return [{"label": candidate_labels[0], "score": 0.6}]

    pipeline = DummyPipeline()
    classifier = SiglipClassifier(model_name="dummy-model", pipeline=pipeline)
    images = [Image.new("RGB", (8, 8), color=(0, 0, 0))]
    labels = ["city"]

    results = classifier.classify_batch(images=images, candidate_labels=labels, top_k=1)

    assert pipeline.last_kwargs["candidate_labels"] == labels
    assert pipeline.last_kwargs["top_k"] == 1
    assert isinstance(pipeline.last_kwargs["image"], list)
    assert len(pipeline.last_kwargs["image"]) == len(images)
    assert results[0][0].label == "city"
    assert results[0][0].confidence == 0.6


def test_siglip_classifier_classify_batch_supports_images_keyword():
    class DummyPipeline:
        def __init__(self):
            self.last_kwargs = {}

        def __call__(self, *, images, candidate_labels, top_k=5):
            self.last_kwargs = {
                "images": images,
                "candidate_labels": list(candidate_labels),
                "top_k": top_k,
            }
            return [{"label": candidate_labels[0], "score": 0.7}]

    pipeline = DummyPipeline()
    classifier = SiglipClassifier(model_name="dummy-model", pipeline=pipeline)
    images = [Image.new("RGB", (10, 10), color=(255, 255, 255))]
    labels = ["landmark"]

    results = classifier.classify_batch(images=images, candidate_labels=labels, top_k=3)

    assert pipeline.last_kwargs["candidate_labels"] == labels
    assert pipeline.last_kwargs["top_k"] == 3
    assert isinstance(pipeline.last_kwargs["images"], list)
    assert len(pipeline.last_kwargs["images"]) == len(images)
    assert results[0][0].label == "landmark"
    assert results[0][0].confidence == 0.7


def test_detector_uses_injected_classifier_and_captioner(monkeypatch, tmp_path):
    image_path = tmp_path / "photo.jpg"
    Image.new("RGB", (16, 16), color=(0, 255, 0)).save(image_path)

    class DummyClassifier:
        def __init__(self):
            self.last_call = None

        def classify(self, image_path, candidate_labels, top_k=5):
            self.last_call = {
                "image_path": image_path,
                "candidate_labels": list(candidate_labels),
                "top_k": top_k,
            }
            return [LabelScore(label="dummy-label", confidence=1.0)]

    class DummyCaptioner:
        def __init__(self):
            self.last_call = None

        def generate_caption(self, image_path, max_length=32):
            self.last_call = {
                "image_path": image_path,
                "max_length": max_length,
            }
            return "dummy-caption"

    dummy_classifier = DummyClassifier()
    dummy_captioner = DummyCaptioner()
    monkeypatch.setattr("src.core.detector.BlipCaptioner", lambda: dummy_captioner)

    detector = SigLIPBLIPDetector(model="dummy-model", classifier=dummy_classifier)

    result = detector.analyze(image_path=Path(image_path))

    assert detector.classifier is dummy_classifier
    assert dummy_classifier.last_call is not None
    assert dummy_classifier.last_call["image_path"] == image_path
    assert dummy_classifier.last_call["top_k"] == 5

    assert dummy_captioner.last_call is not None
    assert dummy_captioner.last_call["image_path"] == image_path
    assert result.caption == "dummy-caption"
    assert result.labels[0].label == "dummy-label"


def test_build_detector_respects_explicit_device(monkeypatch):
    captured = {}

    def fake_pipeline(task, model, device=None, model_kwargs=None, torch_dtype=None):
        captured["task"] = task
        captured["model"] = model
        captured["device"] = device
        captured["model_kwargs"] = model_kwargs
        captured["torch_dtype"] = torch_dtype
        return object()

    _install_transformer_stubs(monkeypatch, fake_pipeline, captured)

    fake_torch = SimpleNamespace(float16="fake-float16")
    monkeypatch.setattr("src.core.detector.siglip_module.torch", fake_torch, raising=False)

    config = {
        "detection": {
            "model": "siglip-test",
            "device": "cuda:0",
            "device_map": None,
            "precision": "float16",
            "warmup": False,
        }
    }

    detector = build_detector(config)

    assert captured["task"] == "zero-shot-image-classification"
    assert captured["device"] == "cuda:0"
    assert captured["model_kwargs"]["use_safetensors"] is True
    assert "device_map" not in captured["model_kwargs"]
    assert captured["torch_dtype"] == fake_torch.float16
    assert detector.device == "cuda:0"


def test_build_detector_handles_device_map_auto(monkeypatch):
    captured = {}

    def fake_pipeline(task, model, device=None, model_kwargs=None, torch_dtype=None):
        captured["device"] = device
        captured["model_kwargs"] = model_kwargs
        captured["torch_dtype"] = torch_dtype
        return object()

    _install_transformer_stubs(monkeypatch, fake_pipeline, captured)

    config = {
        "detection": {
            "model": "siglip-test",
            "device": "cuda:0",
            "device_map": "auto",
            "precision": "auto",
            "warmup": False,
        }
    }

    build_detector(config)

    assert captured["device"] is None
    assert captured["model_kwargs"]["device_map"] == "auto"
    assert captured["torch_dtype"] is None


def _install_transformer_stubs(monkeypatch, pipeline_impl, captured):
    class DummyBlipProcessor:
        @classmethod
        def from_pretrained(cls, model_name):
            captured["blip_processor_model"] = model_name
            return cls()

    class DummyBlipModel:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            captured["blip_model_kwargs"] = kwargs
            instance = cls()
            instance.model_name = model_name
            instance.kwargs = kwargs
            return instance

    fake_module = ModuleType("transformers")
    fake_module.pipeline = pipeline_impl
    fake_module.BlipProcessor = DummyBlipProcessor
    fake_module.BlipForConditionalGeneration = DummyBlipModel
    monkeypatch.setitem(sys.modules, "transformers", fake_module)
