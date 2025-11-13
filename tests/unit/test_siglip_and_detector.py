from pathlib import Path

from PIL import Image

from src.core.detector import SigLIPBLIPDetector
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
