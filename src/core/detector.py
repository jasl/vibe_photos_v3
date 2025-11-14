"""Unified detector that combines SigLIP labels with BLIP captions."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import yaml

from PIL import Image

from src.models.blip import BlipCaptioner
from src.models.siglip import LabelScore, SiglipClassifier
from src.utils.config import REPO_ROOT


@dataclass(slots=True)
class DetectionResult:
    """Structured output for downstream persistence."""

    labels: List[LabelScore]
    caption: str | None


def load_candidate_labels(path: Path | None = None) -> List[str]:
    """Load canonical candidate labels from the configuration directory."""
    candidate_path = path or (REPO_ROOT / "config" / "candidates.yaml")
    if not candidate_path.exists():
        candidate_path = REPO_ROOT / "config" / "candidates.example.yaml"

    if not candidate_path.exists():
        return []

    with candidate_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    labels = data.get("labels", [])
    return [entry["name"] if isinstance(entry, dict) else str(entry) for entry in labels]

class SigLIPBLIPDetector:
    """High-level detector that returns labels and captions for an image."""

    def __init__(
        self,
        model: str,
        device: str = "auto",
        candidate_labels_path: Path | None = None,
        classifier: SiglipClassifier | None = None,
        captioner: BlipCaptioner | None = None,
    ) -> None:
        self.classifier = classifier or SiglipClassifier(model_name=model)
        self.captioner = captioner or BlipCaptioner()
        self.device = device
        self.default_labels = load_candidate_labels(candidate_labels_path)
        classify_signature = inspect.signature(self.classifier.classify)
        self._classifier_accepts_image = "image" in classify_signature.parameters

    def analyze(
        self,
        image_path: Path,
        *,
        processed_image: Image.Image | None = None,
        candidate_labels: Sequence[str] | None = None,
    ) -> DetectionResult:
        """Run SigLIP + BLIP over the provided image path."""
        labels = candidate_labels or self.default_labels
        classify_kwargs = {
            "image_path": image_path,
            "candidate_labels": labels,
            "top_k": 5,
        }
        if processed_image is not None and self._classifier_accepts_image:
            classify_kwargs["image"] = processed_image
        label_scores = self.classifier.classify(**classify_kwargs)
        caption_input = processed_image or image_path
        caption = self.captioner.generate_caption(caption_input)
        return DetectionResult(labels=label_scores, caption=caption)

    def analyze_batch(
        self,
        batch: Sequence[tuple[Path, Image.Image]],
        *,
        candidate_labels: Sequence[str] | None = None,
    ) -> List[DetectionResult]:
        """Run SigLIP + BLIP over an in-memory batch of images."""
        if not batch:
            return []

        labels = candidate_labels or self.default_labels
        images = [item[1] for item in batch]
        label_batches = self.classifier.classify_batch(images=images, candidate_labels=labels, top_k=5)

        results: List[DetectionResult] = []
        for (image_path, processed_image), label_scores in zip(batch, label_batches):
            caption = self.captioner.generate_caption(processed_image)
            results.append(DetectionResult(labels=label_scores, caption=caption))

        return results
