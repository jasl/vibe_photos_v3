"""Unified detector that combines SigLIP labels with BLIP captions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import yaml

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
    ) -> None:
        self.classifier = SiglipClassifier(model_name=model)
        self.captioner = BlipCaptioner()
        self.device = device
        self.default_labels = load_candidate_labels(candidate_labels_path)

    def analyze(self, image_path: Path, candidate_labels: Sequence[str] | None = None) -> DetectionResult:
        """Run SigLIP + BLIP over the provided image path."""
        labels = candidate_labels or self.default_labels
        label_scores = self.classifier.classify(image_path=image_path, candidate_labels=labels, top_k=5)
        caption = self.captioner.generate_caption(image_path)
        return DetectionResult(labels=label_scores, caption=caption)
