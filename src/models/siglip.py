"""SigLIP zero-shot classifier wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

from PIL import Image

LabelScores = List["LabelScore"]


@dataclass(slots=True)
class LabelScore:
    """Represents a predicted label and its confidence."""

    label: str
    confidence: float


class SiglipClassifier:
    """Lazy-loading helper for SigLIP zero-shot image classification.

    The classifier lazily imports and constructs the Hugging Face pipeline on first use.
    For advanced scenarios (e.g. pre-loading the pipeline once in a custom entrypoint),
    you can inject a pre-built pipeline instance via the ``pipeline`` argument.
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        pipeline: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self._pipeline = pipeline

    def _load_pipeline(self):
        """Instantiate the huggingface pipeline on demand."""
        if self._pipeline is not None:
            return self._pipeline

        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as error:  # pragma: no cover - exercised only without deps
            message = "transformers is required for SigLIP support (failed to import transformers.pipeline)"
            raise RuntimeError(message) from error

        self._pipeline = hf_pipeline("zero-shot-image-classification", model=self.model_name, use_fast=True)
        return self._pipeline

    def classify(
        self,
        image_path: Path,
        candidate_labels: Sequence[str],
        top_k: int = 5,
    ) -> LabelScores:
        """Run zero-shot classification for the provided labels."""
        if not candidate_labels:
            return []

        pipeline = self._load_pipeline()
        image = Image.open(image_path).convert("RGB")
        result = pipeline(image=image, candidate_labels=list(candidate_labels), top_k=top_k)

        # The pipeline returns either a list of dicts or a dict depending on transformers version.
        if isinstance(result, dict):
            iterable: Iterable[dict] = [result]
        else:
            iterable = result  # type: ignore[assignment]

        return [
            LabelScore(label=item["label"], confidence=float(item["score"]))
            for item in iterable
        ]
