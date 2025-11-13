"""SigLIP zero-shot classifier wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import threading
from PIL import Image

LabelScores = List["LabelScore"]


@dataclass(slots=True)
class LabelScore:
    """Represents a predicted label and its confidence."""

    label: str
    confidence: float


_PIPELINE_CACHE: Dict[str, Any] = {}
_PIPELINE_LOCK = threading.Lock()


class SiglipClassifier:
    """Lazy-loading helper for SigLIP zero-shot image classification.

    The classifier lazily imports and constructs the Hugging Face pipeline on first use.
    For advanced scenarios (e.g. pre-loading the pipeline once in a custom entrypoint),
    you can inject a pre-built pipeline instance via the ``pipeline`` argument.
    """

    def __init__(self, model_name: str = "google/siglip2-base-patch16-224", pipeline: Any | None = None) -> None:
        self.model_name = model_name
        self._pipeline = pipeline

    def _load_pipeline(self):
        """Instantiate the huggingface pipeline on demand."""
        if self._pipeline is not None:
            return self._pipeline

        with _PIPELINE_LOCK:
            if self._pipeline is not None:
                return self._pipeline

            cached = _PIPELINE_CACHE.get(self.model_name)
            if cached is not None:
                self._pipeline = cached
                return cached

            try:
                from transformers import pipeline as hf_pipeline
            except ImportError as error:  # pragma: no cover - exercised only without deps
                message = "transformers is required for SigLIP support (failed to import transformers.pipeline)"
                raise RuntimeError(message) from error

            pipeline = hf_pipeline("zero-shot-image-classification", model=self.model_name, use_fast=True)
            _PIPELINE_CACHE[self.model_name] = pipeline
            self._pipeline = pipeline
            return pipeline

    def _classify_single(
        self,
        image_path: Path,
        candidate_labels: Sequence[str],
        top_k: int,
    ) -> LabelScores:
        """Fallback single-image classification without Hugging Face Datasets."""
        pipeline = self._load_pipeline()
        image = Image.open(image_path).convert("RGB")
        result = pipeline(image=image, candidate_labels=list(candidate_labels), top_k=top_k)

        if isinstance(result, dict):
            iterable: Iterable[dict] = [result]
        else:
            iterable = result  # type: ignore[assignment]

        return [
            LabelScore(label=item["label"], confidence=float(item["score"]))
            for item in iterable
        ]

    def classify(
        self,
        image_path: Path,
        candidate_labels: Sequence[str],
        top_k: int = 5,
    ) -> LabelScores:
        """Run zero-shot classification for the provided labels.

        When the optional ``datasets`` library is available, this method wraps the
        pipeline call with a tiny ``datasets.Dataset`` and ``map(..., batched=True)``
        invocation. This keeps the implementation aligned with Hugging Face's
        recommended dataset integration while preserving the existing single-image
        behavior and test doubles.
        """
        if not candidate_labels:
            return []

        try:
            from datasets import Dataset, Image as HFImage  # type: ignore[import]
        except Exception:
            # Datasets is not installed; fall back to the original single-image path.
            return self._classify_single(image_path=image_path, candidate_labels=candidate_labels, top_k=top_k)

        pipeline = self._load_pipeline()

        # Build a minimal dataset with a single image entry and use `map` in batched
        # mode. Even with one item, this keeps the code ready for future batch
        # extensions while matching the existing classifier semantics.
        dataset = Dataset.from_dict({"image": [str(image_path)]})
        dataset = dataset.cast_column("image", HFImage())

        def _apply(batch: Dict[str, Any]) -> Dict[str, Any]:
            images = batch["image"]
            # We expect a single image in the batch for now.
            image_obj = images[0]
            outputs = pipeline(image=image_obj, candidate_labels=list(candidate_labels), top_k=top_k)

            if isinstance(outputs, dict):
                items: Iterable[dict] = [outputs]
            else:
                items = outputs  # type: ignore[assignment]

            labels = [str(item["label"]) for item in items]
            scores = [float(item["score"]) for item in items]
            # Wrap in lists so the shapes align with the batched map expectation.
            return {"labels": [labels], "scores": [scores]}

        mapped = dataset.map(_apply, batched=True, batch_size=1)
        labels = mapped["labels"][0]
        scores = mapped["scores"][0]

        return [
            LabelScore(label=str(label), confidence=float(score))
            for label, score in zip(labels, scores)
        ]
