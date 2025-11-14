"""SigLIP zero-shot classifier wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import threading
from PIL import Image

try:  # pragma: no cover - optional torch dependency
    import torch
except Exception:  # noqa: BLE001 - torch might be absent in CPU-only tests
    torch = None  # type: ignore[assignment]

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

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        pipeline: Any | None = None,
        *,
        device: str | int | None = None,
        device_map: str | None = None,
        torch_dtype: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._pipeline = pipeline
        self.device = device
        self.device_map = device_map
        self.torch_dtype = torch_dtype

    def _load_pipeline(self):
        """Instantiate the huggingface pipeline on demand."""
        if self._pipeline is not None:
            return self._pipeline

        with _PIPELINE_LOCK:
            if self._pipeline is not None:
                return self._pipeline

            cache_key = self._cache_key
            cached = _PIPELINE_CACHE.get(cache_key)
            if cached is not None:
                self._pipeline = cached
                return cached

            try:
                from transformers import pipeline as hf_pipeline
            except ImportError as error:  # pragma: no cover - exercised only without deps
                message = "transformers is required for SigLIP support (failed to import transformers.pipeline)"
                raise RuntimeError(message) from error

            model_kwargs: Dict[str, Any] = {"use_safetensors": True}
            if self.device_map:
                model_kwargs["device_map"] = self.device_map
            dtype = self._resolve_dtype()
            if dtype is not None:
                model_kwargs.setdefault("torch_dtype", dtype)

            pipeline = hf_pipeline(
                "zero-shot-image-classification",
                model=self.model_name,
                use_fast=True,
                device=self._resolve_device_argument(),
                model_kwargs=model_kwargs,
                torch_dtype=dtype,
            )
            _PIPELINE_CACHE[cache_key] = pipeline
            self._pipeline = pipeline
            return pipeline

    @property
    def _cache_key(self) -> str:
        """Return a cache key that includes device placement hints."""
        return ":".join(
            [
                self.model_name,
                str(self.device_map or "none"),
                str(self.device or "auto"),
                str(self.torch_dtype or "auto"),
            ]
        )

    def _resolve_device_argument(self) -> int | str | None:
        """Determine the pipeline ``device`` argument."""
        if self.device_map:
            # device_map takes priority and mutually excludes explicit device indices.
            return None
        return self.device

    def _resolve_dtype(self) -> Any | None:  # type: ignore[override]
        """Resolve the torch dtype object from a string hint."""
        if isinstance(self.torch_dtype, str) and self.torch_dtype.lower() in {"auto", "default"}:
            dtype_hint = None
        else:
            dtype_hint = self.torch_dtype

        if dtype_hint and torch is not None:
            return getattr(torch, str(dtype_hint), None)
        if self.torch_dtype is None and torch is not None and self._is_cuda_available():
            return torch.float16
        return None

    @staticmethod
    def _is_cuda_available() -> bool:
        return bool(torch and torch.cuda.is_available())

    def _classify_single(
        self,
        image_path: Path | None,
        image: Image.Image | None,
        candidate_labels: Sequence[str],
        top_k: int,
    ) -> LabelScores:
        """Fallback single-image classification without Hugging Face Datasets."""
        pipeline = self._load_pipeline()
        if image is None:
            if image_path is None:
                message = "Either image_path or image must be provided"
                raise ValueError(message)
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
        *,
        image: Image.Image | None = None,
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
            return self._classify_single(
                image_path=image_path,
                image=image,
                candidate_labels=candidate_labels,
                top_k=top_k,
            )

        pipeline = self._load_pipeline()

        # Build a minimal dataset with a single image entry and use `map` in batched
        # mode. Even with one item, this keeps the code ready for future batch
        # extensions while matching the existing classifier semantics.
        image_column: List[Any]
        if image is not None:
            image_column = [image]
        else:
            image_column = [str(image_path)]

        dataset = Dataset.from_dict({"image": image_column})
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

    def classify_batch(
        self,
        images: Sequence[Image.Image],
        candidate_labels: Sequence[str],
        top_k: int = 5,
    ) -> List[LabelScores]:
        """Run zero-shot classification for a batch of in-memory images."""
        if not images:
            return []

        pipeline = self._load_pipeline()
        outputs = pipeline(images=list(images), candidate_labels=list(candidate_labels), top_k=top_k)
        if isinstance(outputs, dict):
            outputs = [outputs]

        batched: List[LabelScores] = []
        for entry in outputs:
            if isinstance(entry, list):
                iterable = entry
            else:
                iterable = [entry]
            batched.append(
                [LabelScore(label=str(item["label"]), confidence=float(item["score"])) for item in iterable]
            )
        return batched

    def ensure_loaded(self) -> None:
        """Ensure the underlying pipeline is instantiated (for warmup)."""
        self._load_pipeline()
