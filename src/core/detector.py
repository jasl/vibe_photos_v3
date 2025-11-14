"""Unified detector that combines SigLIP labels with BLIP captions."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import yaml

from PIL import Image

import src.models.siglip as siglip_module
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


def build_detector(config: Dict[str, Any], *, logger: logging.Logger | None = None) -> SigLIPBLIPDetector:
    """Construct a detector that respects device placement hints."""
    log = logger or logging.getLogger(__name__)
    detection_config = config.get("detection", {})
    model_name = detection_config.get("model", "google/siglip2-base-patch16-224")
    device = detection_config.get("device", "auto")
    device_map = detection_config.get("device_map", "auto")
    precision = detection_config.get("precision", "auto")

    def _normalize_device(value: str | int | None) -> str | int | None:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        lowered = value.lower()
        if lowered in {"auto", "default"}:
            return None
        return value

    device_arg = _normalize_device(device)
    device_map_arg = None if not device_map else device_map
    torch_dtype_hint = precision if isinstance(precision, str) and precision.lower() not in {"auto", "default"} else None

    torch_module = getattr(siglip_module, "torch", None)
    dtype_obj = getattr(torch_module, torch_dtype_hint, None) if torch_module is not None and torch_dtype_hint else None

    try:
        from transformers import (
            BlipForConditionalGeneration,
            BlipProcessor,
            pipeline as hf_pipeline,
        )
    except ImportError as error:  # pragma: no cover
        log.error(
            "transformers is required for SigLIP/BLIP support but is not installed",
            extra={"error": str(error)},
        )
        raise

    log.info(
        "Loading SigLIP zero-shot pipeline",
        extra={"model_name": model_name, "device": device_arg, "device_map": device_map_arg},
    )
    siglip_model_kwargs: Dict[str, Any] = {"use_safetensors": True}
    if device_map_arg:
        siglip_model_kwargs["device_map"] = device_map_arg
    if dtype_obj is not None:
        siglip_model_kwargs["torch_dtype"] = dtype_obj

    pipeline_device = None if device_map_arg else device_arg
    siglip_pipeline = hf_pipeline(
        "zero-shot-image-classification",
        model=model_name,
        device=pipeline_device,
        model_kwargs=siglip_model_kwargs,
        torch_dtype=dtype_obj,
    )
    classifier = SiglipClassifier(
        model_name=model_name,
        pipeline=siglip_pipeline,
        device=pipeline_device,
        device_map=device_map_arg,
        torch_dtype=torch_dtype_hint,
    )

    log.info(
        "SigLIP zero-shot pipeline loaded",
        extra={"model_name": model_name, "device": device},
    )

    blip_model_name = "Salesforce/blip-image-captioning-base"
    log.info(
        "Loading BLIP captioning components",
        extra={"model_name": blip_model_name, "device_map": device_map_arg},
    )
    blip_processor = BlipProcessor.from_pretrained(blip_model_name)
    blip_model_kwargs: Dict[str, Any] = {"use_safetensors": True}
    if device_map_arg:
        blip_model_kwargs["device_map"] = device_map_arg
    if dtype_obj is not None:
        blip_model_kwargs["torch_dtype"] = dtype_obj

    blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_name, **blip_model_kwargs)
    captioner = BlipCaptioner(
        model_name=blip_model_name,
        processor=blip_processor,
        model=blip_model,
        device_map=device_map_arg,
        torch_dtype=torch_dtype_hint,
    )

    log.info(
        "BLIP captioning components loaded",
        extra={"model_name": blip_model_name},
    )

    detector = SigLIPBLIPDetector(
        model=model_name,
        device=device,
        classifier=classifier,
        captioner=captioner,
    )

    if detection_config.get("warmup", True):
        classifier.ensure_loaded()

    return detector
