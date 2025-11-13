"""BLIP captioning wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image


class BlipCaptioner:
    """Generate captions for images using BLIP."""

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base") -> None:
        self.model_name = model_name
        self._model = None
        self._processor = None

    def _load_components(self):
        """Instantiate BLIP lazily to avoid blocking startup."""
        if self._model is not None and self._processor is not None:
            return self._processor, self._model

        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as error:  # pragma: no cover - exercised only without deps
            raise RuntimeError("transformers is required for BLIP captioning") from error

        self._processor = BlipProcessor.from_pretrained(self.model_name)
        self._model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        return self._processor, self._model

    def generate_caption(self, image_path: Path, max_length: int = 32) -> str:
        """Generate a caption for a single image."""
        processor, model = self._load_components()
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        output_ids = model.generate(**inputs, max_length=max_length)
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()
