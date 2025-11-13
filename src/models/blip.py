"""BLIP captioning wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import threading
from PIL import Image

_BLIP_CACHE: dict[str, tuple[Any, Any]] = {}
_BLIP_LOCK = threading.Lock()


class BlipCaptioner:
    """Generate captions for images using BLIP.

    The captioner lazily imports and constructs BLIP components on first use. For
    advanced scenarios (e.g. pre-loading models in an entrypoint), you can inject
    a pre-built processor and model via the ``processor`` and ``model`` arguments.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        processor: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self._processor = processor
        self._model = model

    def _load_components(self):
        """Instantiate BLIP lazily to avoid blocking startup."""
        if self._model is not None and self._processor is not None:
            return self._processor, self._model

        with _BLIP_LOCK:
            if self._model is not None and self._processor is not None:
                return self._processor, self._model

            cached = _BLIP_CACHE.get(self.model_name)
            if cached is not None:
                self._processor, self._model = cached
                return cached

            try:
                from transformers import BlipForConditionalGeneration, BlipProcessor
            except ImportError as error:  # pragma: no cover - exercised only without deps
                message = (
                    "transformers is required for BLIP captioning "
                    "(failed to import BlipForConditionalGeneration/BlipProcessor)"
                )
                raise RuntimeError(message) from error

            processor = BlipProcessor.from_pretrained(self.model_name, use_fast=True)
            model = BlipForConditionalGeneration.from_pretrained(self.model_name)

            cached = (processor, model)
            _BLIP_CACHE[self.model_name] = cached

            self._processor, self._model = cached
            return cached

    def generate_caption(self, image_path: Path, max_length: int = 32) -> str:
        """Generate a caption for a single image."""
        processor, model = self._load_components()
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        output_ids = model.generate(**inputs, max_length=max_length)
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()
