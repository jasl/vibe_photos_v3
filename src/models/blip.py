"""BLIP captioning wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import threading
from PIL import Image

try:  # pragma: no cover - optional torch dependency
    import torch
except Exception:  # noqa: BLE001 - optional during CPU-only tests
    torch = None  # type: ignore[assignment]

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
        *,
        device_map: str | None = None,
        torch_dtype: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._processor = processor
        self._model = model
        self.device_map = device_map
        self.torch_dtype = torch_dtype

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

            model_kwargs: dict[str, Any] = {"use_safetensors": True}
            if self.device_map:
                model_kwargs["device_map"] = self.device_map
            dtype = self._resolve_dtype()
            if dtype is not None:
                model_kwargs["torch_dtype"] = dtype

            processor = BlipProcessor.from_pretrained(self.model_name, use_fast=True)
            model = BlipForConditionalGeneration.from_pretrained(self.model_name, **model_kwargs)
            model.eval()

            cached = (processor, model)
            _BLIP_CACHE[self.model_name] = cached

            self._processor, self._model = cached
            return cached

    def _resolve_dtype(self):
        if isinstance(self.torch_dtype, str) and self.torch_dtype.lower() in {"auto", "default"}:
            dtype_hint = None
        else:
            dtype_hint = self.torch_dtype

        if dtype_hint and torch is not None:
            return getattr(torch, str(dtype_hint), None)
        if dtype_hint is None and torch is not None:
            if torch.cuda.is_available():
                return torch.float16
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.float16
        return None

    def generate_caption(self, image: Path | Image.Image, max_length: int = 32) -> str:
        """Generate a caption for a single image."""
        processor, model = self._load_components()
        if isinstance(image, Path):
            image = Image.open(image).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        if torch is not None:
            try:
                device = next(model.parameters()).device  # type: ignore[call-arg]
            except StopIteration:  # pragma: no cover - defensive
                device = None
            if device is not None:
                inputs = {key: value.to(device) for key, value in inputs.items()}
        context = torch.inference_mode if torch is not None else _nullcontext
        with context():
            output_ids = model.generate(**inputs, max_length=max_length)
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()


def _nullcontext():
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        yield

    return _ctx()
