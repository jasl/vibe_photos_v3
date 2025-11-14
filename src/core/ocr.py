"""PaddleOCR wrapper used during ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import threading

from src.utils.logging import get_logger

_ENGINE_CACHE: Dict[str, Any] = {}
_ENGINE_LOCK = threading.Lock()
_ENGINE_CALL_LOCKS: Dict[str, threading.Lock] = {}


@dataclass(slots=True)
class OCRText:
    """Represents a single OCR text block."""

    text: str
    confidence: float
    language: str | None
    bbox: List[List[float]] | None


class PaddleOCREngine:
    """Thin wrapper around PaddleOCR with lazy initialization."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.enabled = config.get("enabled", True)
        self._ocr = None
        self.logger = get_logger(__name__)

    def _engine_key(self) -> str:
        languages = self.config.get("languages", ["ch"])
        if isinstance(languages, list):
            return ",".join(str(lang) for lang in languages)
        return str(languages)

    def _load_engine(self):
        if not self.enabled:
            return None

        if self._ocr is not None:
            return self._ocr

        with _ENGINE_LOCK:
            if self._ocr is not None:
                return self._ocr

            cached = _ENGINE_CACHE.get(self._engine_key())
            if cached is not None:
                self._ocr = cached
                return cached

            try:
                from paddleocr import PaddleOCR
            except ImportError as error:  # pragma: no cover - executed only without deps
                raise RuntimeError("paddleocr is required for OCR support") from error

            use_angle_cls = self.config.get("use_angle_cls", True)
            languages = self.config.get("languages", ["ch"])
            lang = languages[0] if isinstance(languages, list) else languages
            # PaddleOCR>=3.3 dropped the legacy `show_log` and `use_angle_cls` constructor
            # arguments in favor of `use_textline_orientation`. We keep the configuration
            # key name for backwards compatibility and map it to the new flag here.
            engine = PaddleOCR(lang=lang, use_textline_orientation=use_angle_cls)

            key = self._engine_key()
            _ENGINE_CACHE[self._engine_key()] = engine
            if key not in _ENGINE_CALL_LOCKS:
                _ENGINE_CALL_LOCKS[key] = threading.Lock()
            self._ocr = engine
            return engine

    def extract_text(self, image_path: Path) -> List[OCRText]:
        """Run OCR on the given image."""
        engine = self._load_engine()
        if engine is None:
            return []

        languages = self.config.get("languages", ["ch"])
        lang_value = ",".join(languages) if isinstance(languages, list) else str(languages)

        # Serialize calls to the shared PaddleOCR engine as it is not
        # thread-safe under heavy concurrent usage.
        key = self._engine_key()
        call_lock = _ENGINE_CALL_LOCKS.get(key)
        if call_lock is None:
            call_lock = threading.Lock()
            _ENGINE_CALL_LOCKS[key] = call_lock

        try:
            # The `cls`/`use_angle_cls` parameter used in older PaddleOCR versions has
            # been replaced by `use_textline_orientation` on the engine itself, so we
            # no longer pass it per-call and instead rely on the engine configuration.
            with call_lock:
                result = engine.ocr(str(image_path))
        except Exception as error:  # noqa: BLE001
            # Hardening: treat OCR failures as non-fatal and surface them via logging
            # instead of aborting the ingestion pipeline.
            self.logger.exception(
                "OCR failed for image %s: %s",
                str(image_path),
                error,
                extra={"path": str(image_path), "error": str(error)},
            )
            return []

        ocr_text: List[OCRText] = []

        for item in result or []:
            # New PaddleOCR>=3.3 path: list[OCRResult], where each item is a dict-like
            # object exposing `rec_texts`, `rec_scores`, and `rec_polys`.
            if isinstance(item, dict) and "rec_texts" in item:
                texts = item.get("rec_texts") or []
                scores = item.get("rec_scores") or []
                polys = item.get("rec_polys") or []
                for text, score, bbox in zip(texts, scores, polys):
                    bbox_value = bbox.tolist() if hasattr(bbox, "tolist") else bbox
                    ocr_text.append(
                        OCRText(
                            text=str(text),
                            confidence=float(score),
                            language=lang_value,
                            bbox=bbox_value,
                        )
                    )
                continue

            # Backwards-compatibility path for older PaddleOCR versions that returned
            # nested lists of (bbox, (text, confidence)) tuples.
            for block in item or []:
                for bbox, (text, confidence) in block:
                    bbox_value = bbox.tolist() if hasattr(bbox, "tolist") else bbox
                    ocr_text.append(
                        OCRText(
                            text=text,
                            confidence=float(confidence),
                            language=lang_value,
                            bbox=bbox_value,
                        )
                    )

        return ocr_text
