"""PaddleOCR wrapper used during ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


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

    def _load_engine(self):
        if not self.enabled:
            return None

        if self._ocr is not None:
            return self._ocr

        try:
            from paddleocr import PaddleOCR
        except ImportError as error:  # pragma: no cover - executed only without deps
            raise RuntimeError("paddleocr is required for OCR support") from error

        use_angle_cls = self.config.get("use_angle_cls", True)
        languages = self.config.get("languages", ["ch"])
        lang = languages[0] if isinstance(languages, list) else languages
        self._ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, show_log=False)
        return self._ocr

    def extract_text(self, image_path: Path) -> List[OCRText]:
        """Run OCR on the given image."""
        engine = self._load_engine()
        if engine is None:
            return []

        lang = self.config.get("languages", ["ch"])
        lang_value = ",".join(lang) if isinstance(lang, list) else str(lang)
        result = engine.ocr(str(image_path), cls=self.config.get("use_angle_cls", True))
        ocr_text: List[OCRText] = []

        for block in result or []:
            for bbox, (text, confidence) in block:
                ocr_text.append(
                    OCRText(
                        text=text,
                        confidence=float(confidence),
                        language=lang_value,
                        bbox=bbox,
                    )
                )

        return ocr_text
