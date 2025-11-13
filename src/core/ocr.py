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
        # PaddleOCR>=3.3 dropped the legacy `show_log` and `use_angle_cls` constructor
        # arguments in favor of `use_textline_orientation`. We keep the configuration
        # key name for backwards compatibility and map it to the new flag here.
        self._ocr = PaddleOCR(lang=lang, use_textline_orientation=use_angle_cls)
        return self._ocr

    def extract_text(self, image_path: Path) -> List[OCRText]:
        """Run OCR on the given image."""
        engine = self._load_engine()
        if engine is None:
            return []

        languages = self.config.get("languages", ["ch"])
        lang_value = ",".join(languages) if isinstance(languages, list) else str(languages)
        # The `cls`/`use_angle_cls` parameter used in older PaddleOCR versions has been
        # replaced by `use_textline_orientation` on the engine itself, so we no longer
        # pass it per-call and instead rely on the engine configuration.
        result = engine.ocr(str(image_path))
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
