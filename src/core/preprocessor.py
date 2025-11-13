"""Image preprocessing utilities used before running detectors."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from PIL import Image, ImageOps


@dataclass(slots=True)
class PreprocessedImage:
    """Metadata produced by the preprocessing pipeline."""

    original_path: Path
    processed_path: Path
    thumbnail_path: Path | None
    phash: str
    file_size: int
    width: int
    height: int


class ImagePreprocessor:
    """Normalize images, generate thumbnails, and compute perceptual hashes."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        paths = config.get("paths", {})
        self.processed_dir = Path(paths.get("processed", "cache/images/processed"))
        self.thumbnail_dir = Path(paths.get("thumbnails", "cache/images/thumbnails"))
        self.hash_cache_path = Path(paths.get("hash_cache", "cache/hashes/phash_cache.json"))
        self.max_dimension = int(config.get("max_dimension", 4096))

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)
        self.hash_cache_path.parent.mkdir(parents=True, exist_ok=True)

    def preprocess(self, image_path: Path) -> PreprocessedImage:
        """Normalize the image and return metadata used downstream."""
        image = Image.open(image_path)
        image = image.convert("RGB")
        original_size = image.size
        image = self._resize_if_needed(image)

        output_basename = self._build_output_basename(image_path)
        processed_path = self.processed_dir / f"{output_basename}.jpeg"
        image.save(processed_path, format="JPEG", quality=90, optimize=True)

        thumbnail_path = None
        thumbnail_config = self.config.get("thumbnail", {})
        if thumbnail_config.get("enabled", True):
            thumbnail_size = tuple(thumbnail_config.get("size", [512, 512]))
            thumbnail_image = image.copy()
            thumbnail_image.thumbnail(thumbnail_size)
            thumbnail_path = self.thumbnail_dir / f"{output_basename}_thumb.jpeg"
            thumbnail_image.save(
                thumbnail_path,
                format=thumbnail_config.get("format", "JPEG"),
                quality=thumbnail_config.get("quality", 85),
                optimize=True,
            )

        phash = self._compute_phash(image)
        stat = processed_path.stat()

        return PreprocessedImage(
            original_path=image_path.resolve(),
            processed_path=processed_path.resolve(),
            thumbnail_path=thumbnail_path.resolve() if thumbnail_path else None,
            phash=phash,
            file_size=stat.st_size,
            width=original_size[0],
            height=original_size[1],
        )

    def _resize_if_needed(self, image: Image.Image) -> Image.Image:
        """Resize oversized images while preserving aspect ratio."""
        width, height = image.size
        max_dimension = max(width, height)
        if max_dimension <= self.max_dimension:
            return image

        scale = self.max_dimension / max_dimension
        new_size = (int(width * scale), int(height * scale))
        return image.resize(new_size, Image.LANCZOS)

    def _compute_phash(self, image: Image.Image, hash_size: int | None = None) -> str:
        """Compute a basic perceptual hash for duplicate detection."""
        hash_size = hash_size or self.config.get("deduplication", {}).get("hash_size", 8)
        grayscale = ImageOps.grayscale(image)
        resized = grayscale.resize((hash_size, hash_size), Image.LANCZOS)
        pixels = list(resized.getdata())
        avg = sum(pixels) / len(pixels)
        bits = "".join("1" if pixel > avg else "0" for pixel in pixels)
        width = 4
        return "".join(f"{int(bits[i : i + width], 2):x}" for i in range(0, len(bits), width))

    def _build_output_basename(self, image_path: Path) -> str:
        """Return a deterministic, collision-resistant basename for derived artifacts."""
        resolved = image_path.resolve()
        digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
        return f"{image_path.stem}_{digest}"
