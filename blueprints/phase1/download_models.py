#!/usr/bin/env python3
"""Model download utility for Vibe Photos Phase 1."""

from __future__ import annotations

import sys
import tarfile
import zipfile
from inspect import signature
import shutil
from pathlib import Path
from typing import Dict, Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(exist_ok=True)

TAR_SUPPORTS_FILTER = "filter" in signature(tarfile.TarFile.extractall).parameters

MODELS_CONFIG: Dict[str, Dict] = {
    "models_info": {
        "name": "Vibe Photos Phase 1 models",
        "description": "Includes SigLIP, BLIP, and PaddleOCR assets",
        "note": "SigLIP and BLIP artifacts are downloaded automatically via transformers.",
    },
    "auto_download": {
        "siglip": {
            "name": "google/siglip2-base-patch16-224",
            "description": "Multilingual image classification model",
            "size": "~400MB",
            "source": "Hugging Face",
            "note": "Fetched automatically by transformers",
        },
        "blip": {
            "name": "Salesforce/blip-image-captioning-base",
            "description": "Image captioning model",
            "size": "~990MB",
            "source": "Hugging Face",
            "note": "Fetched automatically by transformers",
        },
    },
    "paddleocr": {
        "name": "PaddleOCR",
        "description": "Chinese/English OCR models",
        "files": {
            "det_model": {
                "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar",
                "size": "~4.9MB",
                "path": "paddleocr/ch_PP-OCRv4_det_infer.tar",
                "extract": True,
            },
            "rec_model": {
                "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar",
                "size": "~10MB",
                "path": "paddleocr/ch_PP-OCRv4_rec_infer.tar",
                "extract": True,
            },
            "cls_model": {
                "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
                "size": "~2.2MB",
                "path": "paddleocr/ch_ppocr_mobile_v2.0_cls_infer.tar",
                "extract": True,
            },
        },
    },
}


class ModelDownloader:
    """Download helper for PaddleOCR assets."""

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, dest_path: Path, expected_size: Optional[str] = None) -> bool:
        """Download a remote file with basic progress logging."""
        if dest_path.exists():
            print(f"✓ File already exists: {dest_path.name}")
            return True

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading: {dest_path.name}")
        if expected_size:
            print(f"  Expected size: {expected_size}")

        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(dest_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        progress = downloaded / total_size * 100
                        print(f"  Progress: {progress:.1f}%", end="\r")

            print(f"\n✓ Download complete: {dest_path.name}")
            return True

        except Exception as error:  # noqa: BLE001
            print(f"\n✗ Download failed: {error}")
            if dest_path.exists():
                dest_path.unlink()
            return False

    def extract_archive(self, archive_path: Path) -> bool:
        """Extract tar or zip archives into the model directory."""
        extract_dir = archive_path.parent

        try:
            if archive_path.suffix == ".tar":
                with tarfile.open(archive_path, "r") as tar:
                    if TAR_SUPPORTS_FILTER:
                        tar.extractall(extract_dir, filter="data")
                    else:
                        tar.extractall(extract_dir)
            elif archive_path.suffix == ".zip":
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
            else:
                print(f"Unsupported archive format: {archive_path.suffix}")
                return False

            print(f"✓ Extracted: {archive_path.name}")
            return True

        except Exception as error:  # noqa: BLE001
            print(f"✗ Extraction failed: {error}")
            self._cleanup_corrupted_archive(archive_path)
            return False

    def _cleanup_corrupted_archive(self, archive_path: Path) -> None:
        """Remove partially downloaded archives and extracted folders."""
        try:
            if archive_path.exists():
                archive_path.unlink()

            candidate_dir = archive_path.with_suffix("")
            if candidate_dir.exists() and candidate_dir.is_dir():
                shutil.rmtree(candidate_dir)
        except Exception as cleanup_error:  # noqa: BLE001
            print(f"  ⚠️ Cleanup after extraction failure failed: {cleanup_error}")

    def download_paddleocr_models(self) -> bool:
        """Download and extract PaddleOCR artifacts."""
        print("\n" + "=" * 50)
        print("Downloading PaddleOCR assets")
        print("=" * 50)

        paddleocr_config = MODELS_CONFIG["paddleocr"]
        success = True

        for file_info in paddleocr_config["files"].values():
            dest_path = self.models_dir / file_info["path"]

            if not self.download_file(file_info["url"], dest_path, file_info.get("size")):
                success = False
                continue

            if file_info.get("extract") and not self.extract_archive(dest_path):
                success = False

        return success

    def show_auto_download_info(self) -> None:
        """Display models managed automatically by transformers."""
        print("\n" + "=" * 50)
        print("Models fetched automatically")
        print("=" * 50)

        for model_info in MODELS_CONFIG["auto_download"].values():
            print(f"\n{model_info['name']}")
            print(f"  Description: {model_info['description']}")
            print(f"  Size: {model_info['size']}")
            print(f"  Source: {model_info['source']}")
            print(f"  Notes: {model_info['note']}")

        print("\nThese models will be cached under ~/.cache/huggingface/hub on first use.")

    def download_all(self) -> bool:
        """Run the full download pipeline."""
        print("\n" + "=" * 50)
        print("Vibe Photos Phase 1 model downloader")
        print("=" * 50)

        self.show_auto_download_info()

        if not self.download_paddleocr_models():
            print("\n⚠️ Some models failed to download. Check your network connection and retry.")
            return False

        print("\n" + "=" * 50)
        print("✅ All manual downloads are ready!")
        print("=" * 50)
        print("\nTip:")
        print("1. SigLIP and BLIP models download on first run via transformers.")
        print("2. To prefetch them, execute the following Python snippet:")
        print("\n```python")
        print("from transformers import AutoModel, AutoProcessor, BlipForConditionalGeneration, BlipProcessor")
        print("AutoModel.from_pretrained('google/siglip2-base-patch16-224')")
        print("AutoProcessor.from_pretrained('google/siglip2-base-patch16-224')")
        print("BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')")
        print("BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')")
        print("```")

        return True


def main() -> None:
    """CLI entrypoint."""
    downloader = ModelDownloader()
    sys.exit(0 if downloader.download_all() else 1)


if __name__ == "__main__":
    main()
