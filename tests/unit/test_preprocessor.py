from pathlib import Path

from PIL import Image

from src.core.preprocessor import ImagePreprocessor


def test_preprocess_generates_artifacts(tmp_path):
    img_path = tmp_path / "sample.png"
    Image.new("RGB", (600, 400), color=(255, 0, 0)).save(img_path)

    config = {
        "paths": {
            "processed": str(tmp_path / "processed"),
            "thumbnails": str(tmp_path / "thumbnails"),
            "hash_cache": str(tmp_path / "hash_cache.json"),
        },
        "thumbnail": {"enabled": True, "size": [128, 128]},
        "max_dimension": 256,
        "deduplication": {"hash_size": 8},
    }

    preprocessor = ImagePreprocessor(config)
    result = preprocessor.preprocess(img_path)

    assert Path(result.processed_path).exists()
    assert Path(result.thumbnail_path).exists()
    assert len(result.phash) > 0
    assert result.width == 600
    assert result.height == 400
