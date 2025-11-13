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


def test_preprocess_avoids_collisions_for_duplicate_stems(tmp_path):
    first_dir = tmp_path / "camera_a"
    second_dir = tmp_path / "camera_b"
    first_dir.mkdir()
    second_dir.mkdir()

    first_image = first_dir / "IMG_0001.JPG"
    second_image = second_dir / "IMG_0001.JPG"
    Image.new("RGB", (300, 200), color=(0, 0, 255)).save(first_image)
    Image.new("RGB", (500, 800), color=(0, 255, 0)).save(second_image)

    config = {
        "paths": {
            "processed": str(tmp_path / "processed"),
            "thumbnails": str(tmp_path / "thumbnails"),
            "hash_cache": str(tmp_path / "hash_cache.json"),
        },
        "thumbnail": {"enabled": True, "size": [128, 128]},
        "max_dimension": 1024,
        "deduplication": {"hash_size": 8},
    }

    preprocessor = ImagePreprocessor(config)
    first_result = preprocessor.preprocess(first_image)
    second_result = preprocessor.preprocess(second_image)

    assert Path(first_result.processed_path).name != Path(second_result.processed_path).name
    assert Path(first_result.thumbnail_path).name != Path(second_result.thumbnail_path).name
