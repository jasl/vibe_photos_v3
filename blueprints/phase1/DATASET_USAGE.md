# Phase 1 Dataset Usage — Coding AI Instructions

## Preparing the Dataset
1. Copy curated test images into `samples/` (hierarchical folders allowed). Non-image files will be skipped automatically.
2. Configure `config/settings.yaml` with:
```yaml
dataset:
  directory: "samples"
  incremental: true
  supported_formats: [.jpg, .jpeg, .png, .heic, .webp]
```
3. Run ingestion via `uv run python process_dataset.py` (legacy wrapper remains under `blueprints/phase1/` for reference). The script handles dedupe, normalization, thumbnail creation, detection, OCR, and persistence.

## Incremental Workflow
- Add new images to `samples/` and rerun the script; processed files are tracked via perceptual hash.
- Cache directories (`cache/images`, `cache/detections`, `cache/ocr`) store reusable artifacts keyed by `phash`.

## Directory Contract
```
samples/          # raw read-only assets
data/             # SQLite DB + state snapshots
cache/
  images/{processed, thumbnails}
  detections/
  ocr/
  hashes/phash_cache.json
```

## Configuration Tips
- Adjust dedupe sensitivity in `preprocessing.deduplication.threshold` (lower = stricter).
- Keep the perceptual hash index at `preprocessing.paths.hash_cache`; delete the file to force a full rebuild.
- Control batch size and worker count under `batch_processing` to match hardware.
- Export `TRANSFORMERS_CACHE` and `PADDLEOCR_HOME` to point at `models/` for faster runs.

Follow this guide whenever you prepare or refresh the Phase 1 dataset.
