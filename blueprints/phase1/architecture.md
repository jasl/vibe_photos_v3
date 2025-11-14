# Phase 1 Architecture Snapshot — Coding AI

## Principles
- Keep the system as a single-process monolith for ease of iteration.
- Run ingestion offline (CLI or background job) while serving read-only APIs/UI.
- Use SQLite + local file system; design schema to transition smoothly to PostgreSQL later.

## Layers
1. **Interfaces** — Streamlit dashboard (read-only), Typer CLI (`process_dataset`, `search`), long-lived ingestion service, FastAPI endpoints (`/import`, `/search`, `/assets`).
2. **Core Services** — File-backed task queue, Detector (SigLIP+BLIP), OCR (PaddleOCR), Processor (batch orchestration), Searcher (SQLite FTS + metadata filters).
3. **Persistence** — SQLite database (`images`, `labels`, `captions`, `ocr_blocks`, `embeddings`, `jobs`) plus cached artifacts and queue files under `cache/`.
4. **Storage** — Raw assets in `samples/`, normalized copies + thumbnails in `cache/images/`, model weights in `models/`.

## Data Flow
```
CLI/file discovery → FileTaskQueue (cache/ingestion_queue) → Ingestion service (warm BatchProcessor) → Cache artifacts → SQLite persist (optional)
FastAPI `/search` → Searcher → SQLite FTS + metadata → Response
Streamlit UI → FastAPI → same search pipeline
```

## Schema Outline
- `images(id, filename, original_path, processed_path, thumbnail_path, phash, file_size, width, height, captured_at, status, error_message, embedding_json)`
- `labels(image_id, label, confidence)`
- `captions(image_id, text, source)`
- `ocr_blocks(image_id, text, language, bbox)`
- `jobs(id, job_type, started_at, finished_at, status, metadata_json)`

## Extension Hooks
- Add vector index tables once embeddings are active (Phase 2).
- Introduce job queue adapters (Celery/Redis) while keeping same processor API.

Treat this document as the high-level picture; implementation contracts live in `implementation.md`.
