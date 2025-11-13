# Phase Final Solution Design — Coding AI Summary

## Architecture Overview
- **Presentation:** Gradio web app + FastAPI + CLI/automation hooks.
- **Services:** Detection (SigLIP/BLIP), OCR (PaddleOCR), Search (hybrid metadata + embeddings), Annotation/Learning, Task orchestration (Celery workers).
- **Persistence:** PostgreSQL + pgvector, object storage (local or S3-compatible), Redis cache, versioned model registry.

## Core Workflows
1. **Ingestion** — Streamlined pipeline: thumbnail → baseline classification → asynchronous deep analysis (object labels, OCR, embeddings). Results stored in PostgreSQL with cached artifacts on disk/object store.
2. **Hybrid Search** — Combine structured filters (tags, metadata, time), full-text indexes, and vector similarity. Provide explainable results (source of match, confidence).
3. **Annotation Assistance** — Suggest labels using AI predictions, similar asset lookup, OCR hints, and user history. Support batch acceptance and quick corrections.
4. **Few-Shot Personalization** — Register user-provided exemplars; build prototypes using DINOv2 embeddings; integrate into ranking pipeline with configurable thresholds.

## Component Responsibilities
- **Celery Workers:** Handle ingestion batches, embedding recalculations, scheduled reindexing.
- **Redis:** Queue broker + cache for hot search results and annotation suggestions.
- **Monitoring:** Prometheus metrics exposed by FastAPI/Celery; Grafana dashboards for throughput, latency, job failures.

## Data Model Highlights
- `assets` table with metadata, status, capture time, and normalized storage paths.
- `labels`, `captions`, `ocr_blocks`, `embeddings` linked via foreign keys.
- `jobs` and `annotations` tables to track workflow state and user interventions.

Use this design summary with `03_technical_choices.md` and `04_implementation_guide.md` when planning Phase Final implementation.
