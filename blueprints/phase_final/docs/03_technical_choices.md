# Phase Final Technical Choices — Coding AI Digest

## Guiding Principles
- Prefer pragmatic, well-supported tools over experimental options.
- Maintain CPU compatibility; leverage GPU acceleration opportunistically.
- Design for staged rollout: Phase 1 (SQLite), Phase 2 (hybrid), Phase Final (PostgreSQL + pgvector).

## Stack Decisions
| Layer | Choice | Notes |
|-------|--------|-------|
| API | FastAPI | Async, typed, OpenAPI support. |
| UI | Streamlit | Single UI surface from MVP through production; revisit after stabilization if needed. |
| Task Queue | Celery + Redis | Handles ingestion/analysis jobs and scheduled maintenance. |
| Database | PostgreSQL 16 + pgvector | Unified store for metadata and embeddings. |
| Cache | Redis | Session + search result caching. |
| Monitoring | Prometheus + Grafana | Metrics + dashboards. |

## Model Strategy
- **Classification/Embeddings:** SigLIP-base-i18n for general use, SigLIP-large optional for higher accuracy.
- **Captioning:** BLIP-base; upgrade to BLIP-large when GPU resources permit.
- **OCR:** PaddleOCR (Chinese/English support).
- **Few-shot:** DINOv2 embeddings with prototype-based classifiers.

## Search & Storage
- Hybrid search: PostgreSQL FTS for text + pgvector for embeddings; apply Reciprocal Rank Fusion for final ranking.
- File storage: local filesystem or S3-compatible bucket; maintain versioned cache for processed assets and thumbnails.

## Environments
- **Dev/Test:** Docker-compose stack (PostgreSQL, Redis, MinIO). GPU optional.
- **Prod:** Kubernetes or VM-based deployment with autoscaled Celery workers and monitoring stack.

Refer to `AI_DECISION_RECORD.md` for status updates when choices evolve.
