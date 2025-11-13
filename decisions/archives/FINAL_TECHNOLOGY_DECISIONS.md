# Archive: Original Technology Stack Plan

This document preserves the initial tech stack proposal before we adopted the current progressive roadmap.

## Phase 1 (original proposal)
- Python 3.12 with `uv`
- FastAPI + Streamlit
- SQLite with full-text search
- SigLIP + BLIP for perception, PaddleOCR for text
- PyTorch 2.9.1 as the DL backbone

## Phase Final (planned at the time)
- PostgreSQL + pgvector
- Redis cache, Celery workers
- Prometheus + Grafana monitoring

The active decisions have since moved to `AI_DECISION_RECORD.md`; keep this file for historical traceability only.
