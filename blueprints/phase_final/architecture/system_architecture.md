# Phase Final System Architecture — Coding AI Summary

## Layered View
1. **Interface Layer** — Streamlit web app, FastAPI endpoints, CLI automations.
2. **Service Layer** — Detection/OCR, annotation assistance, search/ranking, learning engine, task orchestration via Celery.
3. **Persistence Layer** — PostgreSQL + pgvector, Redis cache/queue, object storage for originals & thumbnails, model registry.

## Key Flows
- **Ingestion:** API/CLI enqueue jobs → Celery workers generate thumbnails, run SigLIP/BLIP, extract OCR, compute embeddings, persist results, and notify clients.
- **Search:** FastAPI aggregates metadata filters, FTS queries, and vector similarity; combines scores with Reciprocal Rank Fusion before returning annotated results.
- **Learning:** User corrections trigger prototype updates (few-shot) and scheduled retraining; versioned models deployed through controlled rollout.

## Supporting Services
- Prometheus/Grafana for metrics and dashboards.
- Feature flags to toggle experimental pipelines.
- Audit logging for all automated label changes and manual overrides.

Reference this overview with the detailed subsystem documents for implementation specifics.
