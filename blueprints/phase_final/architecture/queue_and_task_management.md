# Queue & Task Management — Coding AI Summary

## Selected Stack
- **Celery** as the task orchestrator with **Redis** serving as broker and result backend.
- Queues: `high_priority` (single image/UX-critical), `default` (batch ingestion), `learning` (model updates), `maintenance` (reindex, cleanup).

## Patterns
- Use `chain` for sequential workflows (validate → detect → persist → notify).
- Use `group`/`chord` for parallel batch processing with aggregation steps.
- Enforce retries with exponential backoff for transient failures; capture metrics via Prometheus exporters.

## Operational Notes
- Limit concurrency per worker type (GPU-bound vs CPU-bound tasks).
- Implement Redis-based locks for mutually exclusive jobs (few-shot training, reindexing).
- Surface job status via WebSocket or polling endpoints so UI/CLI can report progress.

Document any queue topology changes in `AI_DECISION_RECORD.md` and update monitoring dashboards accordingly.
