# Vibe Photos Roadmap — Coding AI Execution View

This roadmap converts the product strategy into concrete milestones for coding AIs. Use it to plan work, coordinate transitions between phases, and understand success criteria.

## Phase Overview
| Phase | Duration | Primary Goal | Key Deliverables | Exit Criteria |
|-------|----------|--------------|------------------|---------------|
| Phase 1 — MVP Validation | ~2 weeks | Prove detector + OCR + search workflow on local datasets. | Detector/OCR modules, SQLite persistence, FastAPI endpoints, CLI + Streamlit MVP, ingestion benchmark. | ≥10 images/sec ingestion throughput, working `/search` endpoint, qualitative approval from stakeholders. |
| Phase 2 — Semantic Search Upgrade | ~1 month | Introduce vector embeddings and hybrid search for improved relevance. | Embedding pipelines, vector index adapter, search ranking fusion, advanced filters UI. | Demonstrated semantic search accuracy improvements, documentation of embedding configurations. |
| Phase Final — Production Platform | 2–3 months | Harden system for production deployment. | PostgreSQL + pgvector infra, Celery/Redis tasks, monitoring stack, Streamlit production UI, CI/CD pipelines. | Stable deployment scripts, observability dashboards, automated smoke tests. |

## Phase 1 Backlog Themes
1. **Environment & Tooling** — `uv` setup, dependency pinning, model download automation.
2. **Perception Stack** — SigLIP + BLIP integration, OCR service, caching strategy.
3. **Data Layer** — SQLite schema, repository layer, migrations stub.
4. **Interfaces** — FastAPI endpoints, Typer CLI, Streamlit MVP.
5. **Quality** — Unit/integration/performance tests, logging, metrics baseline.

## Phase 2 Preview
- Embed text + images using SigLIP (and optional BLIP) into vector stores.
- Manage SQLite cosine search in Phase 1 while preparing migration paths to pgvector; implement hybrid ranking (metadata + embeddings).
- Expand UI for semantic filters and similarity exploration.
- Record experiments and performance metrics in `blueprints/phase_final/research/`.

## Phase Final Preview
- Replace SQLite with PostgreSQL + pgvector and production migrations.
- Establish asynchronous processing via Celery workers and Redis queue.
- Harden the Streamlit UI for production, design monitoring/alerting stack, and integrate CI/CD automation.

## Cadence & Checkpoints
- **Weekly sync:** Review `AI_TASK_TRACKER.md`, update statuses, surface blockers.
- **Phase exit review:** Use `FINAL_CHECKLIST.md` to confirm documentation, tests, and metrics are complete before advancing.
- **Decision updates:** Any architectural change must be recorded in `decisions/TECHNICAL_DECISIONS.md` or a new ADR.

## Guiding Principles
- Move sequentially through phases; do not borrow tasks from future phases unless prerequisites are met.
- Timebox experiments and document outcomes in blueprint research archives.
- Keep benchmarks reproducible—record environment details and dataset versions.

Keep this roadmap in sync with reality. When deliverables shift, update the table and communicate via the decision logs so every coding AI remains aligned.
