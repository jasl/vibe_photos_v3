# Technical Decisions Summary — Coding AI Digest

This document aggregates the active decisions from `AI_DECISION_RECORD.md` for quick reference. For rationale and history, consult the detailed log or archives.

## Phase 1 (Active)
| Domain | Decision | Status |
|--------|----------|--------|
| Architecture | Progressive monolith with offline ingestion; search endpoints remain realtime. | ✅ Active |
| Language & Tooling | Python 3.12 managed exclusively by `uv`. | ✅ Active |
| API Layer | FastAPI (async) + Pydantic schemas. | ✅ Active |
| Data Store | SQLite + FTS5 with Alembic migrations prepped for PostgreSQL. | ✅ Active |
| Models | SigLIP + BLIP + PaddleOCR as perception stack; avoid RTMDet. | ✅ Active |
| UI | Streamlit MVP + Typer CLI. | ✅ Active |
| Quality | Unit/integration/perf tests with ≥80% coverage expectation. | ✅ Active |

## Phase 2 (Planned)
| Domain | Decision | Trigger |
|--------|----------|---------|
| Embeddings | Introduce SigLIP-based embedding services and hybrid ranking. | After Phase 1 validation |
| Vector Index | Evaluate FAISS vs pgvector; prepare migration path. | Dataset >10k assets |
| UI | Enhance filters & semantic search UX. | When embedding accuracy validated |

## Phase Final (Planned)
| Domain | Decision | Trigger |
|--------|----------|---------|
| Database | Migrate to PostgreSQL + pgvector. | Production readiness |
| Task Queue | Adopt Celery + Redis for ingestion/offline jobs. | Sustained workload >100 jobs/day |
| Monitoring | Deploy Prometheus/Grafana dashboards. | Prior to public release |
| UI | Transition to Gradio-based production interface. | When backend stabilizes |

## Maintenance Notes
- Update this summary whenever the status of a decision changes.
- Link back to ADR entries for context; never duplicate rationale here.
- Move outdated rows to `archives/` with references to replacements.
