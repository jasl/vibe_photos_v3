# Phase 1 Implementation Plan — Coding AI Checklist

## Timeline Snapshot
| Days | Focus |
|------|-------|
| 1–3 | Environment bootstrap, directory scaffolding, SQLite schema. |
| 4–7 | Detector + OCR wrappers, batch processor, caching. |
| 8–10 | FastAPI endpoints, CLI commands, Streamlit MVP. |
| 11–14 | Tests, benchmarks, documentation updates. |

## Workstreams
1. **Environment (ENV-001…ENV-005)**
   - Generate structure via `./init_project.sh` (wraps the quick-start checklist).
   - Pin dependencies in `pyproject.toml` and `uv.lock`.
   - Sync models (`blueprints/phase1/download_models.py`) and config templates.
2. **Perception (DET-001…OCR-001)**
   - Build SigLIP + BLIP wrappers with caching and candidate labels.
   - Implement PaddleOCR batch service with GPU/CPU autodetect.
3. **Processing & Persistence (DET-004, DB-001/002)**
   - Batch orchestrator merging perception results, dedupe via `phash`.
   - SQLite repositories with Alembic migrations.
4. **Interfaces (API-001…CLI-001, UI-001)**
   - FastAPI `/import`, `/search`, `/assets/{id}` endpoints.
   - Typer CLI commands mirroring API functionality.
   - Streamlit dashboard for browsing/searching results.
5. **Quality (TEST-001+, PERF-001)**
   - Unit/integration tests, ingestion benchmark harness, logging verification.

## Deliverables
- `src/core/*.py` modules with tests.
- `src/api/main.py` + route modules.
- `src/cli.py`, `blueprints/phase1/app.py` (Streamlit).
- Updated docs (`AI_IMPLEMENTATION_DETAILS.md`, dataset guides, tracker entries).

Use this checklist to track progress; update `AI_TASK_TRACKER.md` as each workstream completes.
