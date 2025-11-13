# AI Decision Log — Coding AI Edition

Each entry captures a binding choice. Update the table when decisions evolve.

## Architecture
| ID | Priority | Status | Decision | Notes |
|----|----------|--------|----------|-------|
| ARC-001 | 🔴 MUST | ✅ Active | Progressive monolith: single FastAPI app with modular boundaries; evolve to services only when PostgreSQL/pgvector + Celery become mandatory. | Keep Phase 1 in-process; introduce workers in Phase 2; split services in Phase Final. |
| ARC-002 | 🔴 MUST | ✅ Active | Offline-first ingestion; search endpoints remain realtime. | Batch imports via CLI/API tasks; queue support planned for Phase Final. |

## Technology Stack
| ID | Priority | Status | Decision | Enforcement |
|----|----------|--------|----------|-------------|
| TECH-001 | 🔴 MUST | ✅ Active | Python 3.12 managed exclusively with `uv`. | No `pip`/`conda`/`poetry`; update `uv.lock` on dependency changes. |
| TECH-002 | 🔴 MUST | ✅ Active | FastAPI for all HTTP APIs with async endpoints and Pydantic schemas. | Build app factory `create_app()`, centralize dependencies via DI. |
| TECH-003 | 🟡 SHOULD | ✅ Active | SQLite for MVP, migrate to PostgreSQL + pgvector when scale triggers (data >10 GB or concurrent users >50). | Maintain Alembic migrations even on SQLite to ease future transition. |

## Models & AI
| ID | Priority | Status | Decision | Notes |
|----|----------|--------|----------|-------|
| MODEL-001 | 🔴 MUST | ✅ Active | SigLIP (`google/siglip2-base-patch16-224`) + BLIP (`Salesforce/blip-image-captioning-base`) as primary perception stack. | Load once per process, share embeddings cache. |
| MODEL-002 | 🔴 MUST | ✅ Active | PaddleOCR (chinese + english) for text extraction. | Provide batching + caching. |
| MODEL-003 | 🔴 MUST | ✅ Active | Avoid RTMDet due to conflicting deps; rely on SigLIP+BLIP combos until alternative vetted. | Document future experiments in blueprint research. |

## Data & Search
| ID | Priority | Status | Decision | Notes |
|----|----------|--------|----------|-------|
| DATA-001 | 🔴 MUST | ✅ Active | Store assets + metadata in SQLite with FTS5; keep schema upgrade path to PostgreSQL. | Ensure migrations scriptable. |
| SEARCH-001 | 🟡 SHOULD | ✅ Active | Implement hybrid search (metadata + embedding) by Phase 2. | Document scoring function; allow fallback to metadata-only search. |

## Process
| ID | Priority | Status | Decision | Notes |
|----|----------|--------|----------|-------|
| PROC-001 | 🔴 MUST | ✅ Active | TDD/coverage ≥80% for `src/core` and `src/api`. | CI gate once pipelines exist. |
| PROC-002 | 🔴 MUST | ✅ Active | All docs updated alongside code; tracker statuses maintained. | Enforced via `FINAL_CHECKLIST.md`. |

Append new rows instead of editing history; mark superseded entries and move detailed write-ups to `archives/` as needed.
