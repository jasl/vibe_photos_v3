# Vibe Photos Task Tracker — Coding AI Backlog

Use this tracker as the single source of truth for what each coding AI should execute next. Update statuses and notes as you work.

## Legend
- **Status:** ⬜ Not started · 🟨 In progress · ✅ Done · 🔄 Rework · ❌ Dropped
- **Priority:** 🔴 P0 (blocker), 🟠 P1 (critical), 🟡 P2 (important), 🟢 P3 (nice-to-have)
- **Notes column:** Log blockers, decisions, links to commits/PRs, or context for the next assignee.

## Phase 1 — MVP Delivery (Active)
| Status | Priority | ID | Description | Dependencies | Expected Output | Notes |
|--------|----------|----|-------------|--------------|-----------------|-------|
| ⬜ | 🔴 P0 | ENV-001 | Materialize repository structure & init script. | — | `init_project.sh`, directory tree | |
| ⬜ | 🔴 P0 | ENV-002 | Finalize `pyproject.toml` + lockfile. | ENV-001 | Updated `pyproject.toml`, `uv.lock` | |
| ⬜ | 🔴 P0 | ENV-003 | Bootstrap `uv` environment and core deps install. | ENV-002 | Reproducible venv instructions | |
| ⬜ | 🟠 P1 | ENV-004 | Model cache bootstrap (SigLIP, BLIP, PaddleOCR). | ENV-003 | Cached models under `models/` | |
| ⬜ | 🟠 P1 | ENV-005 | Configuration templates (`config/settings.yaml`). | ENV-001 | Template + documentation | |
| ⬜ | 🔴 P0 | DET-001 | Implement SigLIP loader abstraction. | ENV-004 | `src/models/siglip.py` | |
| ⬜ | 🔴 P0 | DET-002 | Implement BLIP loader abstraction. | ENV-004 | `src/models/blip.py` | |
| ⬜ | 🔴 P0 | DET-003 | Compose unified detector (labels + captions). | DET-001, DET-002 | `src/core/detector.py` | |
| ⬜ | 🟠 P1 | DET-004 | Batch processor orchestrating detector + OCR. | DET-003, OCR-001 | `src/core/processor.py` | |
| ⬜ | 🔴 P0 | OCR-001 | PaddleOCR service with caching + batching. | ENV-004 | `src/core/ocr.py` | |
| ⬜ | 🔴 P0 | DB-001 | Define SQLite schema & migrations. | ENV-001 | `src/core/database.py`, migrations | |
| ⬜ | 🔴 P0 | DB-002 | Persistence services (CRUD + search helpers). | DB-001 | Repository classes/tests | |
| ⬜ | 🔴 P0 | API-001 | FastAPI app factory + health endpoint. | DB-002 | `src/api/main.py` | |
| ⬜ | 🔴 P0 | API-002 | `/import` ingestion endpoint (async upload). | API-001, DET-004 | `routes/import.py` | |
| ⬜ | 🔴 P0 | API-003 | `/search` endpoint returning ranked assets. | API-001, DB-002 | `routes/search.py` | |
| ⬜ | 🟠 P1 | CLI-001 | Typer CLI commands (`ingest`, `search`). | DET-004, DB-002 | `src/cli.py` | |
| ⬜ | 🟠 P1 | UI-001 | Streamlit MVP dashboard hooking core services. | DET-004, DB-002 | `blueprints/phase1/app.py` | |
| ⬜ | 🟠 P1 | TEST-001 | Unit tests for detector/ocr/database/search. | DET-003, DB-002 | `tests/unit/...` | |
| ⬜ | 🟠 P1 | TEST-002 | API + CLI integration tests. | API-003, CLI-001 | `tests/integration/...` | |
| ⬜ | 🟡 P2 | PERF-001 | Benchmark ingestion throughput (≥10 img/s). | DET-004 | `tests/perf/test_ingestion_speed.py` | |
| ⬜ | 🟡 P2 | DOC-001 | Update docs + diagrams after MVP stabilization. | INT-001 | Updated manuals | |

## Phase 2 — Semantic Search Upgrade (Planned)
| Status | Priority | ID | Description | Dependencies | Output |
|--------|----------|----|-------------|--------------|--------|
| ⬜ | 🔴 P0 | EMB-001 | Image embedding pipeline (SigLIP features). | Phase 1 complete | `src/models/embedder.py` |
| ⬜ | 🔴 P0 | EMB-002 | Text embedding pipeline for captions/OCR. | EMB-001 | Unified embedding interface |
| ⬜ | 🟠 P1 | SRCH-001 | Vector index management (FAISS/pgvector). | EMB-001 | Vector store adapter |
| ⬜ | 🟠 P1 | SRCH-002 | Hybrid search ranking (vector + metadata). | SRCH-001 | Ranking module |
| ⬜ | 🟡 P2 | UI-002 | Advanced filtering UI for semantic search. | SRCH-002 | Updated Streamlit/Gradio views |

## Phase Final — Production Platform (Planned)
| Status | Priority | ID | Description | Dependencies | Output |
|--------|----------|----|-------------|--------------|--------|
| ⬜ | 🔴 P0 | INF-001 | Provision PostgreSQL + pgvector infra scripts. | Phase 2 | Terraform/Ansible templates |
| ⬜ | 🔴 P0 | INF-002 | Celery + Redis task fabric. | INF-001 | Worker deployment configs |
| ⬜ | 🟠 P1 | OPS-001 | Observability stack (Prometheus/Grafana). | INF-002 | Monitoring dashboards |
| ⬜ | 🟡 P2 | OPS-002 | CI/CD automation + smoke tests. | INF-001 | Pipeline definitions |

## Usage Notes
- Update this file immediately after changing a task’s status.
- Link related commits or PR numbers in the Notes column.
- When closing a task, ensure deliverables meet the Definition of Done in `AI_DEVELOPMENT_GUIDE.md`.
