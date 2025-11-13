# Vibe Photos Development Playbook — For Coding AI Only

This playbook transforms the product contract into concrete work packages. Follow it sequentially; it encodes the assumptions that every future coding AI will rely on.

## 1. System Definition
- **Product name:** Vibe Photos
- **Primary users:** Chinese content creators managing large personal photo libraries.
- **Core promise:** Fast photo understanding (objects + captions + OCR) with semantic search and grouping.
- **Explicit non-goals:** Image editing, cloud sync, social features, and fully autonomous workflows.

## 2. Engineering Constraints
| Domain | Rule |
|--------|------|
| Runtime | Python 3.12 only, managed with `uv`. |
| Architecture | Favor functional decomposition; introduce classes only when stateful orchestration is mandatory. |
| API | FastAPI with async endpoints, streaming responses where practical. |
| Error handling | Guard clauses, explicit typed results (`Result`-style or `Either` patterns acceptable). |
| Language | All UI text, source code, comments, configuration, and documented code snippets must be written in English. |
| Observability | Structured logging (JSON-friendly), correlation IDs propagated across async tasks. |

## 3. Program Phases
1. **Phase 1 — MVP Validation (active)**
   - Deliver detector, captioning, OCR, SQLite-based search, basic Streamlit UI, and CLI.
   - Throughput target: ≥10 images/sec on CPU for 512px JPEGs.
2. **Phase 2 — Semantic Search Upgrade (pending)**
   - Introduce hybrid vector search, object detection enhancements, improved annotations.
3. **Phase Final — Production Platform (planned)**
   - PostgreSQL + pgvector, Celery/Redis task fabric, Streamlit hardening, deployment automation.

## 4. Reference Architecture (Phase 1)
```
src/
├── core/
│   ├── detector.py        # SigLIP+BLIP fusion, optional candidate labels
│   ├── ocr.py             # PaddleOCR wrapper with batching support
│   ├── processor.py       # Bulk ingestion, dedupe, metadata extraction
│   ├── searcher.py        # Embedding cache + cosine similarity queries (SQLite)
│   └── database.py        # Persistence layer (SQLite + SQLAlchemy models)
├── models/
│   ├── siglip.py
│   ├── blip.py
│   └── paddleocr.py
├── api/
│   ├── main.py            # FastAPI app factory
│   ├── routes/            # Import/search/annotation endpoints
│   └── schemas.py         # Pydantic models & response envelopes
├── utils/
│   ├── image_ops.py
│   ├── cache.py
│   └── logging.py
└── cli.py                 # Typer-based command surface
```
Tests live in `tests/` mirroring the module structure with fixtures under `tests/fixtures/`.

## 5. Phase 1 Task Grid
| Track | Deliverable | Notes |
|-------|-------------|-------|
| Environment | `uv` environment with pinned dependencies, dataset download scripts, model cache bootstrap. | Scripts live in `blueprints/phase1`. |
| Perception | Unified detector returning labels + captions + OCR text with confidence scores. | Ensure deterministic ordering for downstream search. |
| Persistence | SQLite schema for assets, tags, OCR text, embeddings, and job history. | Provide migration utilities even if simple. |
| Interfaces | FastAPI routes (`/import`, `/search`, `/assets/{id}`), Streamlit MVP dashboard, CLI commands for batch ingestion. | Document each command in CLI help strings. |
| Quality | Unit tests ≥80% coverage on core logic, smoke tests for API, benchmark harness for ingestion throughput. | Capture metrics in `log/`. |

## 6. Execution Sequence for Each Module
1. Draft acceptance criteria from `AI_IMPLEMENTATION_DETAILS.md`.
2. Write failing tests (unit + integration where needed).
3. Implement minimal code to pass tests while respecting coding standards.
4. Instrument logging and metrics hooks.
5. Update relevant documentation (blueprints, tracker, changelogs).

## 7. Tooling Commands
```bash
# Bootstrap environment
uv venv --python 3.12
source .venv/bin/activate
uv sync

# Run formatters / linters
task fmt   # optional make task if defined; otherwise run ruff/black manually
uv run ruff check src tests
uv run pytest -q

# Launch services for manual verification
uv run uvicorn src.api.main:app --reload
uv run streamlit run blueprints/phase1/app.py
```

## 8. Performance & Reliability Targets
- **Throughput:** ≥10 images/sec CPU baseline; document hardware used during validation.
- **Latency:** Search endpoint P95 < 1.5s for 5k images dataset.
- **Resilience:** Handle missing/corrupted files gracefully, emit actionable error logs.
- **Caching:** Reuse embeddings across runs, share caches via `cache/` directory guidelines.

## 9. Definition of Done
- Tests, linters, and benchmarks passing.
- Documentation updated (including this guide if scopes change).
- Task entry in `AI_TASK_TRACKER.md` moved to ✅ with relevant notes/links.
- PR summary captures affected modules and verification evidence.

Follow this playbook rigorously; divergence introduces drift for every future coding AI inheriting the project.
