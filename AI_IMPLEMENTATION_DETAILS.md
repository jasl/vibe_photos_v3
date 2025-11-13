# Implementation Field Manual — Vibe Photos Coding AI

This manual extends the development playbook with executable details. Each section maps directly to a backlog tag (ENV, CORE, API, DATA, QA). When you pick up a task, locate the matching section below and follow the prescribed steps.

## ENV — Environment & Scaffolding
### ENV-001: Repository Skeleton Generator
Use the following script to (re)hydrate the canonical directory structure. Run it only when bootstrapping a clean checkout.

> **Scope clarification**: the `src/` tree referenced throughout this manual is not yet checked into the repository. Creating it is explicitly deferred to the ENV track (e.g. when you execute `ENV-001`). Treat the script below as the authoritative source of truth, and only materialize the skeleton when a backlog item instructs you to do so.
```bash
#!/usr/bin/env bash
set -euo pipefail

mkdir -p src/core/{detector,processor,searcher,learner}
mkdir -p src/models/{siglip,blip,ocr,embedder}
mkdir -p src/data/{database,cache,storage}
mkdir -p src/api/{routes,schemas,middleware}
mkdir -p src/utils/{logger,config,metrics}
mkdir -p src/cli
mkdir -p tests/{unit,integration,fixtures/images}
mkdir -p tests/fixtures/{electronic,food,document,landscape,person}
mkdir -p config data cache/{images/{thumbnails,processed},detections,ocr,embeddings,hashes} models log tmp
find src tests -type d -exec touch {}/__init__.py \;
```

### ENV-002: `pyproject.toml` Baseline
The snippet below reflects the authoritative dependency set and tooling configuration. Sync it before adding new packages to avoid drift.
```toml
[project]
name = "vibe-photos"
version = "1.0.0"
description = "AI-powered photo management system for content creators"
readme = "README.md"
requires-python = "==3.12.*"
license = { text = "AGPL-3.0-or-later" }
authors = [{ name = "Vibe Photos Team" }]

dependencies = [
    "fastapi==0.121.1",
    "uvicorn==0.38.0",
    "streamlit==1.51.0",
    "sqlalchemy==2.0.44",
    "pillow==11.3.0",
    "python-multipart==0.0.20",
    "aiofiles==24.1.0",
    "pydantic==2.11.10",
    "requests==2.32.3",
    "pyyaml==6.0.2",
    "torch==2.9.1",
    "torchvision==0.24.1",
    "transformers==4.57.1",
    "sentence-transformers==5.1.2",
    "paddlepaddle==3.2.1",
    "paddleocr==3.3.1",
    "numpy==2.3.4",
]

[project.optional-dependencies]
dev = [
    "pytest==9.0.0",
    "pytest-asyncio==1.3.0",
    "pytest-cov==7.0.0",
    "pytest-mock==3.15.1",
    "httpx==0.28.1",
    "black==25.11.0",
    "ruff==0.14.4",
    "mypy==1.18.2",
    "isort==5.13.2",
    "ipython==9.7.0",
    "ipdb==0.13.13",
    "rich==14.2.0",
    "python-dotenv==1.2.1",
    "mkdocs==1.6.1",
    "mkdocs-material==9.7.0",
    "pdoc==16.0.0",
    "memory-profiler==0.61.0",
    "line-profiler==5.0.0",
    "py-spy==0.4.1",
]
phase_final = [
    "psycopg2-binary==2.9.9",
    "pgvector==0.2.5",
    "redis==5.0.1",
    "celery==5.3.4",
    "prometheus-client==0.19.0",
]

[tool.uv.sources]
# Install PyTorch with CUDA support on Linux/Windows (CUDA doesn't exist for Mac).
# NOTE: We must explicitly request them as `dependencies` above. These improved
# versions will not be selected if they're only third-party dependencies.
torch = [
  { index = "pytorch-cuda", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchvision = [
  { index = "pytorch-cuda", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]

[[tool.uv.index]]
name = "pytorch-cuda"
# Use PyTorch built for NVIDIA Toolkit version 13.0.
# Available versions: https://pytorch.org/get-started/locally/
url = "https://download.pytorch.org/whl/cu130"
# Only use this index when explicitly requested by `tool.uv.sources`.
explicit = true

[project.scripts]
vibe = "src.cli:app"
vibe-server = "src.api.main:run"

[tool.ruff]
line-length = 150
target-version = "py312"

[tool.pytest.ini_options]
minversion = "9.0"
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-ra -q --strict-markers"

[tool.mypy]
python_version = "3.12"
warn_return_any = true
disallow_untyped_defs = true
```

## CORE — Functional Modules
For each module described below, pair this manual with `AI_DEVELOPMENT_GUIDE.md` and implement both happy path and failure modes.

### CORE-DET: `src/core/detector.py`
- Compose SigLIP (classification) and BLIP (captioning) inference.
- Accept optional `candidate_labels: list[str]` for targeted scoring; default to canonical label set stored in `config/candidates.yaml` (see `config/CONFIG_CONTRACT.md`).
- Return structured payload:
  ```python
  DetectedImage(
      labels=list[LabelScore],
      caption=str,
      embeddings=EmbeddingVector,
      image_id=uuid4,
  )
  ```
- Handle missing files and unsupported formats gracefully; log and continue batch.

### CORE-OCR: `src/core/ocr.py`
- PaddleOCR batch wrapper with GPU/CPU auto-detection.
- Provide streaming generator for large batches to control memory.
- Cache results keyed by file hash.

### CORE-PROC: `src/core/processor.py`
- Orchestrate detection + OCR + metadata extraction.
- Deduplicate using file checksum + perceptual hash.
- Persist results through `database.AssetRepository`.

### CORE-SRCH: `src/core/searcher.py`
- Maintain an embedding cache stored alongside asset metadata in SQLite; run cosine similarity locally for Phase 1.
- Structure adapters so migrating to PostgreSQL + pgvector in Phase Final only swaps out the storage implementation.
- Provide `search(query: str, limit: int = 20)` returning ranked assets, matched text, and explanation metadata.

### CORE-DB: `src/core/database.py`
- SQLAlchemy ORM models: `Asset`, `Label`, `Caption`, `TextBlock`, `Embedding`, `JobRun`.
- Migration workflow using Alembic even for SQLite (store scripts under `migrations/`).

## API — Interfaces
- **FastAPI (`src/api/main.py`):** create app factory `create_app()`; mount routes from `routes/ingest.py`, `routes/search.py`, `routes/annotations.py`.
- **Schemas:** All responses wrap payload inside `{ "status": "ok", "data": ... }` or `{ "status": "error", "message": ... }`.
- **CLI (`src/cli.py`):** Typer commands `ingest`, `search`, `rebuild-index`. Document options via Typer help strings.
- **Streamlit MVP:** Lives at the project root (`app.py`, with a blueprint stub for reference); ensure CLI and API reuse core services rather than duplicating logic.

## DATA — Storage & Assets
- Respect directory contracts from `DIRECTORY_STRUCTURE.md`.
- Provide dataset loaders in `blueprints/phase1/DATASET_USAGE.md` with explicit caching instructions.
- Log ingestion metrics to `log/ingestion.log` (rotating file handler via `utils/logging.py`).

## QA — Validation & Observability
- Minimum unit test coverage 80% for `src/core` and `src/api`.
- Include contract tests for CLI commands (use `CliRunner`).
- Provide benchmark script `tests/perf/test_ingestion_speed.py` to assert throughput target.
- Emit structured logs with correlation IDs: adopt helper `get_correlation_id()` with contextvar fallback.

## Integration Notes
- When introducing new dependencies, extend `DEPENDENCIES.md` and run `uv lock` so the root `uv.lock` stays the single source of truth.
- Document any deviations or experimental findings in `blueprints/phase_final/research/`.
- Sync acceptance criteria back to `AI_TASK_TRACKER.md` upon completion.

Use this manual to eliminate ambiguity. If an instruction conflicts with upstream decision records, halt and resolve the conflict before writing code.
