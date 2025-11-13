# Implementation Field Manual — Vibe Photos Coding AI

This manual extends the development playbook with executable details. Each section maps directly to a backlog tag (ENV, CORE, API, DATA, QA). When you pick up a task, locate the matching section below and follow the prescribed steps.

## ENV — Environment & Scaffolding
### ENV-001: Repository Skeleton Generator
Use the following script to (re)hydrate the canonical directory structure. Run it only when bootstrapping a clean checkout.
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
mkdir -p config data cache/{images/{thumbnails,processed},detections,ocr,embeddings} models logs tmp
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
requires-python = ">=3.12"
license = { text = "MIT" }
authors = [{ name = "Vibe Photos Team", email = "team@vibephotos.ai" }]

dependencies = [
    "torch==2.9.1",
    "torchvision==0.24.1",
    "transformers==4.57.1",
    "pillow==11.3.0",
    "fastapi==0.121.1",
    "uvicorn[standard]==0.38.0",
    "python-multipart==0.0.6",
    "sqlalchemy==2.0.44",
    "alembic==1.13.1",
    "pydantic==2.11.10",
    "pydantic-settings==2.2.1",
    "paddlepaddle==2.6.0",
    "paddleocr==2.7.3",
    "typer==0.20.0",
    "rich==14.2.0",
    "numpy==2.3.4",
    "python-dotenv==1.0.0",
    "aiofiles==23.2.1",
    "httpx==0.25.2",
    "tqdm==4.66.1",
]

[project.optional-dependencies]
dev = [
    "pytest==8.0.0",
    "pytest-asyncio==0.23.2",
    "pytest-cov==4.1.0",
    "ruff==0.6.0",
    "mypy==1.8.0",
    "ipython==8.18.1",
    "notebook==7.0.6",
]
phase2 = [
    "sentence-transformers==2.5.1",
    "faiss-cpu==1.7.4",
]
phase3 = [
    "psycopg2-binary==2.9.9",
    "redis==5.0.1",
    "celery==5.3.4",
    "prometheus-client==0.19.0",
]

[project.scripts]
vibe = "src.cli:app"
vibe-server = "src.api.main:run"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = []

[tool.uv.sources]
torch = [{ index = "pytorch-cuda", marker = "sys_platform == 'linux' or sys_platform == 'win32'" }]
torchvision = [{ index = "pytorch-cuda", marker = "sys_platform == 'linux' or sys_platform == 'win32'" }]

[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "ARG", "ERA"]
ignore = ["E501"]

[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = "-ra -q --strict-markers --cov=src --cov-report=term-missing"

[tool.mypy]
python_version = "3.12"
warn_return_any = true
disallow_untyped_defs = true
```

## CORE — Functional Modules
For each module described below, pair this manual with `AI_DEVELOPMENT_GUIDE.md` and implement both happy path and failure modes.

### CORE-DET: `src/core/detector.py`
- Compose SigLIP (classification) and BLIP (captioning) inference.
- Accept optional `candidate_labels: list[str]` for targeted scoring; default to canonical label set stored in `config/candidates.yaml` (create if missing).
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
- Manage embedding index (FAISS for Phase 1) with fallback to cosine search over SQLite.
- Provide `search(query: str, limit: int = 20)` returning ranked assets, matched text, and explanation metadata.

### CORE-DB: `src/core/database.py`
- SQLAlchemy ORM models: `Asset`, `Label`, `Caption`, `TextBlock`, `Embedding`, `JobRun`.
- Migration workflow using Alembic even for SQLite (store scripts under `migrations/`).

## API — Interfaces
- **FastAPI (`src/api/main.py`):** create app factory `create_app()`; mount routes from `routes/import.py`, `routes/search.py`, `routes/annotations.py`.
- **Schemas:** All responses wrap payload inside `{ "status": "ok", "data": ... }` or `{ "status": "error", "message": ... }`.
- **CLI (`src/cli.py`):** Typer commands `ingest`, `search`, `rebuild-index`. Document options via Typer help strings.
- **Streamlit MVP:** Lives under `blueprints/phase1/app.py`; ensure CLI and API reuse core services rather than duplicating logic.

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
- When introducing new dependencies, extend `DEPENDENCIES.md` and run `uv lock` to update `uv.lock`.
- Document any deviations or experimental findings in `blueprints/phase_final/research/`.
- Sync acceptance criteria back to `AI_TASK_TRACKER.md` upon completion.

Use this manual to eliminate ambiguity. If an instruction conflicts with upstream decision records, halt and resolve the conflict before writing code.
