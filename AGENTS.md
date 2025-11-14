# Repository Guidelines

## Project Structure & Module Organization
Blueprint specs live in `blueprints/phase1/`, but the runnable Phase 1 scripts now live at the project root (duplicate stubs remain under `blueprints/` for reference). Long-term architecture stays in `blueprints/phase_final/`. Runtime settings sit in `config/` (`config/settings.yaml` lands via `./init_project.sh`). Treat `data/`, `cache/`, `log/`, and `tmp/` as disposable run outputs, keep heavy model weights inside `models/`, and source read-only evaluation assets from `samples/`. Consult root docs before shifting scope.

## Environment & Configuration
Python 3.12 + `uv` is non-negotiable. Create the env (`uv venv --python 3.12`, `source .venv/bin/activate`), run `uv sync`, and execute every script via `uv run …` to keep extras aligned. Run `./init_project.sh` once to build `config/settings.yaml`, then keep dataset paths and incremental flags current there. Point `TRANSFORMERS_CACHE` and `PADDLEOCR_HOME` at `models/` to prevent redundant downloads.

## Build, Test, and Development Commands
- `uv run python process_dataset.py` — batch-ingest samples and refresh SQLite/cache outputs.
- `uv run streamlit run app.py` — launch the dashboard after processing completes.
- `uv run python quick_start.py` — same logic as the shell wrapper, useful when scripting.
- `uv run python download_models.py` — fetch SigLIP/BLIP/PaddleOCR weights into `models/`.
- `uv run pytest` — execute the whole suite with strict markers enabled; add `-k` filters for targeted runs.

## Coding Style & Naming Conventions
Follow `AI_CODING_STANDARDS.md`: `black` + `ruff` enforce the 150-char line cap, import grouping (stdlib → third-party → local), and lint suites (`E/W/F/I/B/UP`). Public APIs require type hints + docstrings, dataclasses/TypedDicts trump loose dicts, and everything (code, comments, logs) stays in English. Use the shared logger with structured `extra={}` metadata, avoid prints, and rely on guard clauses to keep functions short.

## AI Assistant Library Usage Rules
- If you are not clearly familiar with a library or dependency API, inspect local usages and pinned versions (including `pyproject.toml` and `DEPENDENCIES.md`) before writing or modifying code.
- Consult the official documentation and/or changelog for the specific version in use to confirm signatures and behaviors, and prefer documented patterns over memory.
- When you cannot confidently verify an API (for example, ambiguous or missing docs), state this explicitly and avoid guessing or relying on potentially obsolete patterns.

## Testing Guidelines
Tests live under `tests/` with files named `test_*.py` (see `pyproject.toml`). Follow `blueprints/phase1/testing.md`: unit coverage for detectors/OCR/repos, integration smoke for CLI/API/Streamlit flows, and performance batches that log to `log/perf.log`. Maintain ≥80 % coverage across `src/core` + `src/api`, tag slow suites with `@pytest.mark.integration` or `@pytest.mark.performance`, and keep fixtures in `tests/fixtures/`.

## Commit & Pull Request Guidelines
Commits follow `type(scope): summary` (e.g., `feat(detector): add confidence calibration`). Bundle code, tests, and doc updates together, and avoid mixing refactors with feature work unless called out. PRs must reference roadmap items or decision logs, summarize behavior + tests, attach `uv run pytest` results (plus perf logs when applicable), and include screenshots for UI/CLI updates. Update `AI_TASK_TRACKER.md` and relevant blueprints whenever behavior or interfaces shift so downstream agents inherit an accurate state.
