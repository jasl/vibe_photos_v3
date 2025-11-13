# UV Usage Contract — Coding AI Guide

`uv` is the mandatory toolchain manager for Vibe Photos. This document summarizes the commands and conventions every coding AI must follow.

## 1. Installation
```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
# macOS via Homebrew
brew install uv
# Windows PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# Fallback when installers are unavailable
python -m pip install uv
```

> **Note**  
> Using `pip` is acceptable **only** for installing the `uv` tool itself. All
> project dependency operations—including syncing, adding, or removing
> packages—must still be performed through `uv` to preserve the reproducible
> environment contract.

## 2. Core Workflow
```bash
uv venv --python 3.12         # create virtual environment at .venv/
source .venv/bin/activate     # (Windows: .venv\Scripts\activate)
uv sync                       # install dependencies from lock file
uv run python script.py       # execute Python scripts within the managed env
uv run pytest                 # run tests
```

`uv sync` is the canonical shortcut that reads `pyproject.toml` and `uv.lock` in one step—no extra arguments are required compared to the older `uv pip sync uv.lock` pattern.

## 3. Dependency Management Rules
- Declare direct dependencies in `pyproject.toml`; regenerate `uv.lock` after changes.
- Use `uv add PACKAGE==VERSION` to introduce new packages.
- Remove packages with `uv remove PACKAGE` and re-lock.
- Keep development extras under `[project.optional-dependencies].dev`; syncing `uv.lock` brings them in for local development.

## 4. Model Downloads
Set caches so large downloads remain in `models/`:
```bash
export TRANSFORMERS_CACHE="$(pwd)/models"
export PADDLEOCR_HOME="$(pwd)/models/paddleocr"
```
Run bootstrap scripts via `uv run python blueprints/phase1/download_models.py`.

## 5. Prohibited Tools
Do **not** use `pip`, `pip-tools`, `poetry`, `pipenv`, `conda`, or plain `venv`. All commands must go through `uv` to guarantee reproducibility.

## 6. Maintenance Checklist
- [ ] `.venv/` present and activatable.
- [ ] `uv.lock` committed and up to date.
- [ ] `uv run pytest` passes before each commit.
- [ ] Environment variables for model caches documented in relevant scripts.

Keep this contract enforced; divergence introduces hard-to-debug environment drift for subsequent coding AIs.
