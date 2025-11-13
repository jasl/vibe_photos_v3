# Vibe Photos v3 — Coding AI Operations Brief

This repository is the shared workspace for every coding AI that will deliver the Vibe Photos intelligent photo management platform. Use this README as your command center: it compresses the minimum situational awareness you need before acting and points you to the detailed playbooks in the docs folder.

## 1. Mission Snapshot
- **Target users:** Chinese content creators who need to mine large personal photo libraries for product shots, recipes, tutorials, and documents.
- **Value proposition:** Rapid image understanding + semantic search + OCR driven tagging.
- **Current program phase:** Phase 1 (proof of capability). Later phases are defined and frozen—do not improvise beyond the documented scope.

## 2. Repository Protocol Map
```
vibe_photos_v3/
├── blueprints/             # Delivery blueprints grouped by program phase
│   ├── phase1/             # Active phase – executable prototypes and datasets
│   └── phase_final/        # Target architecture once Phase 1 validates assumptions
├── decisions/              # Binding decision records and archives
├── data/, cache/, log/, tmp/  # Runtime storage areas (see DIRECTORY_STRUCTURE.md)
├── docs overview files     # *.md manuals rewritten for coding AI consumption
├── pyproject.toml, uv.lock # Python 3.12 toolchain managed through uv only
└── LICENSE
```
Consult [`DIRECTORY_STRUCTURE.md`](DIRECTORY_STRUCTURE.md) for operational detail on every folder that matters during execution.

## 3. Document Jump Table
| Objective | Read This First |
|-----------|-----------------|
| Understand every maintained document quickly | [`AI_PROJECT_MAP.md`](AI_PROJECT_MAP.md) |
| Align with timeline & deliverables | [`ROADMAP.md`](ROADMAP.md) |
| Confirm environment tooling | [`UV_USAGE.md`](UV_USAGE.md) + [`DEPENDENCIES.md`](DEPENDENCIES.md) |
| Check mandatory dev practices | [`AI_CODING_STANDARDS.md`](AI_CODING_STANDARDS.md) |
| Prepare for handoff or reviews | [`FINAL_CHECKLIST.md`](FINAL_CHECKLIST.md) |
| Investigate Phase 1 specifics | [`blueprints/phase1/README.md`](blueprints/phase1/README.md) |
| Investigate target architecture | [`blueprints/phase_final/README.md`](blueprints/phase_final/README.md) |

## 4. Execution Quickstart
1. **Pin the toolchain** – Python 3.12 only, managed via `uv`. Follow [`UV_USAGE.md`](UV_USAGE.md) step-by-step.
2. **Rehydrate the environment** – Activate the Phase 1 venv and sync `requirements.txt`.
   ```bash
   cd blueprints/phase1
   uv venv --python 3.12
   source .venv/bin/activate
   uv pip sync requirements.txt
   ```
3. **Prime the models** – Run the provided download script once, then process the sample dataset.
   ```bash
   uv run python download_models.py
   uv run python process_dataset.py
   ```
4. **Stay within scope** – every deviation must be justified against the decision logs in `decisions/` before implementation.

## 5. Technology Baseline
- **Perception stack:** SigLIP (multilingual classification) + BLIP (captioning) + PaddleOCR.
- **Search stack:** SQLite/FAISS for Phase 1, PostgreSQL + pgvector + Celery + Redis for Phase Final.
- **Serving stack:** FastAPI + Uvicorn, Streamlit UI for MVP, Gradio UI for production.
- **Language rules:** Implementation code in English, documentation can be bilingual when necessary.

## 6. Delivery Status
- ✅ Document suite synchronized for coding AI.
- ✅ Technical decisions locked for Phase Final (see `decisions/TECHNICAL_DECISIONS.md`).
- 🚧 Phase 1 build-out in progress—treat documentation as authoritative requirements.
- ⏳ Phase 2 and Final execution pending validation milestones.

## 7. License
Distributed under the [MIT License](LICENSE).
