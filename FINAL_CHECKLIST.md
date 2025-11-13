# Final Documentation Checklist — Coding AI Handoff

Run through this checklist before declaring a milestone complete or handing the project to another coding AI.

## 1. Repository Hygiene
- [ ] `README.md` reflects current phase status and quickstart commands.
- [ ] `AI_PROJECT_MAP.md` lists every newly created or renamed document.
- [ ] `DIRECTORY_STRUCTURE.md` matches the actual folder layout.
- [ ] `DEPENDENCIES.md` & `uv.lock` updated after dependency changes.

## 2. Decision & Roadmap Alignment
- [ ] Roadmap milestones updated with accurate deliverable status.
- [ ] New architectural choices recorded in `decisions/AI_DECISION_RECORD.md` (or existing records amended).
- [ ] Deprecated decisions archived under `decisions/archives/` with a note referencing replacements.

## 3. Implementation Documentation
- [ ] `AI_DEVELOPMENT_GUIDE.md` reflects current scope and DoD.
- [ ] `AI_IMPLEMENTATION_DETAILS.md` includes module-specific expectations for all touched components.
- [ ] Blueprint documents updated with diagrams, flows, or research notes relevant to recent work.

## 4. Quality Evidence
- [ ] Tests and linters executed; results captured in commit/PR description.
- [ ] Performance benchmarks (if applicable) recorded in `log/perf.log` with environment details.
- [ ] `AI_TASK_TRACKER.md` statuses + notes reflect the real backlog state.

## 5. Handoff Notes
- [ ] Outstanding risks or TODOs documented in tracker notes or dedicated blueprint research files.
- [ ] Environment bootstrap steps validated (no missing commands or broken scripts).
- [ ] Any manual data preparation steps described in `blueprints/phase1/DATASET_USAGE.md`.

Complete every box before transitioning ownership. This keeps the repository reliable for autonomous coding AIs.
