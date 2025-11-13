# Blueprint Consumption Guide — Coding AI

Blueprints translate requirements into implementable modules. Follow this guide to extract the information you need before coding.

## Navigation Steps
1. **Orient:** Read `phase1/README.md` for the active phase overview.
2. **Deep dive:** Open architecture, implementation, testing, and dataset docs inside the phase folder relevant to your task.
3. **Cross-check:** Align with `decisions/AI_DECISION_RECORD.md` to ensure technical choices match.
4. **Log findings:** Update blueprint research notes if you discover better approaches or record experiment results.

## Module Map (Phase 1)
| Module | File | Purpose |
|--------|------|---------|
| Detector | `phase1/implementation.md` (`src/core/detector.py`) | SigLIP + BLIP fusion, label/caption generation. |
| OCR | `phase1/implementation.md` (`src/core/ocr.py`) | PaddleOCR integration with caching. |
| Processor | `phase1/implementation.md` (`src/core/processor.py`) | Batch orchestration and dedupe. |
| Database | `phase1/architecture.md` / `implementation.md` | SQLite schema & repository. |
| API | `phase1/architecture.md` / `testing.md` | FastAPI routes, request/response contracts. |
| UI/CLI | `phase1/README.md` | Streamlit MVP & Typer commands. |

## Research & Future Phases
- `phase_final/docs/*.md` — Production-level requirements, solution design, implementation roadmaps.
- `phase_final/architecture/*.md` — System diagrams, vector DB clarifications, queue strategy.
- `phase_final/research/*.md` — Experiment archives, optimization notes, lessons learned.

## Maintenance Checklist
- Update blueprint docs whenever implementation deviates from the plan.
- Add cross-links to code modules and ADR entries for traceability.
- Archive superseded plans in `phase_final/research/REVIEW_REPORT_ARCHIVE.md` or similar files.

Blueprints are living documents—keep them synchronized with the codebase to prevent drift for future coding AIs.
