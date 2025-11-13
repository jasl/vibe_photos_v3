# Blueprints Index — Coding AI Guide

Blueprints house the phase-specific plans, research notes, and implementation guides. Use this index to locate the right blueprint before coding.

## Structure
```
blueprints/
├── phase1/        # Active MVP guidance
├── phase2/        # Placeholder for future enhancements
└── phase_final/   # Target production architecture & research archives
```

## How to Consume
1. Start with [`AI_BLUEPRINT_GUIDE.md`](AI_BLUEPRINT_GUIDE.md) for instructions on navigating blueprint content.
2. When working on Phase 1 tasks, read `phase1/README.md` plus the relevant module docs (architecture, implementation, testing, dataset usage).
3. For long-term planning or ADR context, reference `phase_final/` and the research archives under `phase_final/research/`.

## Phase Summary
| Phase | Focus | Primary Docs |
|-------|-------|--------------|
| Phase 1 | Validate detector, OCR, SQLite search via CLI/API/UI prototypes. | `phase1/README.md`, `phase1/architecture.md`, `phase1/implementation.md`, `phase1/testing.md` |
| Phase 2 | Enhance semantic search and ranking (planned). | `phase2/` (placeholder) |
| Phase Final | Production blueprint with PostgreSQL, Celery, monitoring, Gradio UI. | `phase_final/README.md`, `phase_final/docs/*.md`, `phase_final/architecture/*.md` |

Keep this index in sync whenever blueprint files are added, removed, or renamed so future coding AIs can orient quickly.
