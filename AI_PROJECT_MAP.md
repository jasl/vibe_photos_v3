# Vibe Photos Documentation Map — Coding AI Edition

Treat this file as the directory of directories. Every maintained artifact is listed here with its purpose so you can navigate the knowledge base without guesswork.

## 1. Primary Manuals
| File | Why it matters | When to read |
|------|----------------|--------------|
| [`README.md`](README.md) | High-level orientation, repository entry checklist. | First touch each session. |
| [`AI_DEVELOPMENT_GUIDE.md`](AI_DEVELOPMENT_GUIDE.md) | Execution blueprint: scope, milestones, reference architecture. | Before starting a feature or refactor. |
| [`AI_IMPLEMENTATION_DETAILS.md`](AI_IMPLEMENTATION_DETAILS.md) | Module-by-module expectations, data flow notes. | When implementing or modifying a component. |
| [`AI_CODING_STANDARDS.md`](AI_CODING_STANDARDS.md) | Style, logging, testing, error-handling guardrails. | While coding and reviewing. |
| [`AI_TASK_TRACKER.md`](AI_TASK_TRACKER.md) | Source of truth for backlog, owners, and current status. | At the beginning and end of every work cycle. |

## 2. Decision Intelligence
| File | Scope |
|------|-------|
| [`decisions/REQUIREMENTS_BRIEF.md`](decisions/REQUIREMENTS_BRIEF.md) | Product contract: user stories, must-have features, anti-goals. |
| [`decisions/TECHNICAL_DECISIONS.md`](decisions/TECHNICAL_DECISIONS.md) | Binding tech choices for each program phase. |
| [`decisions/AI_DECISION_RECORD.md`](decisions/AI_DECISION_RECORD.md) | Lightweight ADRs with rationale and status. |
| [`decisions/archives/*.md`](decisions/archives) | Superseded decisions (read only for context). |

## 3. Blueprint Library
- `blueprints/phase1/` – Active phase documents, runnable prototypes, dataset usage guides, architecture slices.
- `blueprints/phase_final/` – Target system design, integration plans, and research summaries for the production build.
- `blueprints/AI_BLUEPRINT_GUIDE.md` – How to consume and extend blueprint material.

## 4. Delivery Lifecycle Artifacts
| Purpose | Artifact |
|---------|----------|
| Release planning | [`ROADMAP.md`](ROADMAP.md) |
| Dependency governance | [`DEPENDENCIES.md`](DEPENDENCIES.md), [`UV_USAGE.md`](UV_USAGE.md) |
| Quality gates | [`FINAL_CHECKLIST.md`](FINAL_CHECKLIST.md), [`AI_AUDIT_REPORT.md`](AI_AUDIT_REPORT.md) |
| Phase transition notices | [`POC_PHASE_NOTICE.md`](POC_PHASE_NOTICE.md) |

## 5. How to Consume This Map
1. Identify the task you are about to perform (e.g., implement detector API, adjust vector DB integration).
2. Use the tables above to open the documents that constrain or inform that task.
3. Record any gaps you discover in `AI_TASK_TRACKER.md` under the "Notes" column so the next coding AI can patch the docs.
4. If multiple artifacts disagree, default to the most specific scope (e.g., blueprint > roadmap > README) and flag the conflict.

## 6. Maintenance Expectations
- Keep links valid when renaming or relocating files.
- When adding new documentation, register it here with the same structure so future coding AIs can discover it instantly.
- Update status indicators (e.g., ✅/🚧/⬜) in the referenced docs instead of duplicating progress notes here.

Stay disciplined about using this map—documentation drift is the fastest way to make the project hostile to autonomous contributors.
