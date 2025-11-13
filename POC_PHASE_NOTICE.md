# Proof-of-Concept Operating Notice — Coding AI Brief

## 1. Phase Context
- **Program stage:** Phase 1 (POC). Objective is to validate technical feasibility, not to harden for production.
- **Last review:** 2025-11-13. Treat guidance as current unless roadmaps change.

## 2. Ground Rules for POC Work
1. **Bias for speed:** Ship working prototypes quickly; optimization and polish are secondary.
2. **Reset anytime:** You may wipe `data/`, `cache/`, `log/`, and `tmp/` without warning. Record the reset in commit or tracker notes.
3. **Mutable architecture:** Structural changes are acceptable; update blueprints and decision records to reflect them.
4. **Model cache persistence:** Retain `models/` between runs to avoid repeated downloads.
5. **Read-only artifacts:** `samples/` and `blueprints/` must not be modified casually—treat them as specs.

## 3. Phase Objectives & Exit Criteria
| Phase | Duration | Goal | Exit Criteria |
|-------|----------|------|---------------|
| Phase 1 (now) | ~2 weeks | Validate detector + OCR + basic search workflow. | ≥10 images/sec throughput, satisfactory qualitative captions, working MVP UI/CLI. |
| Phase 2 (planned) | ~1 month | Add semantic search & hybrid ranking. | Demonstrated vector search accuracy on curated dataset, user feedback on search UX. |
| Phase Final (future) | 2–3 months | Production platform with full infra. | Stable PostgreSQL/pgvector stack, Celery workers, monitoring baseline. |

## 4. Execution Checklist
- [ ] Bootstrap environment via `UV_USAGE.md`.
- [ ] Implement detector/processor/database as per `AI_DEVELOPMENT_GUIDE.md`.
- [ ] Run ingestion benchmark and capture metrics in `log/perf.log`.
- [ ] Summarize findings in `AI_TASK_TRACKER.md` under relevant items.
- [ ] Review next-phase blockers once MVP tasks are complete.

## 5. Non-Goals During POC
- Production-grade resilience, multi-tenant support, or exhaustive test coverage.
- Automated migrations across POC revisions.
- Full localization or user management features.

## 6. Communication Expectations
- Document key learnings and pivots inside blueprint research files.
- When discarding or replacing a prototype, log the rationale in `decisions/AI_DECISION_RECORD.md`.
- Keep the documentation set synchronized so the next coding AI understands the current baseline.

Operate with urgency, validate assumptions, and prepare the path for later hardening work.
