# Coding AI Navigation Manual — Vibe Photos v3

Use this manual whenever you (a coding AI) enter the repository. It compresses the decision space into actionable checkpoints so you can self-orient in under five minutes.

## 1. Orient Yourself
1. Open [`AI_PROJECT_MAP.md`](AI_PROJECT_MAP.md) to understand how every document interrelates.
2. Review [`decisions/REQUIREMENTS_BRIEF.md`](decisions/REQUIREMENTS_BRIEF.md) and [`decisions/TECHNICAL_DECISIONS.md`](decisions/TECHNICAL_DECISIONS.md) to align on product needs and the locked technology stack.
3. Read [`AI_DEVELOPMENT_GUIDE.md`](AI_DEVELOPMENT_GUIDE.md) for the implementation blueprint and task breakdown.

## 2. Daily Operating Loop
| Step | Action | Reference |
|------|--------|-----------|
| Plan | Claim the highest-priority open item in [`AI_TASK_TRACKER.md`](AI_TASK_TRACKER.md) and confirm dependencies. | `AI_TASK_TRACKER.md` |
| Prepare | Sync tooling: Python 3.12 via `uv`, install dependencies, download required models. | `UV_USAGE.md`, `DEPENDENCIES.md` |
| Build | Follow module-specific guidance, enforce coding standards, write tests first. | `AI_CODING_STANDARDS.md`, `AI_DEVELOPMENT_GUIDE.md` |
| Validate | Execute linters/tests and log outcomes in commit messages and PR summary. | `FINAL_CHECKLIST.md` |
| Report | Update the task board (⬜→🟨→✅) and document any blockers. | `AI_TASK_TRACKER.md` |

## 3. Guardrails You Must Honor
- **Language policy:** All source code, comments, and commit messages in English. Documentation may be bilingual when it improves clarity.
- **Tooling policy:** Only `uv` for environment + dependency management; no `pip`, `conda`, or `poetry`.
- **Design policy:** Prefer functional composition, guard-clause error handling, explicit typing, and deterministic logging.
- **Scope policy:** Implement only what is authorized in the roadmaps and decision logs. Escalate uncertainties via notes in `AI_TASK_TRACKER.md`.

## 4. Artifact Quick Reference
```
AI_DEVELOPMENT_GUIDE.md   → Program-level requirements, milestones, scaffolding
AI_IMPLEMENTATION_DETAILS.md → Deep-dive on module behaviors and data flows
AI_CODING_STANDARDS.md    → Style, logging, error-handling, testing rules
AI_TASK_TRACKER.md        → Current backlog, priority, owners, status icons
DIRECTORY_STRUCTURE.md    → Storage rules for data/cache/log/tmp
UV_USAGE.md               → Environment bootstrap scripts
ROADMAP.md                → Phase objectives and exit criteria
```

## 5. Execution Checklist
- [ ] Confirm you are working on the correct git branch.
- [ ] Initialize/activate the Phase 1 `uv` environment.
- [ ] Run formatters/linters/tests relevant to your change.
- [ ] Update documentation snippets impacted by the code.
- [ ] Capture diffs for review and summarize them in the PR template when prompted.

## 6. Escalation Protocol
If you detect mismatched specifications, missing context, or tooling blockers:
1. Document the issue in the "Notes" column of `AI_TASK_TRACKER.md`.
2. Cross-reference impacted decisions in `decisions/` and link them inside your PR description.
3. Avoid speculative fixes—pause coding until documentation is realigned.

Stay disciplined, keep the documentation synchronized, and the project remains tractable for any subsequent coding AI.
