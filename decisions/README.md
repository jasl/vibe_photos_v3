# Decisions Index — Coding AI Reference

Use this index to locate every binding decision made on Vibe Photos. Keep it synchronized whenever new choices are logged.

## Primary Records
- [`REQUIREMENTS_BRIEF.md`](REQUIREMENTS_BRIEF.md) — Business and user requirements with no technical noise.
- [`TECHNICAL_DECISIONS.md`](TECHNICAL_DECISIONS.md) — Aggregated decision log grouped by topic and phase.
- [`AI_DECISION_RECORD.md`](AI_DECISION_RECORD.md) — Incremental ADRs with rationale, status, and follow-up actions.

## Archive
`archives/` contains superseded decisions. Review them only for historical context and reference the replacement in active docs.
- `FINAL_TECHNOLOGY_DECISIONS.md`
- `design_decisions.md`
- `SIGLIP_CHOICE.md`

## Maintenance Rules
1. Document every new decision in `AI_DECISION_RECORD.md` first, then summarize it in `TECHNICAL_DECISIONS.md`.
2. When a decision is replaced, move the outdated entry into `archives/` with a note pointing to its successor.
3. Keep timestamps, owners, and impacted modules up to date so coding AIs can trace accountability.

## Related Blueprints
- Architecture & system design: `../blueprints/phase_final/architecture/`
- Phase-specific plans: `../blueprints/phase1/` and `../blueprints/phase_final/`

If you change the decision-making process, document the update here to keep future coding AIs aligned.
