# Documentation Cleanup Summary — Coding AI Note

- All active decisions now live in `decisions/AI_DECISION_RECORD.md` and Phase Final docs; research files remain for historical context only.
- Vector store references standardized to PostgreSQL + pgvector; Faiss appears solely as a future option when scale demands it.
- Reorganized docs so implementers read decision/architecture guides first, then consult research archives if needed.

Maintain this separation when adding new materials: decisions in `decisions/`, implementation guides in `blueprints/`, exploratory notes here.
