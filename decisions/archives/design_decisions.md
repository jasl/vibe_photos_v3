# Archive: Phase 1 Design Adjustments

This archived note captures the simplifications we made after early reviews.

## Key Simplifications (Nov 2024)
- Dropped real-time pipelines; committed to offline batch ingestion.
- Reduced recognition scope to high-level categories + captions (no brand/model detection).
- Deferred few-shot personalization and advanced search until later phases.
- Chose SQLite + FTS5 over PostgreSQL/pgvector for MVP to minimize complexity.
- Adopted Streamlit MVP UI instead of a full React frontend.

## Rationale
The initial design tried to match production ambitions. Reviewers highlighted risk of scope creep and dependency churn. We aligned Phase 1 with a "prove it works" mindset and postponed optimizations until metrics justified them.

Refer to `AI_DECISION_RECORD.md` for the current, active decisions.
