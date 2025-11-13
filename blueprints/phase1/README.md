# Phase 1 Blueprint — Coding AI Brief

## Mission
Validate the Vibe Photos perception + search pipeline using an offline-first architecture. Deliver a working prototype within ~2 weeks.

## Dataset Context
- ~30k PNG images (videos ignored) spanning 2012–2025 across multiple regions.
- Maintained outside the repo; copy samples into `samples/` before running pipelines.

## Scope
**In scope:** batch ingestion, SigLIP classification, BLIP captioning, PaddleOCR extraction, SQLite persistence, metadata search, Streamlit UI, Typer CLI.

**Out of scope:** real-time ingestion, vector search, brand/model detection, user auth, few-shot learning, non-image media handling.

## Success Criteria
- Process ≥30k images without failure; throughput target ≥10 images/sec on CPU.
- Achieve ≥80% accuracy on primary categories; OCR accuracy ≥85% on mixed-language text.
- Provide search endpoint/UI with acceptable relevance for stakeholder review.

## Execution Cadence
1. **Days 1–3:** Bootstrap environment, implement detector + OCR wrappers.
2. **Days 4–7:** Build processor + database layer, seed ingestion scripts.
3. **Days 8–10:** Wire FastAPI, CLI, Streamlit UI.
4. **Days 11–14:** Testing, benchmarks, documentation.

## Document Map
| Topic | File |
|-------|------|
| Architecture overview | `architecture.md` |
| Implementation plan & module contracts | `implementation.md` |
| Testing strategy | `testing.md` |
| Dataset handling | `DATASET_USAGE.md`, `DATASET_ANALYSIS.md` |
| Design rationale | `design_decisions.md` |

Follow these blueprints closely and update them when the implementation diverges.
