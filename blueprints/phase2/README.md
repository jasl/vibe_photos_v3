# Phase 2 Blueprint — Initial Draft

## Mission
Evolve the Vibe Photos prototype into a scalable perception service that supports rich object understanding, selective OCR, and cache-first data lifecycles. Phase 2 extends Phase 1 by introducing multi-object detection, adaptive text extraction, and semi-automatic labeling while hardening ingestion ergonomics for large photo libraries.

## Key Objectives
1. **Object Intelligence Upgrade** — Replace the single-shot SigLIP flow with an open-vocabulary detector (Grounding DINO or OWL-ViT) plus regional captioning to handle crowded scenes, food spreads, and lifestyle photos.
2. **Selective OCR** — Add a lightweight text-presence gate so PaddleOCR only runs when confidence justifies it, reducing false positives for text-free images.
3. **Label Automation Loop** — Generate machine labels via embeddings + clustering, then surface batches for human confirmation to bootstrap high-quality taxonomies without hand-tagging 30k photos.
4. **Operational Resilience** — Stand up a long-lived ingestion worker, detach caches from the relational schema, and add compatibility guardrails against upstream API churn.

## Scope
**In scope:**
- Grounding DINO / OWL-ViT based detection pipeline with SigLIP/BLIP re-ranking.
- Stream-aware ingestion service (queue + worker) that keeps ML weights hot-loaded.
- Cache redesign (Parquet/JSONL) with versioning to decouple from SQLite migrations.
- OCR gating heuristics (text detector, caption heuristics, or quick classifier).
- Auto-labeling jobs (caption embedding, clustering, human-in-the-loop review tasks).
- Compatibility smoke tests for transformers/PaddleOCR integrations.

**Out of scope:**
- Real-time sync, push notifications, or mobile capture flows.
- Dedicated annotation UI (Phase 2 will enrich the existing Streamlit surface only).
- Vector database rollout (remains in phase_final scope).
- Large-scale retraining or fine-tuning of foundation models (focus on orchestration first).

## Success Criteria
- Detect ≥90% of salient objects on mixed-scene validation sets with <5% hallucinated boxes per image.
- Reduce unnecessary OCR executions by ≥70% on a no-text benchmark set while maintaining recall on receipts/menus.
- Generate machine cluster labels covering ≥80% of the library with <10% manual edits per batch.
- Ingestion worker sustains ≥20 images/sec on CPU-only hardware with <2s cold start when models are already cached.
- Dry-run and compatibility tests block deprecated API usage (no DeprecationWarnings during scripted warmups).

## Execution Cadence
1. **Sprint 1:** Ship ingestion worker skeleton, cache schema, and dry-run tooling; port existing pipeline into the worker.
2. **Sprint 2:** Integrate Grounding DINO / OWL-ViT, regional re-ranking, and caption refinements. Validate memory/perf trade-offs.
3. **Sprint 3:** Implement OCR gating, cache writers/readers, and compatibility smoke tests (including CI hooks when available).
4. **Sprint 4:** Deliver auto-labeling job, UI review tooling, and polish documentation/tests ahead of phase hand-off.

## Document Map
| Topic | File |
|-------|------|
| Architecture updates & ingestion topology | `architecture.md` |
| Implementation roadmap & task breakdown | `implementation.md` |
| Testing strategy & manual checklists | `testing.md` |

Keep this draft in sync with design decisions logged under `decisions/` as Phase 2 evolves.
