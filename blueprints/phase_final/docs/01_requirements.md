# Phase Final Requirements — Coding AI Summary

## Users & Context
- Primary audience: Chinese content creators managing 30k+ local photos for reviews, tutorials, and lifestyle content.
- Pain points: retrieving specific assets (rare devices, food shots, documents) quickly, often under deadline pressure.

## Functional Needs
1. **Perception** — Recognize categories, brands/models, captions, and text (OCR). Support few-shot personalization for rare devices.
2. **Search** — Natural-language queries, hybrid filters (metadata + embeddings + time), similarity grouping, export bundles.
3. **Operations** — Bulk ingestion, incremental updates, dedupe, audit trails, manual overrides.
4. **Learning Loop** — Capture corrections, adapt suggestions, surface confidence scores for human review.

## Success Criteria
- ≥85% accuracy for common categories; ≥60% for niche devices with human-in-the-loop confirmation.
- Search latency ≤500 ms for 50k assets; ingestion throughput ≥10 images/sec sustained with workers.
- Users derive results within 3 minutes, saving ≥30 minutes per day on asset retrieval.

## Non-Goals
- No cloud storage mandate; operate locally or within user-controlled infra.
- No social sharing or heavy image editing features.
- No guarantee of fully autonomous labeling—human participation remains integral.

Use this summary with `02_solution_design.md` to map requirements to architecture.
