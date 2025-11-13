# Vector DB & Model Versioning — Coding AI Summary

## Vector Store Strategy
- Use PostgreSQL + pgvector to co-locate metadata and embeddings, benefiting from ACID transactions and hybrid SQL/vector queries.
- Default to HNSW indexes for fast similarity search; rebuild or tune parameters as dataset size grows.
- Maintain `photo_embeddings` table with columns for `embedding_model`, `embedding_version`, and vector payload.

## Model Versioning
- Track every deployed model (SigLIP, BLIP, DINOv2, OCR) with semantic versions (e.g., `siglip-base@v1.2.0`).
- Store metadata in a `model_registry` table referencing artifact locations and compatible vector dimensions.
- When updating models:
  1. Generate new embeddings in the background.
  2. Write to staging tables/indices.
  3. Swap aliases once validation metrics meet thresholds; keep old versions for rollback.

## Update Workflow
```
new model → background embedding job → build pgvector index → validate recall/latency → promote → archive previous version
```

Document any deviations or tooling decisions in `AI_DECISION_RECORD.md` for future maintainers.
