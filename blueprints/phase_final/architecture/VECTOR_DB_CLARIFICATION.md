# Vector DB Clarification — Coding AI Note

**Decision:** PostgreSQL + pgvector is sufficient for the current scale (~30k embeddings) and remains the primary vector store.

## Reasons
- Keeps metadata and embeddings in a single transactional system.
- Offers hybrid SQL + vector queries with manageable latency (<20 ms for 30k vectors on HNSW).
- Simplifies operations; no secondary synchronization layer required.

## When to Reconsider
Introduce an external vector service only if:
- Embedding count exceeds ~1M and latency crosses 200 ms.
- Concurrency requires GPU-accelerated search.
- Operational monitoring reveals sustained bottlenecks despite tuning.

Until then, continue investing in pgvector (index tuning, read replicas) and document any performance findings in `research/OPTIMIZATION_SUMMARY.md`.
