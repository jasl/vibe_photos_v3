# Optimization Summary — Coding AI Notes

- Adopted Celery + Redis for task orchestration with prioritized queues; Flower/Prometheus monitor throughput and failures.
- Standardized vector storage on PostgreSQL + pgvector with HNSW indexes; Faiss reserved for future scale if required.
- Reaffirmed product focus: image ingestion, understanding, and search—avoid diluting effort into non-core domains.

Use this summary as a reminder of past tuning decisions and revisit the underlying docs if additional optimization work begins.
