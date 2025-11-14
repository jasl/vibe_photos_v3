# Phase 2 Architecture Draft

## 1. System Overview
Phase 2 refactors the ingestion layer from a batch CLI into a resident service. A Typer-based CLI enqueues image paths into a durable queue (initially SQLite-backed, upgrade path to Redis/SQS). One or more async workers consume jobs, reuse pre-loaded models, and emit results into a versioned cache store before optional database import. The Streamlit UI and FastAPI layer continue to query SQLite but can replay cache snapshots after schema changes.

```
+-------------+      enqueue       +----------------+      batched jobs      +---------------------+
| Typer CLI   |  ───────────────▶  | SQLite Queue   |  ───────────────────▶  | Async Worker Pool   |
+-------------+                    +----------------+                        |  (detector/OCR)     |
       │                                   │                                +----------┬----------+
       │                                   │                                           │
       │                                   ▼                                           ▼
       │                             Cache Writer (Parquet/JSONL)                Cache Importer
       │                                   │                                           │
       ▼                                   ▼                                           ▼
Streamlit/REST consume SQLite ◀────── Cache Loader ───────────────────────────────▶ SQLite Database
```

## 2. Component Highlights
- **Ingestion CLI**: Discovers filesystem assets, chunks paths, and pushes queue entries with idempotency tokens.
- **Worker Runtime**: Asyncio event loop with limited concurrency per device type. Keeps Grounding DINO / OWL-ViT, SigLIP, BLIP, and PaddleOCR instances alive. Supports hot reloading of config/candidate labels without restart.
- **Detection Pipeline**: Grounding DINO (primary) proposes boxes; OWL-ViT can serve as lightweight fallback on CPU. Each region is cropped and re-ranked by SigLIP classifier; BLIP captioner produces global + regional captions. Regions are stored alongside metadata to support later UI overlays.
- **OCR Gate**: Fast text-detection stage (e.g., PaddleOCR detection-only mode or DBNet/EAST). Worker only runs recognition on crops where detection confidence > threshold or captions mention textual cues.
- **Cache Store**: Versioned directories in `cache/phase2/<version>/[detections|ocr|embeddings]` using Parquet for structured data and JSONL for quick diffs. Includes manifest with schema hash and model versions to ensure compatibility across database rebuilds.
- **Database Importer**: Separate job that reads cache manifests and populates SQLite (or future Postgres) schemas. Allows destructive migrations without reprocessing the photo corpus.

## 3. Operational Concerns
- **Warmup Strategy**: Worker boot loads models concurrently and runs synthetic warmups (using sample images) to surface deprecation warnings early. Warmup logs feed into compatibility tests.
- **Scaling**: Start with single-process worker; design queue abstraction to support horizontal scale (multiple workers pulling from shared queue) in later phases.
- **Observability**: Worker publishes metrics (processing latency, detection/OCR skip counts, queue depth) to `log/perf.log` and exposes heartbeat file for health checks.
- **Failure Handling**: Failed jobs are retried with exponential backoff and dead-lettered after configurable attempts. Cache writer marks partial batches so importers can skip inconsistent outputs.

## 4. External Dependencies
- **Grounding DINO / OWL-ViT** via Hugging Face transformers & timm.
- **SentenceTransformers** (or similar) for embedding captions when seeding auto-label clusters.
- **Optional GPU acceleration** but CPU fallback must be preserved for ingestion worker.

## 5. Open Questions
- Evaluate whether to wrap Grounding DINO in ONNX/TensorRT for faster inference on GPU-heavy nodes.
- Determine best lightweight text gate (DBNet, CRAFT, or caption heuristics) based on accuracy vs. dependency footprint.
- Decide if cache importer should live inside ingestion worker (separate coroutine) or as a discrete CLI triggered post-processing.
