# Phase 2 Implementation Draft

## 1. Workstreams
1. **Ingestion Service**
   - Build a Typer CLI (`src/cli/ingest_service.py`) that enumerates media, writes queue records, and can resume from checkpoints.
   - Implement an asyncio worker (`src/services/ingestion_worker.py`) with cooperative batching, structured logging, and graceful shutdown.
   - Add queue abstraction (`src/services/queue.py`) with pluggable backends (SQLite now, Redis later). Include idempotent dedupe per image hash.

2. **Detection Stack Refresh**
   - Introduce open-vocabulary detector module (`src/models/grounding.py`) wrapping Grounding DINO / OWL-ViT with configurable proposal count.
   - Extend `SiglipClassifier` to accept crops + bounding boxes, returning per-region label candidates.
   - Update detector orchestrator (`src/core/detector.py`) to merge region-level outputs, attach confidences, and emit multi-object payloads.
   - Surface bounding boxes + captions in cache schema for later UI overlays.

3. **Selective OCR**
   - Create a text gate helper (`src/core/text_gate.py`) using PaddleOCR detection-only pass or heuristic classifier.
   - Modify worker pipeline to evaluate gate results before calling `PaddleOCREngine.extract_text`.
   - Persist skip reasons and detection confidences for observability.

4. **Cache & Import Pipeline**
   - Define cache manifest spec (`cache/phase2/README.md`) with schema version, model hashes, generation timestamp.
   - Write cache writer utilities (`src/services/cache_writer.py`) that store detection/OCR/embedding outputs separately.
   - Implement database importer CLI (`src/cli/import_cache.py`) to replay cache files into SQLite tables.

5. **Label Automation**
   - Add embedding job (`src/jobs/label_bootstrap.py`) that clusters captions + detected object labels using SentenceTransformers + HDBSCAN.
   - Provide annotation summary export (CSV/JSON) for manual review and re-import.
   - Update Streamlit dashboard to visualize machine clusters and accept manual overrides (Phase 2 baseline UI).

6. **Compatibility Guardrails**
   - Ship minimal smoke tests ensuring `build_detector()` and `PaddleOCREngine` warmups execute without `DeprecationWarning`.
   - Document manual pre-commit checks (`uv run python -m pip check`, `uv run python -W error::DeprecationWarning process_dataset.py --dry-run`).
   - Add ADR covering long-lived worker + cache-first approach.

## 2. Dependencies & Tooling
- Pin Grounding DINO / OWL-ViT weights in `pyproject.toml` + `DEPENDENCIES.md`.
- Evaluate `onnxruntime` vs. PyTorch runtime for CPU ingestion.
- Ensure `uv` scripts include new CLIs and worker entrypoints.

## 3. Data Contracts
- Detection result schema: `[image_id, region_id, bbox, labels[], caption, detector_version]`.
- OCR result schema: `[image_id, region_id?, text, confidence, language, source]`.
- Cache manifest contains `phase`, `version`, `models`, `generated_at`, `schema_hash`, and `dataset_digest`.

## 4. Migration Notes
- Maintain compatibility adapters to ingest Phase 1 cache until new run completes.
- Provide rollback plan: worker can be paused and CLI fallback re-enabled if queue fails.

## 5. Risks
- Model size and inference cost may require GPU to meet throughput target; monitor CPU fallback metrics.
- Grounding DINO licensing/weights might change; keep local mirror.
- SentenceTransformer clustering may yield noisy labels; design UI to triage quickly.
