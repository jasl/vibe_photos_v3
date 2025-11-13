# Phase 1 Testing Plan — Coding AI

## Objectives
- Validate detector/OCR/search functionality against acceptance targets.
- Measure ingestion throughput and search latency.
- Confirm user workflows (CLI/API/UI) are operable without manual hacks.

## Test Suite Outline
| Area | Tests |
|------|-------|
| Unit | Detector label selection, caption generation, OCR extraction, database repositories, search scoring helpers. |
| Integration | CLI ingestion on small dataset, `/import` + `/search` API flows, Streamlit view smoke test. |
| Performance | Batch 500 images; ensure ≥10 img/sec. Record metrics in `log/perf.log`. |
| Regression | Snapshot known results for sample assets to detect drift when models/configs change. |

## Datasets
- Curate 100–200 representative images (electronics, food, documents, mixed scenes) with ground-truth annotations.
- Store fixtures under `tests/fixtures/` with metadata JSON for expected labels and OCR text.

## Acceptance Targets
- Category detection accuracy ≥80% on curated set.
- OCR accuracy ≥85% for Chinese/English text.
- Search latency P95 ≤1s for 5k-record SQLite database.
- CLI ingestion completes without errors on 500-image batch.

## Reporting
- Log test results and benchmarks in the PR summary.
- Update `AI_TASK_TRACKER.md` with pass/fail status per scenario.
- Archive detailed findings or anomalies in `phase_final/research/REVIEW_REPORT_ARCHIVE.md` when necessary.
