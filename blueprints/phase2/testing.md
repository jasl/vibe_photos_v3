# Phase 2 Testing Draft

## 1. Automated Test Plan
- **Compatibility Smoke Tests**: Ensure detector/OCR warmups run cleanly.
  - `tests/unit/test_compatibility_guard.py` (new) asserts that `build_detector()` finishes without emitting `DeprecationWarning` when dependencies are stubbed.
  - Add analogous test for `PaddleOCREngine.warmup()` once implemented.
- **Worker Unit Tests**: Simulate queue ingestion and worker loops with fake models to verify batching, dedupe, and retry logic.
- **Integration Tests**: Spin up ingestion worker + cache writer against small fixture dataset to verify queue-to-cache flow.
- **UI Regression**: Extend Streamlit screenshot tests to cover cluster labeling interface.

## 2. Manual Checklist (run before committing)
1. `uv run python -m pip check`
2. `uv run python -W error::DeprecationWarning process_dataset.py --dry-run`
3. `uv run pytest`
4. Review `log/process_dataset.log` and `log/perf.log` for new warnings.

## 3. Metrics Validation
- Track OCR skip ratio, detection latency per image, and queue depth in `log/perf.log`.
- Ensure cache manifest contains matching schema hash before importing into SQLite.

## 4. Tooling Notes
- Consider using `pytest-warnings` plugin to fail the suite on new `DeprecationWarning`/`FutureWarning` categories.
- Add tox/uv tasks for worker smoke tests once the ingestion service stabilizes.
