# Dependency Report — Coding AI Snapshot (2025-11-12)

## Phase 1 Baseline
| Category | Package | Version | Purpose |
|----------|---------|---------|---------|
| Web | fastapi | 0.121.1 | API service |
| | uvicorn | 0.38.0 | ASGI server |
| | streamlit | 1.51.0 | MVP UI |
| Data | sqlalchemy | 2.0.44 | ORM |
| | pydantic | 2.11.10 | Validation |
| AI | torch | 2.9.1 | DL runtime |
| | torchvision | 0.24.1 | Vision helpers |
| | transformers | 4.57.1 | SigLIP/BLIP wrappers |
| | sentence-transformers | 5.1.2 | Semantic search |
| OCR | paddlepaddle | 3.2.1 | OCR backend |
| | paddleocr | 3.3.1 | Text extraction |
| Utilities | pillow | 11.3.0 | Image ops |
| | numpy | 2.3.4 | Numeric ops |
| | requests | 2.32.3 | HTTP |
| | aiofiles | 24.1.0 | Async IO |
| | python-multipart | 0.0.20 | Upload handling |

## Phase Final Add-ons (optional)
- `psycopg2-binary` 2.9.9 + `pgvector` 0.2.5 — PostgreSQL vector stack.
- `loguru` 0.7.3 + `prometheus-client` 0.23.1 — logging/metrics.

> UI Note: Streamlit remains the sole frontend framework; no alternate production UI dependencies are planned.

All dependencies are managed via `uv`; update `pyproject.toml` and `uv.lock` when versions change.
