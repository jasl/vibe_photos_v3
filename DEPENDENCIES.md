# Dependency Manifest — Coding AI Reference

Use this manifest to validate that your environment matches the expected versions for Vibe Photos.

## 1. Python Runtime
- Python **3.12.x** (pinned via `uv python pin 3.12`).

## 2. Phase 1 Baseline Packages
| Category | Package | Version | Purpose |
|----------|---------|---------|---------|
| Web | fastapi | 0.121.1 | REST API service |
| | uvicorn | 0.38.0 | ASGI server |
| | streamlit | 1.51.0 | MVP UI |
| AI Models | torch | 2.9.1 | Core DL framework |
| | torchvision | 0.24.1 | Image utilities |
| | transformers | 4.57.1 | SigLIP/BLIP wrappers |
| | sentence-transformers | 5.1.2 | Semantic search experiments |
| OCR | paddlepaddle | 3.2.1 | OCR backend |
| | paddleocr | 3.3.1 | Text extraction |
| Data | sqlalchemy | 2.0.44 | Persistence layer |
| | pydantic | 2.11.10 | Validation/schemas |
| | numpy | 2.3.4 | Numeric ops |
| | pillow | 11.3.0 | Image handling |

## 3. Phase Final Add-ons
| Category | Package | Version | Usage |
|----------|---------|---------|-------|
| Vector DB | pgvector | 0.2.5 | PostgreSQL vector extension |
| | psycopg2-binary | 2.9.9 | PostgreSQL driver |
| UI | gradio | 5.49.1 | Production UI |
| Monitoring | loguru | 0.7.3 | Enhanced logging |

## 4. Installation Workflow
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip sync requirements.txt              # core deps
uv pip sync requirements-dev.txt          # add dev tooling if needed
```
For GPU builds on NVIDIA hardware:
```bash
uv pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu124
```

## 5. Model Artifacts
- SigLIP: `google/siglip-base-patch16-224-i18n` (~400 MB)
- BLIP: `Salesforce/blip-image-captioning-base` (~990 MB)
- PaddleOCR Chinese package (~200 MB)

## 6. File Layout
```
blueprints/
├── phase1/
│   ├── requirements.txt
│   └── requirements-dev.txt
└── phase_final/
    ├── requirements.txt
    └── requirements-dev.txt
```

Record any dependency adjustments in this file and re-lock the environment so downstream coding AIs inherit a consistent stack.
