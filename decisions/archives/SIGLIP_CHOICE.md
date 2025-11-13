# Archive: SigLIP Adoption Rationale

**Decision Date:** 2024-11-12 (now archived)

## Summary
We replaced RTMDet/CLIP pipelines with the SigLIP + BLIP stack for Phase 1 because of:
- Multi-language support out of the box.
- Higher accuracy (~85% vs ~52% mAP with RTMDet under our constraints).
- Compatibility with Python 3.12 without heavy dependencies (mmcv).

## Historical Notes
| Option | Outcome |
|--------|---------|
| RTMDet | Blocked by mmcv incompatibility on Python ≥3.11. |
| CLIP | English-only and underperformed on object-specific tasks. |
| SigLIP | Chosen; integrates smoothly with Hugging Face ecosystem. |

## Migration Guidance (at time of change)
```bash
uv pip uninstall mmdet mmengine mmcv clip-interrogator
uv pip install transformers==4.57.1 torch==2.9.1
```

Keep this file for reference only—active decisions live in `AI_DECISION_RECORD.md`.
