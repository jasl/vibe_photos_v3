# Configuration Contract — Vibe Photos

This directory holds runtime configuration files that coding AIs and operators must keep in sync. Treat every file here as part of the repository contract: changes require documentation updates and usually a lockfile refresh.

## 1. Files & Purpose
| File | Purpose |
|------|---------|
| `candidates.yaml` | Canonical label set consumed by `src/core/detector.py` when no overrides are provided. |
| `candidates.example.yaml` | Safe-to-commit template demonstrating the expected structure; copy to `candidates.yaml` when bootstrapping a new environment. |

> **Do not** commit environment-specific secrets (API keys, DSNs) in this directory. Place them in `.env` files that stay untracked.

## 2. `candidates.yaml` Schema
The detector expects a YAML document with the following shape:

```yaml
version: 1
updated_at: 2025-02-15
labels:
  - key: food.recipe
    name: "Recipe"
    synonyms: ["home cooking", "kitchen"]
  - key: document.invoice
    name: "Invoice"
    synonyms: ["bill", "receipt"]
  - key: tutorial.step-by-step
    name: "Tutorial"
    synonyms: ["how-to", "guide"]
  - key: product.launch
    name: "Product Launch"
    synonyms: ["campaign", "merch"]
```

- `version` — increment whenever the label taxonomy changes.
- `updated_at` — ISO date indicating when the file was last curated.
- `labels` — ordered list used to seed the detector. Each entry contains:
  - `key`: stable identifier used in the database.
  - `name`: human-readable tag displayed in UIs.
  - `synonyms`: optional array to bias embeddings toward specific phrasing.

## 3. Update Workflow
1. Edit `candidates.yaml` and adjust the `version` + `updated_at` fields.
2. Update downstream documentation that references the label set (e.g., blueprint implementation guides) if semantics change.
3. Add migration notes in `AI_TASK_TRACKER.md` when labels are added or removed so future coding AIs understand the impact on search quality.

## 4. Bootstrap Instructions
- On first checkout, copy `candidates.example.yaml` to `candidates.yaml` and tailor it to the engagement’s needs.
- Tests may rely on the example file. If you introduce new labels, update fixture data accordingly.

Keep this contract authoritative—detector modules should never hardcode labels outside this directory.
