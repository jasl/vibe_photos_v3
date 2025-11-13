# Requirements Brief — Coding AI Edition

This brief captures user-facing needs without prescribing implementation details. Treat it as the product contract.

## Mission Statement
Deliver a local-first, AI-assisted photo management assistant for Chinese content creators who maintain thousands of images for reviews, tutorials, and recommendations.

## Target Users
- Independent or small-team creators producing long-form articles or videos.
- Maintain personal libraries spanning electronics, food, documents, portraits, and scenery.
- Need fast recall of niche items (e.g., specific gadget models) and story-driven photo sets.

## Functional Requirements
1. **Photo Understanding**
   - Classify high-level categories and drill down to brand/model when possible.
   - Extract captions and OCR text for mixed-language assets.
2. **Search & Retrieval**
   - Natural-language queries like “latest iPhone photos from last month”.
   - Combine filters (category + time range + tags) with fuzzy matching.
   - Group near-duplicate or similar images.
3. **Bulk Operations**
   - Import and process thousands of images in a single batch.
   - Support incremental updates—only new files require full processing.
   - Export curated collections for publication workflows.
4. **Adaptive Learning**
   - Accept manual corrections; learn user-specific labels over time.
   - Provide transparent suggestions rather than full automation.

## Success Metrics
- Classification accuracy ≥85% on representative creator datasets.
- Search latency ≤500 ms for libraries up to 10k images.
- Ingestion throughput ≥10 images/sec (CPU baseline).
- Users can complete a typical retrieval task within 3 minutes of logging in.

## Non-Goals
- No image editing or advanced manipulation features.
- No built-in social sharing or cloud synchronization during POC phases.
- No promise of full autonomy; human-in-the-loop workflows remain.

Keep this brief intact. Any scope change must be explicitly approved and recorded in the decision logs.
