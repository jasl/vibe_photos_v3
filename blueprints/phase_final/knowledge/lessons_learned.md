# Lessons Learned — Coding AI Snapshot

## Product Insights
- Narrow focus on self-media creators yields clearer requirements than a generic photo manager.
- AI should augment, not replace, users: high-confidence predictions auto-apply; medium confidence requires confirmation; low confidence defers to humans.
- Batch operations (apply label to similar assets) deliver the biggest productivity gains.

## Engineering Takeaways
- Start simple (SQLite + SigLIP/BLIP) and layer complexity only after validation.
- Avoid premature abstraction—direct implementations with thorough tests outperform over-engineered patterns.
- Cache everything: thumbnails, detections, OCR results, embeddings.

## Process Improvements
- Ship iteratively; prefer weekly increments over large, infrequent drops.
- Collect metrics/feedback continuously and adjust roadmaps accordingly.
- Document decision rationale immediately to keep future coding AIs aligned.
