# SigLIP + BLIP Integration — Coding AI Notes

## Rationale
- RTMDet was removed due to Python 3.12 incompatibility (mmcv) and limited label flexibility.
- SigLIP provides multilingual zero-shot classification; BLIP generates descriptive captions.
- Combination covers both structured labels and narrative summaries, enabling richer search and annotation.

## Usage Pattern
1. Load SigLIP (`google/siglip-base-patch16-224-i18n`) and BLIP (`Salesforce/blip-image-captioning-base`).
2. For each image: compute candidate label scores via SigLIP, generate caption via BLIP.
3. Persist outputs alongside OCR, embeddings, and metadata.

## Implementation Snippet
```python
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224-i18n")
model = AutoModel.from_pretrained("google/siglip-base-patch16-224-i18n")
blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def analyze(image: Image.Image, labels: list[str]) -> dict[str, Any]:
    inputs = processor(text=labels, images=image, padding=True, return_tensors="pt")
    probs = torch.sigmoid(model(**inputs).logits_per_image[0])
    caption = blip_model.generate(**blip_proc(image, return_tensors="pt"))
    return {
        "scores": dict(zip(labels, probs.tolist())),
        "caption": blip_proc.decode(caption[0], skip_special_tokens=True),
    }
```

## Tips
- Maintain a configurable label set per user; fallback to default categories when none provided.
- Cache model weights in `models/`; reuse processors across requests.
- Monitor GPU/CPU usage—consider batching or mixed precision if performance is constrained.

Document any modifications to this pipeline in `AI_DECISION_RECORD.md`.
