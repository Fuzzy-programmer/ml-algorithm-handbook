# Chapter 6.8 — Multimodal Models (Vision-Language, Audio, Video)

[← Section index](README.md) · Prev [6.7](07-diffusion-models.md) · Next [6.9 Tool calling](09-tool-calling.md) · Priority 🟠 · Code: [`genai_demos.py`](../code/genai_demos.py), [`deep_learning_numpy.py`](../code/deep_learning_numpy.py)

## 1. Identity & problem
Models that process and/or generate more than one modality — text, images, audio, video, documents, charts, sensor data — in a shared representation. Tasks: image captioning, visual question answering, document/chart understanding (OCR-free), speech recognition and synthesis, video understanding, text-to-image/video, GUI/computer-use agents.

## 2. Evolution
| Stage | Idea | Landmarks | Limits |
|---|---|---|---|
| Separate unimodal models + glue | Captioning CNN→LSTM (Show and Tell 2015), VQA models | — | Weak alignment |
| **Contrastive alignment** | Embed image and text in one space | **CLIP** (Radford et al. 2021), ALIGN | Global embedding; poor fine detail/counting |
| **Vision encoder + LLM** | Project image features into the LLM's token space | Flamingo (2022), BLIP-2 (2023), **LLaVA** (2023), later commercial multimodal LLMs | Hallucinated visual details |
| **Natively multimodal training** | One Transformer over interleaved text/image/audio tokens | Recent frontier systems (details vary by vendor) | Compute, data, evaluation |
| Speech | Whisper (2022: weakly supervised ASR at scale), neural TTS (VITS, VALL-E-style codec LMs) | | Accents/noise, latency |
| Generation | Text-to-image (diffusion, Ch 6.7), text-to-video, audio generation | | Consistency, control, safety |
| Any-to-any / agents | Unified models; screen/GUI understanding for computer use | | Reliability |

## 3. Core mechanisms
- **Vision Transformer tokenization:** split an image into patches, embed linearly, add position → Transformer (Ch 5.4). **Executed arithmetic:** 224×224 image with 16×16 patches → **196** tokens (+1 CLS); 336×336 with 14×14 patches → **576** tokens. Higher resolution multiplies visual tokens (and cost/latency); many systems tile or resize large images.
- **Contrastive pretraining (CLIP):** batch of N (image, caption) pairs; similarity matrix `S_ij = cos(v_i,t_j)/τ`; symmetric cross-entropy pulls matching pairs together. InfoNCE worked example in Ch 5.7 (loss 0.1446).
- **Zero-shot classification:** embed class prompts ("a photo of a {class}") and pick the highest cosine similarity. **Worked example:** image embedding (0.8, 0.6); text embeddings "cat" (0.9, 0.436), "dog" (−0.5, 0.866) normalized → cosines ≈ 0.98 vs 0.12 → "cat".
- **Connectors:** linear/MLP projector or cross-attention/Q-Former mapping vision features into LLM embedding space; the LLM then treats image tokens like words.
- **Audio:** waveform → log-mel spectrogram (or learned codec tokens) → encoder; ASR decoders output text; speech LMs predict codec tokens.
- **Video:** frames sampled over time + temporal modeling; token cost grows with frames×patches — sampling strategy is critical.

## 4. When to use / not
Use: documents with layout/tables/charts, screenshots, product photos, medical or industrial imagery (with domain validation), voice interfaces, video moderation/search, accessibility. Prefer a dedicated model when you need precision at scale: OCR engines for exact text, object detectors for counting/boxes, ASR specialists for noisy audio. Avoid trusting multimodal LLMs for exact counting, small text in low resolution, fine spatial reasoning, medical diagnosis without clinical validation.

## 5. Strengths / weaknesses
+ Unified interface, zero-shot flexibility, handles messy documents, reduces pipeline glue. − Hallucination about image content, resolution/token trade-offs, bias, privacy (faces, documents), adversarial images/typographic attacks, high cost per image/frame.

## 6. Hyperparameters / usage choices
Image resolution and tiling, number of frames, audio chunking, modality-specific prompts (ask for coordinates/quotes), output schema, which encoder/connector (when training), temperature, max tokens; for retrieval: embedding model, index per modality.

## 7. Multimodal RAG
Embed images/page screenshots (CLIP-style or document-page embeddings) alongside text; retrieve pages; pass images (or OCR'd text + layout) to a multimodal LLM. Evaluate retrieval separately (Recall@K on page-level gold) and answer faithfulness (Ch 6.4).

## 8. Implementation templates (not executed)
Calling a hosted multimodal model with an image (Anthropic Python SDK; check current docs for supported media types/limits):
```python
import anthropic, base64
client = anthropic.Anthropic()
img = base64.standard_b64encode(open("invoice.png", "rb").read()).decode()
msg = client.messages.create(model="claude-opus-5-5", max_tokens=1024, messages=[{"role": "user", "content": [
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}},
    {"type": "text", "text": "Extract vendor, date and total as JSON."}]}])
```
Local CLIP zero-shot (Hugging Face `transformers.CLIPModel`) and Whisper (`openai/whisper-*` checkpoints) are common open baselines.

## 9. Evaluation
Task benchmarks (VQA, DocVQA/ChartQA, MMMU-style multi-discipline, video QA) — useful but contaminated/saturating; build **your own labeled set** from real inputs (scans, phone photos, screenshots) with field-level accuracy; measure hallucination rate (unsupported claims), robustness to rotation/blur/compression, latency/cost per image, subgroup performance, privacy compliance. For ASR: word error rate by accent/noise condition.

## 10. Case study (template)
Invoice processing: baseline OCR + regex/layout rules; multimodal LLM with a JSON schema; compare field-level exact-match on 300 labeled invoices across scan quality tiers; add validation (totals must equal line-item sums), human review for low-confidence cases; track cost per page.

## Interview Q&A
- *How does CLIP enable zero-shot classification?* Shared embedding space trained contrastively; classify by similarity to text prompts.
- *How are images fed to an LLM?* Patch-tokenized, encoded by a vision Transformer, projected into the LLM's embedding space as extra tokens.
- *Why is video expensive?* Tokens scale with frames × patches; sampling and compression are essential.
- *Failure modes?* Hallucinated details, counting errors, small-text misreads, bias, prompt injection through images.

## Exercises
Compute visual token counts for several resolutions/patch sizes; build a 50-image labeled set and compare an OCR pipeline vs a multimodal LLM; implement CLIP-style retrieval with an open model and measure recall@5.

## Further reading
Dosovitskiy et al. (2020) ViT; Radford et al. (2021) CLIP; Alayrac et al. (2022) Flamingo; Li et al. (2023) BLIP-2; Liu et al. (2023) LLaVA; Radford et al. (2022) Whisper; Yin et al. (2023) "A survey on multimodal LLMs".
