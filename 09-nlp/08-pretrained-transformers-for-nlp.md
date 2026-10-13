# Chapter N.8 — Contextual Embeddings and Pretrained Transformers for NLP (ELMo → BERT family → sentence embeddings)

[← NLP index](README.md) · Prev [N.7](07-sequence-labeling-ner-pos.md) · Next [N.9 Seq2seq tasks](09-seq2seq-mt-summarization-qa.md) · Priority 🔴 · Notebook: [`nlp_2_transformers.ipynb`](../notebooks/nlp_2_transformers.ipynb) (GPU; **not run by the author** except where noted) · Architecture details: [Ch 5.4](../05-deep-learning/04-attention-transformers.md)

## 1. Identity & problem
Replace one static vector per word (N.5) with representations that depend on the **whole sentence**, learned by pretraining a large network on unlabeled text, then **transfer** to any task by fine-tuning or by using frozen features. This "pretrain → fine-tune" paradigm (2018) replaced task-specific architectures and feature engineering across NLP.

## 2. Lineage and what each step fixed
| Year | Model | Idea | Limitation it fixed / left |
|---|---|---|---|
| 2013–14 | word2vec, GloVe | Static word vectors | One vector per word; no polysemy |
| 2015 | Semi-supervised sequence learning (Dai & Le) | Pretrain LSTM LM, fine-tune | Early; small scale |
| 2018 | **ELMo** (Peters et al.) | Contextual features from bidirectional LSTM LMs | Used as frozen features; LSTM sequential |
| 2018 | **ULMFiT** (Howard & Ruder) | Fine-tune a pretrained LM with discriminative LRs | Established transfer recipe |
| 2018 | **GPT** (Radford et al.) | Transformer decoder, fine-tune | Left-to-right only |
| 2018 | **BERT** (Devlin et al.) | Bidirectional Transformer encoder, **masked LM** + next-sentence prediction | 512-token limit; MLM mismatch (`[MASK]` only in pretraining) |
| 2019 | RoBERTa (more data, no NSP, dynamic masking), ALBERT (parameter sharing), DistilBERT (distillation: ~40% smaller, ~60% faster, retains most accuracy per authors), ELECTRA (replaced-token detection, sample-efficient), XLNet | Better training recipes/efficiency | Still encoders with fixed windows |
| 2019 | **XLM-R**, mBERT | Multilingual pretraining over 100 languages | "Curse of multilinguality", low-resource gaps |
| 2019 | Sentence-BERT (Reimers & Gurevych) | Siamese bi-encoder → fast semantic similarity | Needs task-fine-tuned encoder |
| 2019–20 | T5, BART | Encoder–decoder text-to-text pretraining | Larger |
| 2020–now | GPT-3-style LLMs, instruction-tuned models | Few-shot prompting, generation (Section 10) | Cost, reliability |
| Domain variants | BioBERT, SciBERT, ClinicalBERT, LegalBERT, FinBERT, CodeBERT | Continued pretraining on domain text | Need domain corpus |

## 3. Intuition
Pretraining forces the model to predict masked or next words, so it learns syntax, semantics and world knowledge in its weights. A static embedding gives "bank" one vector; BERT gives different vectors depending on neighbors because every layer mixes information through attention.

## 4. Mathematics and mechanics
- **Input:** WordPiece tokens + `[CLS]`, `[SEP]`; embedding = token + position + segment; Transformer encoder (Ch 5.4) with bidirectional self-attention.
- **Masked LM objective:** choose ~15% of tokens; of those 80% → `[MASK]`, 10% random token, 10% unchanged; loss `−Σ_{masked} log P(x_i | x_{\setminus masked})`.
- **Fine-tuning heads:** sequence classification `softmax(W h_[CLS])`; token classification `softmax(W h_i)` per token (NER); span extraction (QA) `start/end logits`; sentence-pair tasks via `[CLS] A [SEP] B [SEP]` (NLI, paraphrase).
- **Parameter counts:** BERT-base ≈ 110M (12 layers, hidden 768, 12 heads), BERT-large ≈ 340M; DistilBERT-base ≈ 66M. Per-layer ≈ `12d²` weights ⇒ `12·12·768² ≈ 85M` non-embedding parameters + ≈23M embeddings (30,522 × 768).
- **Fine-tuning recipe:** LR 2e-5 to 5e-5, 2–4 epochs, batch 16–32, warmup ~6–10%, linear decay, weight decay 0.01, max length 128–512. Small datasets are unstable across seeds — report mean ± std over several seeds.
- **Sentence embeddings:** mean-pool token vectors, train with contrastive/triplet or NLI objectives so cosine similarity is meaningful (raw BERT `[CLS]` vectors are poor for similarity). Retrieval: encode corpus once, cosine/ANN search (Ch 2.5, RAG Ch G.4).
**Worked example (masked LM).** Executed with `bert-base-uncased`: *"The capital of France is [MASK]."* → `paris` 0.417, `lille` 0.071, `lyon` 0.063. *"She was [MASK] to hear the good news."* → `glad` 0.442, `eager` 0.085, `relieved` 0.059. The model's distribution over words in context is exactly what pretraining optimizes; plausible but not always "correct" alternatives (lille, lyon) show it models likelihood, not truth.
**Executed WordPiece/special tokens:** `hello world` → ids `[101, 7592, 2088, 102]` = `[CLS] hello world [SEP]`.

## 5. Using pretrained models — the decision ladder
1. **Frozen sentence embeddings + logistic regression/k-NN** (minutes, CPU/GPU light, strong with little data).
2. **Fine-tune a small encoder** (DistilBERT/MiniLM/ELECTRA-small) — best accuracy/latency trade-off for most classification, NER and pairwise tasks.
3. **Fine-tune a base/large encoder** (RoBERTa, DeBERTa, XLM-R) when accuracy is critical and compute allows.
4. **Distill** the large model into a small one for serving; **quantize** (Ch G.5).
5. **Prompt an LLM** (few-shot/zero-shot) or use RAG when labels are scarce, tasks change often, or generation is needed (Section 10).
Encoders (BERT-style) still excel on cost-efficient understanding tasks (classification, NER, reranking, embeddings); decoder LLMs dominate generation and flexible instruction following.

## 6. Notebook experiments (what to run and what to expect)
`nlp_2_transformers.ipynb` (Colab T4):
| § | Experiment | What to look for | Your result |
|---|---|---|---|
| 1 | WordPiece tokenization | Rare words split into pieces; matches Section N.1 outputs | _run_ |
| 2 | "bank" in finance vs river contexts at layers 0, 8, 12 | Within-sense cosine > between-sense cosine in upper layers; layer 0 (static) shows no gap. (A quick single-sentence-pair test I ran earlier gave a *counter-intuitive* result — finance-vs-river 0.651 > finance-vs-finance 0.600 at the last layer — because raw BERT vectors are anisotropic and sentence-specific; the notebook averages over 4+4 sentences and compares layers for a fairer test) | _run_ |
| 3 | Fill-mask + extractive QA | QA span for answerable questions; an unanswerable question still gets a span (SQuAD-v1 models can't abstain) | _run_ |
| 4 | Pretrained NER | Entities PER/ORG/LOC/MISC in a sentence | _run_ |
| 5 | MiniLM semantic search vs BM25 (precision@10 over 300 queries) | BM25 reference **0.432**; dense retrieval typically helps paraphrase queries, may tie/lose on keyword-heavy short queries | _run_ |
| 6 | SST-2: TF-IDF+LR vs MiniLM+LR vs fine-tuned DistilBERT | Expect fine-tuned transformer to lead by several points | _run_ |

## 7. When to use / not
Use: any text understanding task with moderate labels; semantic search/dedup/clustering via sentence embeddings; domain adaptation via continued pretraining; multilingual transfer (XLM-R). Avoid: tiny CPU-only edge cases with strict latency (use TF-IDF or distilled/quantized models), very long documents without chunking/long-context encoders (Longformer, BigBird), tasks where you cannot audit bias, and "BERT for everything" when a linear baseline is within noise.

## 8. Strengths / weaknesses
+ Large accuracy gains, transfer from unlabeled data, one architecture for many tasks, rich ecosystem. − Compute/latency/memory, fixed context (512), seed variance on small data, opaque, encode biases, domain shift, catastrophic forgetting in sequential fine-tuning, tokenization mismatch pitfalls.

## 9. Implementation (version-robust skeleton; full code in the notebook)
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
enc = tok(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
loss = model(**enc, labels=labels).loss; loss.backward()          # AdamW + linear warmup/decay, grad clipping 1.0
# frozen features alternative:
from sentence_transformers import SentenceTransformer
emb = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').encode(texts, normalize_embeddings=True)
```
Practical tips: mixed precision (`torch.autocast`), dynamic padding, sort by length for inference, `max_length` ≤ model limit, freeze lower layers or use LoRA (Ch G.5) for memory, early stopping on validation F1, save tokenizer with the model.

## 10. Evaluation
Task metrics (accuracy/macro-F1/entity-F1/EM-F1 for QA), multiple seeds, error analysis by slice, calibration, robustness (typos, paraphrase, negation — CheckList, Ribeiro et al. 2020), out-of-domain test, efficiency (latency, memory, cost), fairness. Benchmarks: GLUE/SuperGLUE (saturated), SQuAD, CoNLL-2003, XNLI, MTEB (embeddings) — use for orientation, not for deciding your product.

## Interview Q&A
- *What does BERT pretraining do?* Masked-LM (and originally NSP) on large unlabeled corpora to learn bidirectional contextual representations.
- *BERT vs GPT?* Bidirectional encoder for understanding vs causal decoder for generation.
- *Why [CLS]?* A learned position whose final state summarizes the sequence for classification heads.
- *Frozen vs fine-tuned?* Frozen is cheap and stable; fine-tuning gives higher accuracy at more cost.
- *Why do raw BERT vectors perform poorly for similarity?* Anisotropy and no similarity objective; use Sentence-BERT-style training.
- *DistilBERT?* Knowledge-distilled smaller BERT: faster, slightly less accurate.

## Exercises
Fine-tune three seeds of DistilBERT and report mean±std; plot layer-wise probing accuracy (e.g. POS) from frozen hidden states; compare MiniLM+LR vs TF-IDF+LR at 50/500/5,000 labeled examples (learning curves); distill a larger model into a TF-IDF linear model and compare.

## Further reading
Peters et al. (2018) ELMo; Howard & Ruder (2018) ULMFiT; Devlin et al. (2018) BERT; Liu et al. (2019) RoBERTa; Sanh et al. (2019) DistilBERT; Clark et al. (2020) ELECTRA; Conneau et al. (2020) XLM-R; Reimers & Gurevych (2019) Sentence-BERT; Rogers, Kovaleva & Rumshisky (2020) "A primer in BERTology"; Hugging Face NLP course.
