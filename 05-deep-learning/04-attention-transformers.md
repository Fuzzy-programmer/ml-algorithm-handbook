# Chapter 5.4 — Attention and Transformers (BERT, GPT-style decoders, Encoder–Decoder)

[← Roadmap](../README.md) · Prev [5.3](03-sequence-models.md) · Next [5.5 Generative models](05-generative-ae-vae-gan.md) · Priority 🔴 · Code: [`deep_learning_numpy.py`](../code/deep_learning_numpy.py)

## 1. Identity & problem
Architecture that models relations among all positions of a sequence (or set) in parallel via **attention**, replacing recurrence. Foundation of modern NLP, vision (ViT), speech, protein models and all LLMs (Ch 6.2).

## 2. History
Bahdanau et al. (2014) additive attention in seq2seq; Luong et al. (2015) multiplicative attention → **"Attention Is All You Need" (Vaswani et al., 2017)**: Transformer encoder–decoder, parallel training, state-of-the-art translation → **GPT** (Radford et al. 2018, decoder-only, generative pretraining), **BERT** (Devlin et al. 2018, encoder-only, masked-LM), GPT-2/3 (2019/2020), **T5** (Raffel et al. 2019, text-to-text enc–dec), ViT (2020), scaling laws (Kaplan et al. 2020; Hoffmann et al. 2022 "Chinchilla"), RoPE (Su et al. 2021), FlashAttention (Dao et al. 2022), GQA/MQA, Mixture-of-Experts. Limitation motivating later work: attention cost `O(n²)` in sequence length.

## 3. Intuition
Every token asks a question (query), every token advertises what it holds (key) and carries content (value). Each token's output is a weighted mix of all values, with weights given by query–key match. Stacks of this plus feed-forward layers build contextual meaning.

## 4. Math
Scaled dot-product attention: `Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V`. Scaling prevents large dot products from saturating softmax. **Multi-head:** run h attention heads in subspaces of size `d_k=d/h`, concatenate, project (`W_O`). Self-attention: Q,K,V are linear projections of the same input X.
**Worked example (computed in code).** `Q=[[1,0],[0,1]]`, `K=[[1,0],[0,1],[1,1]]`, `V=[[1,0],[0,2],[3,3]]`, `d_k=2`. Query 1 scores `[1,0,1]/√2` → softmax weights **[0.401, 0.198, 0.401]** → output `0.401[1,0]+0.198[0,2]+0.401[3,3] = [1.604, 1.599]`. Query 2 weights `[0.198,0.401,0.401]` → output `[1.401, 2.006]`.
**Block (pre-norm):** `x ← x + MHA(LN(x))`; `x ← x + FFN(LN(x))`, FFN = `W₂·GELU(W₁x)` with hidden ≈ 4d. **Positional information** is injected (sinusoidal, learned, relative, RoPE, ALiBi) since attention is permutation-equivariant. **Masks:** causal mask (upper triangle −∞) for autoregressive decoders — verified in code: with the mask, token 0 attends only to itself and the upper triangle weights are exactly 0.
Variants: **encoder-only** (BERT, bidirectional; classification, embeddings, retrieval), **decoder-only** (GPT; next-token generation; the LLM default), **encoder–decoder** (T5, original Transformer; translation/summarization).
Parameters per layer ≈ `12d²` (4d² attention + 8d² FFN).
**Costs:** training/inference attention `O(n²·d)` time, `O(n²)` memory per head — at n=4,096 one head's attention matrix has 16.8M entries (≈33.5 MB in fp16) per layer-head, which is why FlashAttention (IO-aware exact attention), sparse/linear attention, sliding windows, GQA and KV caching matter. Generation: KV cache makes each new token `O(n·d)` instead of recomputing.

## 5. Training
Next-token or masked-token objective on massive corpora; AdamW with warmup + decay; mixed precision (bf16); gradient clipping; large batches; data/model/pipeline parallelism. Fine-tuning and adaptation: Ch 6.5.

## 6. Evolution & trade-offs
| Step | Gain | Cost |
|---|---|---|
| RNN → attention-augmented RNN | Long-range access | Still sequential |
| → Transformer | Parallel training, scales with data/compute | Quadratic attention, data hungry |
| Encoder-only (BERT) | Strong understanding/embeddings | Not generative |
| Decoder-only (GPT) | Unified generation, in-context learning | Unidirectional context |
| + MoE / GQA / FlashAttention / long-context tricks | Capacity/efficiency | Engineering complexity, routing instability |

## 7. When to use / not
Use: text, code, speech, multi-modal, large pretraining + fine-tuning, anything with long-range dependencies and enough data/compute (or a pretrained checkpoint). Avoid: small tabular data, tiny datasets with no pretrained model, hard real-time on tiny devices, very long sequences without efficient attention.

## 8. Hyperparameters
Layers L, model dim d, heads h, FFN ratio, context length, vocab/tokenizer, dropout, LR/warmup/schedule, weight decay, batch tokens, positional scheme, norm placement (pre vs post), initialization.

## 9. Implementation
NumPy (verified): scaled dot-product attention and a causal multi-head attention in `deep_learning_numpy.py`:
```python
S = Q @ K.T / np.sqrt(dk); S += np.triu(np.full((n, n), -1e9), 1)       # causal mask
A = np.exp(S - S.max(-1, keepdims=True)); A /= A.sum(-1, keepdims=True); out = A @ V
```
Hugging Face template (not executed here):
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("bert-base-uncased"); m = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
```
Evaluation (shape check, measured): multi-head output `(5,16)` for 5 tokens, 4 heads of size 4, attention tensor `(4,5,5)`.

## 10. Evaluation
Perplexity (LM), task metrics (accuracy/F1, ROUGE/BLEU/BERTScore, exact match), human/LLM-judged quality (Ch 6.11), calibration, robustness, latency (time-to-first-token, tokens/s), memory (KV cache size), cost per request. Hold-out must exclude contamination from pretraining data when benchmarking.

## 11. Case study (template)
Support-ticket classification: TF-IDF+logistic baseline vs fine-tuned small encoder vs prompted LLM; compare macro-F1, cost and latency; the linear baseline is often within a few points for easy label sets, transformers win on nuance and low-data multilingual settings (check on your data).

## 12. Comparison
Transformers vs RNN/TCN: parallelism + long-range. Transformers vs CNN for vision: less bias, more data. Transformers vs SSMs: quality/ecosystem vs linear-time long context.

## Interview Q&A
- *Why divide by √d_k?* Dot products grow with dimension, pushing softmax into saturation with tiny gradients.
- *Why multiple heads?* Different subspaces/relations attended in parallel.
- *Why positional encodings?* Attention alone ignores order.
- *BERT vs GPT?* Bidirectional masked-LM encoder vs causal next-token decoder.
- *Why is long context expensive?* O(n²) attention and O(n) KV cache per token.
- *What does the causal mask do?* Prevents positions from attending to the future during training/generation.

## Exercises
Implement a full pre-norm Transformer block in NumPy; count parameters for d=768, L=12 (compare with ~85M in non-embedding weights: 12·12·768² ≈ 85M); measure attention memory growth with n.

## Further reading
Vaswani et al. (2017); Devlin et al. (2018); Radford et al. (2018/2019); Brown et al. (2020); Raffel et al. (2019); Kaplan et al. (2020); Hoffmann et al. (2022); Dao et al. (2022); Jay Alammar "The Illustrated Transformer"; Karpathy "Let's build GPT".
