# Chapter 5.7 — Self-Supervised and Representation Learning

[← Roadmap](../README.md) · Prev [5.6](06-gnn.md) · Next [6.1 Time series](../06-specialized-ml/01-time-series.md) · Priority 🔴 · Code: [`deep_learning_numpy.py`](../code/deep_learning_numpy.py)

## 1. Identity & problem
Learn useful representations from **unlabeled** data by inventing a prediction task from the data itself. Labels are expensive; raw data is abundant. The learned encoder is then reused (transfer learning, Ch 6.5) with few labels.

## 2. History and families
| Family | Pretext task | Landmarks |
|---|---|---|
| **Predict context** | Predict neighbors/words | word2vec (Mikolov et al. 2013), GloVe, skip-gram with negative sampling |
| **Autoregressive** | Predict the next token | GPT (2018→), PixelCNN, wav2vec-style |
| **Masked prediction** | Reconstruct masked parts | BERT (2018), MAE (He et al. 2021), BEiT, data2vec |
| **Contrastive** | Pull augmented views of the same item together, push others apart | CPC / InfoNCE (van den Oord 2018), MoCo (He 2019), **SimCLR** (Chen 2020), **CLIP** (Radford 2021; image–text pairs) |
| **Non-contrastive / self-distillation** | Match representations without negatives | BYOL (Grill 2020), SimSiam, DINO (Caron 2021), VICReg |
| **Generative** | Reconstruct/denoise | autoencoders, VAEs, diffusion (Ch 5.5, 6.7) |
| Earlier ideas | Pseudo-tasks (rotation prediction, jigsaw), restricted Boltzmann machines/deep belief nets pretraining (Hinton 2006), semi-supervised "pseudo-labeling" (Ch 6.6) | — |
The progression is driven by removing costs: manual labels → negative pairs (BYOL) → hand-crafted augmentations (masking/AR objectives that scale cleanly).

## 3. Intuition
Hide part of the data and make the model fill it in, or show two versions of the same thing and ask the model to recognize they match. To succeed, the model must learn structure—grammar, objects, physics-like regularities.

## 4. Math
**InfoNCE / NT-Xent:** for anchor `i`, positive `i⁺`, set of negatives `N`: `L_i = −log [ exp(sim(z_i,z_{i⁺})/τ) / ( exp(sim(z_i,z_{i⁺})/τ) + Σ_{j∈N} exp(sim(z_i,z_j)/τ) ) ]`, `sim` = cosine similarity, temperature τ. It lower-bounds mutual information between views.
**Worked example (computed).** Unit vectors: anchor (1,0), positive (0.994,0.110), negatives (0,1) and (−1,0); τ=0.5. Similarities/τ: positive 1.988, negatives 0 and −2. Numerator `e^{1.988}=7.30`; denominator `7.30+1+0.135=8.435`; loss `=−ln(7.30/8.435)=` **0.1446**. Make the negative (−1,0) become (1,0) (identical to the anchor): the denominator grows and the loss rises — the objective penalizes confusing others with the anchor.
**Masked LM:** minimize `−Σ_{masked} log p(x_i | x_{\setminus masked})` (≈15% masked in BERT). **Next-token:** `−Σ log p(x_t|x_{<t})`.
Collapse (all outputs equal) is avoided by negatives (contrastive), momentum/stop-gradient (BYOL/SimSiam), or variance/covariance regularization (VICReg).
Assumptions: augmentations preserve the semantics you care about; the pretext task requires the features you need.

## 5. Training & cost
Pretraining is expensive (large batches for SimCLR, big corpora for LLMs); downstream use is cheap: linear probe, fine-tune, or k-NN on frozen embeddings. Typical large-scale recipe: huge unlabeled data → pretrain → small labeled set.

## 6. When to use / not
Use: abundant unlabeled data, scarce labels, need general embeddings (search, clustering, recommendation, anomaly detection), domain adaptation (continued pretraining). Often best to **start from a public pretrained model** rather than pretraining yourself. Avoid: when you have plenty of labels and a small domain-specific task (supervised training may suffice), or lack compute for meaningful pretraining.

## 7. Strengths / weaknesses
+ Exploits unlabeled data, produces transferable features, foundation of modern AI. − Compute-hungry, augmentation/pretext design sensitive, possible bias/privacy issues from web-scale data, representation quality hard to evaluate.

## 8. Hyperparameters
Temperature τ, batch size/number of negatives, augmentation strength/composition, mask ratio, projection head size, momentum coefficient (MoCo/BYOL), learning rate and schedule, embedding dimension.

## 9. Implementation
NumPy InfoNCE (verified):
```python
z = z / np.linalg.norm(z, axis=1, keepdims=True); sim = z @ z.T / tau
loss = -np.log(np.exp(sim[a, p]) / (np.exp(sim[a, p]) + sum(np.exp(sim[a, j]) for j in negs)))
```
Practical: use a pretrained encoder (e.g. `sentence-transformers`, CLIP, DINOv2) and train a light head. sentence-transformers template (not executed here):
```python
from sentence_transformers import SentenceTransformer
emb = SentenceTransformer("all-MiniLM-L6-v2").encode(["text one", "text two"], normalize_embeddings=True)
```

## 10. Evaluation
Linear-probe accuracy, k-NN accuracy on frozen features, few-shot fine-tuning curves, retrieval recall@k, clustering quality, robustness/shift tests, and downstream task metrics. Always compare to a supervised baseline trained on the same labeled subset and to a *random-init* encoder.

## 11. Case study (template)
Product search with 10k labeled queries: baseline BM25; add a pretrained sentence encoder for dense retrieval (hybrid), fine-tune with contrastive loss on click pairs; evaluate recall@k and NDCG; see Ch 6.4 for the RAG context.

## 12. Comparison
Contrastive: strong but needs negatives/large batches. Masked: simple, scalable (BERT/MAE). Autoregressive: supports generation (LLMs). Non-contrastive: avoids negatives but needs anti-collapse tricks.

## Interview Q&A
- *Why is it "self"-supervised?* Labels come from the data itself.
- *Role of temperature in InfoNCE?* Controls how sharply hard negatives are penalized.
- *How does BYOL avoid collapse without negatives?* Predictor + momentum (EMA) target network with stop-gradient (the exact reason is still analyzed).
- *Linear probing vs fine-tuning?* Measure representation quality vs maximize task performance.

## Exercises
Compute the InfoNCE loss for a batch of 4 pairs by hand; implement skip-gram negative sampling on a toy corpus; compare linear-probe accuracy of random vs pretrained embeddings.

## Further reading
Mikolov et al. (2013); Devlin et al. (2018); van den Oord et al. (2018); Chen et al. (2020) SimCLR; Grill et al. (2020) BYOL; He et al. (2021) MAE; Radford et al. (2021) CLIP; Balestriero et al. (2023) "A Cookbook of Self-Supervised Learning".
