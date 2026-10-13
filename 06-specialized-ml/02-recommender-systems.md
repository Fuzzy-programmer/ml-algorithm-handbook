# Chapter 6.2 — Recommender Systems: Popularity → CF → Matrix Factorization → Two-Tower → Ranking

[← Roadmap](../README.md) · Prev [6.1](01-time-series.md) · Next [6.3 Reinforcement learning](03-reinforcement-learning.md) · Priority 🟠 · Code: [`recsys_demo.py`](../code/recsys_demo.py)

## 1. Identity & problem
Rank items (movies, products, videos, jobs) for each user from interaction history and side information. Interactions are usually **implicit** (clicks, views, purchases) rather than ratings, sparse, biased by exposure, and non-stationary.

## 2. History and what each stage fixed
| Era | Approach | Contribution | Limitation |
|---|---|---|---|
| Baseline | Popularity / recency | Strong, cheap | No personalization |
| 1990s | Content-based filtering; **user-based CF** (GroupLens, Resnick et al. 1994) | Personalization from similar users | O(users²), sparse overlap |
| 2003 | **Item-based CF** (Sarwar 2001; Amazon, Linden et al. 2003) | Stable item similarities, scalable | Weak on sparse/long-tail |
| 2006–09 | **Matrix factorization** from the Netflix Prize (Koren, Bell & Volinsky 2009), implicit ALS (Hu, Koren & Volinsky 2008), BPR (Rendle 2009), Factorization Machines (Rendle 2010) | Latent taste vectors, handles sparsity, add biases/side features | Cold start, linear interaction |
| 2015– | Deep & neural: Neural CF (He 2017), **YouTube two-stage deep retrieval/ranking** (Covington 2016), **two-tower** retrieval, sequential models (GRU4Rec 2015, SASRec 2018), graph (LightGCN 2020), Transformer-based and generative recommenders | Rich features, sequences, scalable ANN retrieval | Data/infra heavy, offline-online gap |
Industrial architecture: **candidate generation (retrieval) → ranking → re-ranking (diversity, business rules)**.

## 3. Intuition
People who liked what you liked will like what you haven't seen; or: each user and item is a point in a taste space and affinity = dot product.

## 4. Math
- **Explicit MF:** `r̂_ui = μ + b_u + b_i + p_u·q_i`; minimize `Σ(r_ui − r̂_ui)² + λ(‖p_u‖²+‖q_i‖²+b²)` by SGD or ALS.
- **Implicit ALS:** preference `p_ui = 1[r_ui>0]`, confidence `c_ui = 1+α r_ui`; minimize `Σ c_ui (p_ui − x_uᵀy_i)² + λ(‖x_u‖²+‖y_i‖²)`; alternate closed-form ridge solves for users/items.
- **BPR:** maximize `Σ log σ(x_uᵀy_i − x_uᵀy_j)` for observed `i` vs sampled unobserved `j` (pairwise ranking).
- **Two-tower:** `s(u,i) = ⟨f(u), g(i)⟩` with in-batch sampled-softmax; item embeddings indexed with ANN for sub-linear retrieval (Ch 2.5).
**Worked SGD step.** True rating 4; `p=[1,0.5]`, `q=[2,1]` ⇒ `r̂=1·2+0.5·1=2.5`, error `e=1.5`. With lr 0.1 (λ=0): `p ← p + lr·e·q = [1.3, 0.65]`, `q ← q + lr·e·p_old = [2.15, 1.075]` ⇒ new `r̂=1.3·2.15+0.65·1.075=3.494`, much closer to 4.
**Ranking metrics:** Recall/HitRate@K, Precision@K, MRR, `NDCG@K = DCG/IDCG`, `DCG=Σ rel_i/log₂(i+1)`; also coverage, diversity, novelty, calibration.

## 5. Training & cost
ALS `O((n_u+n_i)f²·nnz/…)` per sweep, parallelizable; SGD `O(nnz·f)` per epoch; kNN CF needs `O(n²)` similarities (use sparse/ANN). Serving: precompute embeddings, ANN retrieval (HNSW/IVF) in milliseconds, then a heavier ranker on a few hundred candidates.

## 6. When to use / not
Use: catalogs too big to browse, personalization drives value. Choose by data: little data → popularity/content; moderate interactions → item-kNN/MF; large-scale with rich features → two-tower + ranker; sessions → sequence models. Avoid: optimizing offline accuracy alone (feedback loops, popularity bias, filter bubbles), ignoring cold start (new users/items → content features, onboarding, bandits), leaking the future in splits.

## 7. Hyperparameters
Factors `f`, regularization λ, confidence α, negative sampling ratio, learning rate, neighbors k, embedding dims, temperature (softmax), candidate pool size, re-rank diversity weight.

## 8. Implementation
Scratch implicit ALS in `recsys_demo.py`:
```python
for u in range(n_users):    # fix item factors Y, solve each user's ridge problem
    Cu = C[u]; X[u] = np.linalg.solve((Y.T * Cu) @ Y + lam*np.eye(f), (Y.T * Cu) @ R[u])
```
Libraries: `implicit` (ALS/BPR), `surprise` (explicit CF), `LightFM`, `TensorFlow Recommenders`, `RecBole`, `torchrec`.

## 9. Evaluation & results (measured; **synthetic data**)
Synthetic world: 400 users × 300 items, rank-5 taste structure + global popularity, 16% density; one positive held out per user; metrics on 400 test users (standard error of HitRate ≈ ±0.02):
| Method | HitRate@10 | NDCG@10 |
|---|---|---|
| Random | 0.030 | 0.013 |
| Popularity | 0.147 | 0.076 |
| Item-kNN | 0.170 | 0.095 |
| ALS MF, f=2 | 0.172 | 0.083 |
| ALS MF, f=5 (true rank) | 0.217 | 0.113 |
| ALS MF, f=10 | 0.152 | 0.068 |
| User-kNN (k=30) | 0.230 | 0.116 |
Reading: personalization beats popularity by 50–60% in hit rate; MF peaks at the true latent rank (f=5) and overfits at f=10 (no tuning of λ/α here); user-kNN vs MF(f=5) is **within noise** (0.230 vs 0.217) — don't conclude one is better from a single small synthetic run. Real-data results differ; use temporal splits and report on your own logs.
**Offline vs online:** offline metrics can mislead (exposure bias); validate with A/B tests (CTR, conversion, retention), counterfactual/off-policy evaluation, and guardrail metrics.

## 10. Case study (template)
E-commerce "recommended for you": popularity baseline → ALS embeddings → two-tower retrieval with item text/image features → GBM ranker with price/recency/context → diversity re-rank. Cold start: content-based fallback; exploration via bandits (Ch 6.3).

## 11. Comparison
Popularity: strong baseline. CF: needs overlap. MF: compact and strong on interaction matrices. Two-tower: scalable with features. Ranker: precision on a short list, uses cross-features. Sequence/Transformer models: capture order and intent.

## Interview Q&A
- *Explicit vs implicit feedback?* Ratings are scarce; clicks are plentiful but ambiguous (absence ≠ dislike), so treat as confidence-weighted preference.
- *Cold start?* Content/metadata features, popularity within segment, onboarding questions, exploration.
- *Why two stages?* Cheap retrieval over millions of items, then expensive ranking over hundreds.
- *Popularity bias?* Models over-recommend popular items; mitigate with re-ranking, debiasing, exposure-aware training.
- *Evaluate offline?* Time-based split, leave-last-out, HitRate/NDCG@K against popularity; then A/B test.

## Exercises
Derive the ALS user update; implement BPR with sampled negatives; add item popularity bias terms; study how HitRate changes with density.

## Further reading
Koren, Bell & Volinsky (2009) "Matrix factorization techniques for recommender systems"; Hu, Koren & Volinsky (2008); Rendle et al. (2009) BPR; Covington, Adams & Sargin (2016); Ricci et al., *Recommender Systems Handbook*; Aggarwal, *Recommender Systems: The Textbook*.
