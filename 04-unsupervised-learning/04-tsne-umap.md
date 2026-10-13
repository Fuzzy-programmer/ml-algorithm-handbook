# Chapter 4.4 — t-SNE and UMAP (Nonlinear Embeddings for Visualization)

[← Roadmap](../README.md) · Prev [4.3](03-pca-family.md) · Next [4.5 Anomaly detection](05-anomaly-detection.md) · Priority 🟠 · Code: [`unsupervised_family.py`](../code/unsupervised_family.py)

## 1. Identity & problem
Map high-dimensional data to 2-D/3-D so that **neighbors stay neighbors**, mainly to *look at* structure (clusters, batches, outliers). Not a clustering algorithm and not a general-purpose feature extractor.

## 2. History
Earlier manifold learning: MDS (Torgerson 1952), Isomap and LLE (2000), Laplacian eigenmaps. **SNE** (Hinton & Roweis 2002) → **t-SNE** (van der Maaten & Hinton, JMLR 2008): symmetric affinities and a heavy-tailed Student-t in the low-dimensional space fixed SNE's "crowding problem". Barnes–Hut (2014) and FFT-based approximations made it practical. **UMAP** (McInnes, Healy & Melville 2018): fuzzy topological graph + cross-entropy layout; faster, scales better, preserves more global structure in practice. Related: LargeVis, TriMap, PaCMAP.

## 3. Intuition
Measure "who is whose neighbor" in the original space as probabilities; place points in 2-D and push/pull them until the 2-D neighbor probabilities match.

## 4. Math (t-SNE)
High-dim affinities `p_{j|i} = exp(−‖x_i−x_j‖²/2σ_i²)/Σ_k exp(−‖x_i−x_k‖²/2σ_i²)`, with σ_i chosen so that the **perplexity** `2^{H(P_i)}` (≈ effective neighbor count) equals a target; symmetrize `p_ij=(p_{j|i}+p_{i|j})/2n`. Low-dim: `q_ij ∝ (1+‖y_i−y_j‖²)⁻¹`. Minimize `KL(P‖Q)` by gradient descent. KL penalizes placing near points far apart much more than the reverse ⇒ local structure is preserved; distances between far clusters are not meaningful.
**Worked example (perplexity).** A point whose neighbor distribution is uniform over 30 neighbors has entropy log₂30 and perplexity 30; σ_i is tuned per point to hit that value, so dense and sparse regions get comparable neighborhoods.
UMAP: build a weighted k-NN graph (`n_neighbors`), symmetrize with fuzzy union, optimize a low-dimensional layout with attractive/repulsive forces (`min_dist` controls tightness).

## 5. Cost
t-SNE: Barnes–Hut `O(n log n)` (FFT-accelerated variants faster), practical to ~10⁵ points. UMAP: approximate-NN graph (NN-descent) plus SGD layout; scales to millions of points. Measured on the 1,797-sample digits set (4 cores): **t-SNE 9–10 s, UMAP 29 s** (the UMAP figure includes one-time numba compilation, so it is not representative of steady-state speed). Both are stochastic: fix `random_state`; PCA-initialization stabilizes global layout.

## 6. When to use / not
Use: exploratory visualization of embeddings, single-cell data, checking label/batch structure, sanity-checking learned representations, pre-clustering view. Avoid: as input features to downstream models without care (t-SNE has no `transform` for new points; UMAP does), reading cluster sizes/distances/densities as quantitative, drawing conclusions from one run/one hyperparameter, very high-dimensional raw data (reduce with PCA to ~50 first).

## 7. Hyperparameters
t-SNE: `perplexity` (5–50; ≲ n/3), `learning_rate` ("auto"), `n_iter`/`max_iter` (≥1000), `early_exaggeration`, `init="pca"`. UMAP: `n_neighbors` (local vs global emphasis; 15 default), `min_dist` (tight vs spread), `metric`, `n_components`.

## 8. Implementation
```python
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(PCA(50).fit_transform(X))
import umap                       # pip install umap-learn
U = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit(X); emb = U.embedding_; new = U.transform(Xnew)
```

## 9. Evaluation (measured, with caveats)
Digits (10 classes): a 5-NN classifier on the 2-D embedding scores **0.976** (t-SNE) vs **0.603** (PCA-2), showing how much more local class structure survives in t-SNE. **Caveat:** t-SNE was fit on all the data (labels were not used, but the test points were part of the embedding), so this cross-validated number is optimistic and is a visualization-quality check only, not a model accuracy. Better embedding metrics: trustworthiness / continuity, k-NN preservation, and stability across seeds and parameters.

## 10. Common misreadings (Wattenberg, Viégas & Johnson 2016, "How to use t-SNE effectively")
Cluster sizes mean nothing · distances between clusters mean little · random noise can look like clusters at low perplexity · results change with perplexity — always show several.

## 11. Comparison
| | PCA | t-SNE | UMAP |
|---|---|---|---|
| Linear | Yes | No | No |
| Global structure | Preserved | Weak | Better, not guaranteed |
| Speed | Fastest | Slow | Fast |
| `transform` new data | Yes | No | Yes |
| Deterministic | Yes | No | No |
| Best for | features, compression | local cluster view | large-scale view, features for HDBSCAN |

## Interview Q&A
- *Why t-distribution?* Heavy tails let moderately distant points sit farther apart, relieving the crowding problem.
- *Can I cluster on t-SNE?* You can, but distances/densities are distorted; prefer clustering in the original or PCA space and use t-SNE to display.
- *PCA vs t-SNE?* Linear global variance vs nonlinear local neighborhoods.

## Exercises
Embed digits with perplexity 5/30/100 and describe changes; run UMAP with `n_neighbors` 5/15/50; check seed stability.

## Further reading
van der Maaten & Hinton (2008); McInnes, Healy & Melville (2018) "UMAP"; Wattenberg et al. (2016) *Distill*; Kobak & Berens (2019) "The art of using t-SNE for single-cell transcriptomics."
