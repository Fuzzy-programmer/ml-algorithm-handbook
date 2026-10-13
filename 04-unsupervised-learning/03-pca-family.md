# Chapter 4.3 — PCA, SVD, Kernel PCA, ICA, NMF

[← Roadmap](../README.md) · Prev [4.2](02-other-clustering.md) · Next [4.4 t-SNE & UMAP](04-tsne-umap.md) · Priority 🔴 · Code: [`unsupervised_family.py`](../code/unsupervised_family.py)

## 1. Identity & problem
Linear (or kernelized) **dimensionality reduction / representation learning**: compress many correlated features into a few informative ones, for visualization, denoising, speed, de-collinearization, compression.

## 2. History (several related inventions)
**PCA**: Pearson (1901, best-fit lines), Hotelling (1933, principal components). **SVD**: Eckart–Young theorem (1936) — truncated SVD is the best low-rank approximation. Probabilistic PCA (Tipping & Bishop 1999). **Kernel PCA**: Schölkopf, Smola & Müller (1998). **ICA**: Comon (1994), FastICA (Hyvärinen 1999) — finds statistically *independent* sources (cocktail-party problem). **NMF**: Paatero & Tapper (1994), Lee & Seung (1999, Nature) — parts-based, non-negative factors. Nonlinear successors: autoencoders (Ch 5.5), t-SNE/UMAP for visualization (Ch 4.4). These answer different questions, not "versions" of each other.

## 3. Intuition
Rotate the axes to line up with the directions where the data vary most; keep the first few. Like finding the best angle to photograph a cloud of points so its shadow keeps the most spread.

## 4. Math
Center X. Covariance `Σ = XᵀX/(n−1)`. PCA = eigen-decomposition `Σ = VΛVᵀ`; PCs are columns of V; scores `Z = XV_k`. Equivalent: SVD `X = UΣ_sVᵀ`, `λ_i = σ_i²/(n−1)`. Explained variance ratio `λ_i/Σλ`. Reconstruction `X̂ = Z_kV_kᵀ` minimizes squared error among rank-k linear maps.
**Worked example.** Points (1,2),(2,3),(3,4),(4,5): mean (2.5,3.5); centered (−1.5,−1.5),(−0.5,−0.5),(0.5,0.5),(1.5,1.5). `Σ = [[1.667,1.667],[1.667,1.667]]`, eigenvalues **3.333** and **0**; PC1 = (1,1)/√2 holds **100%** of the variance — the 2-D data are really 1-D.
**Kernel PCA:** eigen-decompose the (centered) kernel matrix `K_ij=k(x_i,x_j)` — nonlinear structure, no explicit map; `O(n³)`. **ICA:** find unmixing `W` maximizing non-Gaussianity (kurtosis/negentropy). **NMF:** `X ≈ WH` with `W,H ≥ 0`, minimizing Frobenius or KL error — additive "parts" (topics in text, spectral components, recommender factors).
Assumptions: PCA — linear structure, variance = information (so **standardize** if units differ), orthogonal components; ICA — independent non-Gaussian sources; NMF — non-negative data.

## 5. Training & cost
PCA via SVD `O(min(n,p)²·max(n,p))`; randomized SVD (Halko et al. 2011) `O(npk)` for large matrices; `IncrementalPCA` for streaming; `TruncatedSVD` for sparse matrices (LSA) without centering. NMF: iterative multiplicative/coordinate updates, local minima.

## 6. Choosing k
Scree plot/elbow; cumulative explained variance (90–99%); cross-validated downstream score; Minka's MLE (`n_components="mle"`); parallel analysis.

## 7. When to use / not
Use: collinear features, visualization (2–3 PCs), noise reduction, speeding up distance/linear models, compressing embeddings, whitening. Avoid: when interpretability of original features is required, nonlinear manifolds (use Kernel PCA/UMAP/autoencoders), when target-relevant signal sits in low-variance directions (variance ≠ relevance; consider PLS or supervised selection), tree models (little benefit).

## 8. Strengths / weaknesses
+ Fast, deterministic, closed form, decorrelates, reconstruction guarantees. − Linear only, scale-sensitive, components hard to interpret, unsupervised (ignores y), sensitive to outliers (consider robust PCA).

## 9. Hyperparameters
PCA: `n_components` (int or variance fraction), `whiten`, `svd_solver` (full/randomized/arpack). Kernel PCA: `kernel`, `gamma`, `n_components`. ICA: `n_components`, `fun`. NMF: `n_components`, `init`, `beta_loss`, `alpha_W/H` (regularization).

## 10. Implementation
```python
from sklearn.decomposition import PCA, KernelPCA, FastICA, NMF, TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
pca = make_pipeline(StandardScaler(), PCA(n_components=0.95))     # keep 95% variance
# from scratch:
Xc = X - X.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False); Z = Xc @ Vt[:k].T; ratio = S**2 / (S**2).sum()
```

## 11. Evaluation & results (measured; 8×8 digits, 64 features)
Components needed for 90% / 95% / 99% variance: **21 / 29 / 41**. SVD and sklearn PCA agree exactly (variance ratios 0.1489, 0.1362, 0.1179 for the first three).
Logistic regression 5-fold CV accuracy after PCA (inside the pipeline):
| Components | Accuracy |
|---|---|
| 5 | 0.825 |
| 10 | 0.891 |
| 20 | 0.921 |
| 64 (all) | 0.928 |
20 components (31% of the features) retain 99% of the full-feature accuracy (0.921 vs 0.928). Reconstruction MSE with 10 components: 0.0192 (pixels scaled to [0,1]).
**Kernel PCA vs linear on concentric circles** (logistic regression on the 2-D embedding, CV accuracy): linear PCA **0.448** (chance), RBF Kernel PCA (γ=2) **1.000** — linear projection can't unfold rings; a kernel can.

## 12. Case study
Image/embedding compression and decorrelation: standardize → PCA(0.95) → logistic/k-NN; compare accuracy, training time and memory. For text: TF-IDF → TruncatedSVD(100–300) = LSA; for topic parts use NMF and print top words per component.

## 13. Comparison
PCA (global linear variance) vs ICA (independence) vs NMF (parts, positivity) vs Kernel PCA (nonlinear, `O(n³)`) vs LDA (supervised separation, Ch 2.4) vs t-SNE/UMAP (visualization; no meaningful inverse transform) vs autoencoders (flexible nonlinear). PCA is the default first thing to try.

## Interview Q&A
- *PCA vs SVD?* PCA on centered data = SVD; right singular vectors are PCs.
- *Why standardize?* Otherwise large-unit features dominate variance.
- *Are PCs ordered by importance for prediction?* By variance, not predictive power.
- *When use NMF over PCA?* Non-negative data needing additive, interpretable parts.

## Exercises
Prove PCs are eigenvectors of the covariance; reconstruct a digit with k=2,5,20,64 components; apply NMF to a text corpus and name the topics.

## Further reading
Hotelling (1933); Jolliffe, *Principal Component Analysis*; Halko, Martinsson & Tropp (2011); Lee & Seung (1999); Hyvärinen & Oja (2000) "Independent component analysis: algorithms and applications"; sklearn "Decomposing signals" guide.
