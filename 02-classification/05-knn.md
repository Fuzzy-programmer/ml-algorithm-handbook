# Chapter 2.5 — k-Nearest Neighbors & Nearest Centroid

[← Roadmap](../README.md) · Prev [2.4](04-lda-qda.md) · Next [2.6 SVM](06-svm.md) · Priority 🟠 · Code: [`classification_family.py`](../code/classification_family.py), [`classification_scratch.py`](../code/classification_scratch.py)

## 1. Identity & problem
Instance-based (lazy, non-parametric) classifier/regressor: predict from the labels of the k most similar training points. "Customers like you bought X."

## 2. History
Fix & Hodges (1951) introduced nearest-neighbor discrimination; **Cover & Hart (1967)** proved the 1-NN asymptotic error is at most twice the Bayes-optimal error. Efficiency: k-d trees (Bentley 1975), ball trees, LSH and modern ANN (HNSW, Malkov & Yashunin 2016; FAISS) powering vector search and RAG (Ch 6.4). Nearest Centroid (Rocchio/“Tibshirani's shrunken centroids”) is the compressed, parametric cousin.

## 3. Intuition
Look at your neighbors and vote. Small k ⇒ jagged, memorizing; large k ⇒ smooth, blurry.
```
 k=1 : boundary follows every point (high variance)     k=large : boundary smooth (high bias)
```

## 4. Math
Distance (Minkowski) `d(x,x') = (Σ|x_j−x'_j|^q)^{1/q}` (q=2 Euclidean, q=1 Manhattan), or cosine. Prediction: classification `ŷ = argmax_c Σ_{i∈N_k(x)} w_i 1[y_i=c]`; regression: (weighted) mean. Weights `w_i = 1/d_i` optional.
**Curse of dimensionality:** in high dimensions distances concentrate (nearest and farthest become similar), neighbors stop being meaningful; need exponentially more data. Feature scaling is essential.
Assumes: similar inputs ⇒ similar outputs in the chosen metric; meaningful distance; enough local density.

**Worked example.** 1-D training: x=1(A), 2(A), 3(B), 6(B), 7(B). Query 2.6: distances 1.6, 0.6, 0.4, 3.4, 4.4. k=1 → nearest is x=3 → **B**. k=3 → neighbors {3(B), 2(A), 1(A)} → A wins 2–1 → **A**. Prediction flips with k: that's the bias–variance dial.

## 5. Training & prediction
"Training" stores the data (`O(1)` compute, `O(np)` memory). Prediction brute force `O(n p)` per query; k-d/ball trees help in low dimension (<~20); ANN indexes give sub-linear search for embeddings with approximate recall. Scale features; consider PCA/metric learning first.

## 6. Evolution
| Step | Improvement | Cost |
|---|---|---|
| 1-NN → k-NN | Noise robustness | Choose k |
| Weighted k-NN, kernel regression (Nadaraya–Watson) | Closer points matter more | Bandwidth |
| Brute force → trees → ANN (HNSW/IVF/PQ) | Speed | Approximate answers, index memory |
| k-NN → Nearest Centroid | One prototype per class, tiny model | Assumes compact classes |
| Learned metrics / embeddings (NCA, Siamese, contrastive) | Meaningful distance for images/text | Training needed |

## 7. When to use / not
Use: small/medium data, low dimension, few features that are all meaningful, quick baseline, non-linear boundary without training, recommendation/retrieval over embeddings, anomaly detection via distance. Avoid: high-dimensional raw features, large n with strict latency (without ANN), many irrelevant features, imbalanced classes (majority swamps), when probabilities must be calibrated.

## 8. Strengths / weaknesses
+ No training, naturally multiclass, flexible boundary, simple to explain ("here are the 5 similar cases"), asymptotically near-optimal. − Slow, memory-hungry predictions, sensitive to scaling/irrelevant features/noise, curse of dimensionality, poor extrapolation.

## 9. Hyperparameters
`n_neighbors` (k; use odd for binary; tune by CV), `weights` (uniform/distance), `metric`/`p`, `algorithm` (auto/kd_tree/ball_tree/brute), `leaf_size`.

## 10. Implementation
From scratch (brute force):
```python
d = ((Xq[:,None,:] - Xtr[None])**2).sum(2)            # squared Euclidean, shape (nq, n)
idx = np.argsort(d, 1)[:, :k]
pred = (ytr[idx].mean(1) >= .5).astype(int)           # binary majority vote
```
Library:
```python
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance"))
```

## 11. Evaluation & results (measured)
Breast cancer 5-fold CV: scaled k-NN(5) acc **0.965** vs **0.932 unscaled** (scaling matters); NearestCentroid 0.931.
Moons (nonlinear, noise 0.25): train vs CV accuracy by k:
| k | train | CV |
|---|---|---|
| 1 | 1.000 | 0.908 |
| 5 | 0.950 | 0.926 |
| **15** | 0.940 | **0.944** |
| 50 | 0.936 | 0.932 |
| 200 | 0.860 | 0.832 |
k=1 memorizes (train 1.0), k=200 underfits. k-NN(15) beat every linear model on moons (0.944 vs ~0.85).

## 12. Case study
Content-based "similar items": embed products, index with FAISS/HNSW, return top-k with cosine distance; evaluate with recall@k against exact search and online CTR. For tabular classification use it as a baseline against boosted trees.

## 13. Comparison
vs SVM-RBF: similar flexibility but SVM stores only support vectors and trains a margin. vs trees: trees ignore irrelevant features; k-NN does not. vs logistic: k-NN handles nonlinearity but lacks parametric compactness.

## Interview Q&A
- *Why scale features?* Distances are dominated by large-range features.
- *k too small / too large?* Overfit / underfit.
- *Curse of dimensionality?* Points become equidistant, neighbors uninformative.
- *Inference cost?* O(n·p) naive per query; use ANN indexes.

## Exercises
Add irrelevant noise features and watch accuracy fall; implement distance-weighted k-NN; benchmark exact vs HNSW recall/latency.

## Further reading
Cover & Hart (1967) "Nearest neighbor pattern classification"; Malkov & Yashunin (2016) HNSW; ESL Ch 13; FAISS documentation.
