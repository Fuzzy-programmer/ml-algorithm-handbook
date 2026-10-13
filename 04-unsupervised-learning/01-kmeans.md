# Chapter 4.1 — K-Means and K-Medoids

[← Roadmap](../README.md) · Prev [3.5](../03-trees-and-ensembles/05-gmm-hmm-bayesnet.md) · Next [4.2 Other clustering](02-other-clustering.md) · Priority 🔴 · Code: [`unsupervised_family.py`](../code/unsupervised_family.py)

## 1. Identity & problem
Partition unlabeled data into K compact groups, each represented by a centroid. Customer segmentation, vector quantization, image color compression, preprocessing (cluster features).

## 2. History
Steinhaus (1956); **Lloyd** (Bell Labs 1957; published 1982) for signal quantization; Forgy (1965); **MacQueen (1967)** coined "k-means". **K-Medoids/PAM** (Kaufman & Rousseeuw 1987; CLARA/CLARANS for scale) uses actual data points as centers and any distance. **k-means++** (Arthur & Vassilvitskii 2007) fixed bad initialization; **Mini-batch k-means** (Sculley 2010) for web scale. GMM (Ch 3.5) is the probabilistic generalization.

## 3. Intuition
Drop K pins; assign every point to its nearest pin; move each pin to the average of its points; repeat until pins stop moving.

## 4. Math
Minimize within-cluster sum of squares (inertia): `J = Σ_k Σ_{x∈C_k} ‖x − μ_k‖²`. Lloyd's algorithm alternates assignment and mean update; each step cannot increase J, so it converges, but only to a **local** minimum (NP-hard globally).
**Worked example.** 1-D points 1, 2, 9, 10, 11; initial centers 1 and 10. Assign: {1,2} → c₁, {9,10,11} → c₂. Update: c₁=1.5, c₂=10. Cost = (0.5²+0.5²)+(1+0+1) = **2.5**. Reassigning changes nothing ⇒ converged.
Assumptions implied: roughly spherical, similar-sized, similar-density clusters; Euclidean distance meaningful; K known. Sensitive to scale and outliers (mean).

## 5. Training & cost
`O(n·K·d)` per iteration, few dozen iterations typical; memory `O(nd+Kd)`. Use `k-means++` init and several restarts (`n_init`). Mini-batch for millions of rows (small accuracy loss). Prediction: nearest centroid `O(Kd)`.

## 6. Choosing K
Elbow on inertia (subjective); **silhouette** (Rousseeuw 1987): `s = (b−a)/max(a,b)` ∈ [−1,1], a = mean intra-cluster distance, b = mean nearest-other-cluster distance; gap statistic; BIC with GMM; stability across resamples; and — most importantly — usefulness to the business.

## 7. Evolution
| Step | Improvement | Cost |
|---|---|---|
| Lloyd → k-means++ | Much better/faster starts | Slightly slower seeding |
| → Mini-batch | Scale | Mild quality loss |
| → K-Medoids (PAM) | Robust to outliers, arbitrary distances | `O(K(n−K)²)` classic; slower |
| → K-Modes / K-Prototypes | Categorical / mixed data | Different dissimilarity |
| → Fuzzy c-means, GMM | Soft membership, elliptical shape | More parameters |
| → Spectral / DBSCAN | Non-convex shapes | Different assumptions (Ch 4.2) |

## 8. When to use / not
Use: large numeric data, roughly blob-like clusters, quick segmentation, feature learning (distance-to-centroid features), compression. Avoid: non-convex shapes, very different densities/sizes, high dimensions without reduction (distances concentrate), categorical data, outlier-heavy data (use medoids/HDBSCAN).

## 9. Strengths / weaknesses
+ Simple, fast, scalable, easy to interpret. − Needs K, local optima, spherical bias, scale- and outlier-sensitive, hard assignments, no notion of noise.

## 10. Hyperparameters
`n_clusters`, `init` (`k-means++`/random), `n_init` (restarts), `max_iter`, `tol`, `algorithm` (lloyd/elkan), `batch_size` (mini-batch).

## 11. Implementation
```python
def kmeans(X, K, iters=100, seed=0):
    rng = np.random.default_rng(seed); C = X[rng.choice(len(X), K, replace=False)]
    for _ in range(iters):
        lab = ((X[:, None] - C[None])**2).sum(2).argmin(1)
        newC = np.array([X[lab == k].mean(0) if (lab == k).any() else C[k] for k in range(K)])
        if np.allclose(newC, C): break
        C = newC
    return C, lab
from sklearn.cluster import KMeans, MiniBatchKMeans
km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X_scaled)
```

## 12. Evaluation & results (measured; 800 points, 4 true blobs)
| K | Inertia | Silhouette |
|---|---|---|
| 2 | 12838 | 0.624 |
| 3 | 6106 | 0.602 |
| **4** | **1966** | **0.701** |
| 5 | 1765 | 0.602 |
| 6 | 1609 | 0.493 |
| 7 | 1458 | 0.410 |
Inertia always falls with K (the elbow is at 4); silhouette peaks at the true K=4. **Initialization matters:** one random init gave inertia 6531 versus 1966 for k-means++ with 10 restarts (a bad local optimum merged/split clusters).
Other evaluation: with labels, ARI/NMI; without, silhouette/Davies–Bouldin/Calinski–Harabasz, stability, downstream task performance. On non-spherical shapes K-Means fails: ARI 0.48 on moons, 0.00 on circles (vs 1.0 for density-based methods, Ch 4.2).

## 13. Case study
Customer segmentation on RFM (recency, frequency, monetary): log-transform skewed features, standardize, choose K by silhouette + business interpretability, profile each cluster, validate stability across months, and track drift.

## 14. Comparison
K-Means for compact blobs at scale; GMM when clusters are elliptical or you need probabilities; DBSCAN/HDBSCAN for arbitrary shapes and noise; hierarchical when you want a dendrogram; spectral for graph-like structure.

## Interview Q&A
- *Does K-Means converge?* Yes, to a local minimum (objective monotonic, finite partitions).
- *How choose K?* Elbow/silhouette/gap/stability + domain use.
- *Why scale features?* Distance is dominated by large-range variables.
- *K-Means vs GMM?* Hard/spherical vs soft/elliptical; K-Means is GMM with shared spherical covariance and zero variance limit.

## Exercises
Implement k-means++ seeding; compare inertia over 100 random restarts; run K-Medoids on data with outliers and compare centers.

## Further reading
Lloyd (1982) *IEEE Trans. Inf. Theory*; MacQueen (1967); Arthur & Vassilvitskii (2007) "k-means++"; Kaufman & Rousseeuw, *Finding Groups in Data*; sklearn Clustering guide.
