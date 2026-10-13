# Chapter 4.2 — Hierarchical, DBSCAN, HDBSCAN, OPTICS, Mean Shift, Spectral Clustering

[← Roadmap](../README.md) · Prev [4.1](01-kmeans.md) · Next [4.3 PCA family](03-pca-family.md) · Priority 🟠 · Code: [`unsupervised_family.py`](../code/unsupervised_family.py)

Different clustering algorithms encode different **definitions of a cluster**. Not a linear progression — pick by the shape/noise structure of your data.

| Algorithm | A cluster is… | Needs K? | Handles noise | Shape |
|---|---|---|---|---|
| K-Means (4.1) | compact blob around a mean | Yes | No | Convex, similar size |
| Hierarchical | a branch in a merge tree | Cut level | No | Depends on linkage |
| DBSCAN | dense region separated by sparse | No (eps, minPts) | Yes | Arbitrary |
| HDBSCAN | stable dense region across density levels | No (min cluster size) | Yes | Arbitrary, varying density |
| OPTICS | density ordering, extract clusters | No | Yes | Arbitrary |
| Mean Shift | mode of a density estimate | No (bandwidth) | Partly | Blob-like modes |
| Spectral | connected group in a similarity graph | Yes | No | Non-convex, graph-like |

## A. Hierarchical (agglomerative) clustering
**History:** numerical taxonomy (Sokal & Sneath 1963), Ward's minimum-variance method (1963), dendrograms. **Idea:** start with each point alone; repeatedly merge the two closest clusters. **Linkages:** single (min distance; chains, finds elongated shapes, noise-sensitive), complete (max; compact), average (UPGMA), **Ward** (minimize variance increase; compact, K-Means-like).
**Worked example.** Points A=1, B=2, C=6, D=7, E=15 (1-D). Single linkage: merge (A,B) at distance 1, (C,D) at 1; then {A,B} with {C,D} at 4 (B–C); then E joins at 8 (D–E). Cutting the dendrogram between heights 4 and 8 gives 2 clusters: {A,B,C,D}, {E}; cutting between 1 and 4 gives {A,B},{C,D},{E}.
Cost `O(n²)` memory, `O(n² log n)`–`O(n³)` time ⇒ up to ~10⁴ points. No re-assignment (greedy merges are final). Use for small data, taxonomies, heatmap ordering, when you want a dendrogram. `AgglomerativeClustering(n_clusters, linkage="ward")`, `scipy.cluster.hierarchy`.

## B. DBSCAN
**History:** Ester, Kriegel, Sander & Xu, KDD 1996. **Definitions:** a point is a *core* point if ≥ `min_samples` points lie within radius `eps`; clusters are chains of density-reachable core points; non-reachable points are *noise* (label −1).
Strengths: arbitrary shapes, automatic noise, no K; cost `O(n log n)` with spatial index. Weaknesses: single global `eps` cannot fit varying densities, sensitive to scaling, struggles in high-d; `eps` selection via k-distance plot (knee). Hyperparameters: `eps`, `min_samples` (≥ d+1; often 2·d), `metric`.

## C. HDBSCAN and OPTICS
**OPTICS** (Ankerst et al. 1999) produces a reachability ordering, avoiding fixed `eps`. **HDBSCAN** (Campello, Moulavi & Sander 2013; McInnes et al. 2017) builds a hierarchy over all density levels and extracts the most *stable* clusters ⇒ handles varying densities with one intuitive parameter (`min_cluster_size`); gives soft membership/outlier scores. In scikit-learn ≥1.3 as `HDBSCAN`. Improvement over DBSCAN: variable density; cost: heavier compute, still sensitive in high-d.

## D. Mean Shift
Fukunaga & Hostetler (1975); Comaniciu & Meer (2002). Each point climbs the gradient of a kernel density estimate to a mode; points reaching the same mode share a cluster. Bandwidth controls cluster count; `O(n²)`; good for small low-dimensional data (tracking, image segmentation).

## E. Spectral clustering
Shi & Malik (2000), Ng–Jordan–Weiss (2001). Build a similarity graph, compute the graph Laplacian `L = D − W`, embed points with its smallest eigenvectors, run K-Means there. Finds non-convex clusters connected by similarity (rings, moons). Needs K and a graph; eigen-decomposition `O(n³)` (sparse/approximate methods scale further). Use `affinity="nearest_neighbors"` for shapes.

## Measured comparison (Adjusted Rand Index, 600 points, standardized; **default-ish parameters, not tuned**)
| Dataset | KMeans | Ward | Single | DBSCAN(eps .3) | HDBSCAN | Spectral |
|---|---|---|---|---|---|---|
| 3 blobs | 0.98 | 0.93 | 0.57 | 0.57 | 0.57 | 0.99 |
| Moons | 0.48 | 0.64 | **1.00** | **1.00** | 0.99 | **1.00** |
| Circles | 0.00 | 0.00 | **1.00** | **1.00** | **1.00** | **1.00** |
Reading: centroid/variance methods win on blobs and fail on non-convex shapes; density/graph methods win on shapes but at these settings fragmented or flagged blob points as noise (0.57; blob std is large, so `eps=0.3` is too small — a reminder that DBSCAN needs `eps` tuning). No algorithm is universally best.

## When to use what
| Data | Choose |
|---|---|
| Large, blobs, speed | K-Means / mini-batch |
| Overlapping ellipses, probabilities | GMM |
| Arbitrary shapes + noise | DBSCAN / HDBSCAN |
| Varying densities | HDBSCAN |
| Need a hierarchy/dendrogram, small n | Ward / average |
| Non-convex, graph structure, small–medium n | Spectral |
| High-dimensional embeddings | Reduce (PCA/UMAP) first, then HDBSCAN/K-Means |

## Evaluation
Internal: silhouette (biased toward convex clusters), Davies–Bouldin, DBCV (density-based). External (with labels): ARI, NMI, homogeneity/completeness. Practical: stability under subsampling, cluster profiles, downstream impact. Never trust one internal metric across algorithm families.

## Implementation
```python
from sklearn.cluster import AgglomerativeClustering, DBSCAN, HDBSCAN, SpectralClustering, MeanShift, OPTICS
from sklearn.neighbors import NearestNeighbors
Xs = StandardScaler().fit_transform(X)
d = np.sort(NearestNeighbors(n_neighbors=5).fit(Xs).kneighbors(Xs)[0][:, -1])   # k-distance plot -> pick eps at the knee
labels = DBSCAN(eps=0.3, min_samples=5).fit_predict(Xs)                          # -1 = noise
labels = HDBSCAN(min_cluster_size=20).fit_predict(Xs)
labels = SpectralClustering(2, affinity="nearest_neighbors", random_state=0).fit_predict(Xs)
```

## Pitfalls
Not scaling; treating noise label −1 as a cluster; picking `eps` blindly; clustering raw high-dimensional data; interpreting clusters as truth rather than a useful partition.

## Interview Q&A
- *DBSCAN vs K-Means?* Density/arbitrary shape/noise/no K vs compact blobs/needs K.
- *Why HDBSCAN?* Varying densities; one stable parameter.
- *Single vs Ward linkage?* Single chains and detects elongated clusters but is noise-sensitive; Ward gives compact spherical clusters.
- *How pick eps?* k-distance plot knee.

## Exercises
Produce a dendrogram for 20 points and cut it at 3 heights; tune DBSCAN on the blob set until ARI > 0.95; compare HDBSCAN vs DBSCAN on two clusters of different density.

## Further reading
Ester et al. (1996); Campello et al. (2013); McInnes, Healy & Astels (2017) "hdbscan: Hierarchical density based clustering"; von Luxburg (2007) "A tutorial on spectral clustering"; Comaniciu & Meer (2002); sklearn Clustering guide.
