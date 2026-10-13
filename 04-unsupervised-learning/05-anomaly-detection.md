# Chapter 4.5 — Anomaly Detection: Isolation Forest, LOF, One-Class SVM, Robust Covariance

[← Roadmap](../README.md) · Prev [4.4](04-tsne-umap.md) · Next [4.6 Association rules](06-association-rules.md) · Priority 🟠 · Code: [`unsupervised_family.py`](../code/unsupervised_family.py)

## 1. Problem & framing
Find rare observations that differ from the norm: fraud, intrusions, sensor faults, defects. Three regimes: **supervised** (labeled anomalies; use classifiers, Ch 3.4), **semi-supervised / novelty detection** (train on normal only), **unsupervised outlier detection** (contaminated unlabeled data). Anomalies are rare, evolving, and often poorly defined — evaluation and threshold choice are the hard parts.

## 2. Methods and history
| Method | Idea | Origin |
|---|---|---|
| Statistical (z-score, MAD, Mahalanobis) | Far from center in standardized units | Classic |
| **Robust covariance / Elliptic Envelope** | Fit a Gaussian with the Minimum Covariance Determinant, flag large Mahalanobis distance | Rousseeuw 1984/1999 |
| **LOF** | Compare a point's local density to its neighbors' | Breunig, Kriegel, Ng, Sander 2000 |
| **One-Class SVM** | Max-margin boundary around normal data in kernel space | Schölkopf et al. 2001 |
| **Isolation Forest** | Anomalies are easy to isolate by random splits | Liu, Ting, Zhou 2008 (ICDM) |
| Autoencoders / deep one-class, Time-series methods | High reconstruction error / forecasting residual | 2010s |
Evolution: distance/statistical → density (LOF handles varying density) → isolation (fast, no distance computation, scales) → deep models for complex data. Again, siblings, not a strict ladder.

## 3. Math highlights
**Isolation Forest:** recursively split on a random feature at a random threshold; anomalies are isolated in few splits. Average path length `E[h(x)]` over trees; score `s(x,n) = 2^{−E[h(x)]/c(n)}`, `c(n) = 2H(n−1) − 2(n−1)/n` (H harmonic number). s→1 anomaly, s≈0.5 ambiguous, s→0 normal.
**Worked example.** n=256: `c(256) ≈ 2(ln 255 + 0.5772) − 2·255/256 ≈ 10.24`. A point isolated after on average 3 splits: `s = 2^{−3/10.24} = 0.816` (anomalous). A typical point with path 10.24: `s = 0.5`.
**LOF:** `LOF(x) = mean over neighbors o of lrd(o)/lrd(x)` where local reachability density `lrd` is the inverse of the mean reachability distance; ≈1 normal, ≫1 outlier.
**One-class SVM:** `min ½‖w‖² + (1/νn)Σξ_i − ρ`; ν upper-bounds the outlier fraction. **Mahalanobis:** `d² = (x−μ)ᵀΣ⁻¹(x−μ)` (≈ χ²_p for Gaussian data).

## 4. Complexity
IsolationForest `O(t·ψ log ψ)` training with subsample size ψ (default 256) — linear in n, tiny memory; LOF `O(n²)` brute / `O(n log n)` with trees; OCSVM `O(n²–n³)`; Elliptic Envelope `O(np²)` + MCD cost.

## 5. When to use / not
| Situation | Prefer |
|---|---|
| Tabular, many features, large n, no assumptions | Isolation Forest |
| Clusters of different density; local outliers matter | LOF (novelty=True for new data) |
| Unimodal Gaussian-like normal data, interpretable distance | Robust covariance |
| Small n, smooth boundary, kernel structure | One-Class SVM (scale features, tune γ, ν) |
| Labeled anomalies exist | Supervised model (PR-AUC) |
| Images/sequences | Autoencoders / forecasting residuals |
Avoid: assuming "unsupervised outlier = business-relevant anomaly" without feedback; using accuracy as the metric; high-dimensional distance methods without reduction.

## 6. Hyperparameters
IF: `n_estimators`, `max_samples`, `contamination` (sets threshold only), `max_features`. LOF: `n_neighbors` (20 typical), `contamination`, `novelty`. OCSVM: `nu`, `gamma`, `kernel`. EllipticEnvelope: `contamination`, `support_fraction`.

## 7. Implementation
```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
iso = IsolationForest(n_estimators=300, random_state=0).fit(X);  score = -iso.score_samples(X)   # higher = more anomalous
lof = LocalOutlierFactor(n_neighbors=20).fit(X);                  score = -lof.negative_outlier_factor_
oc  = OneClassSVM(nu=0.03, gamma="scale").fit(X);                 score = -oc.score_samples(X)
ee  = EllipticEnvelope(contamination=0.03).fit(X);                score = ee.mahalanobis(X)
```

## 8. Evaluation & results (measured)
Setup: 1,000 normal points from **two** Gaussian clusters + 30 uniform random outliers (2.9%); labels used only for scoring.
| Method | ROC-AUC | Average precision |
|---|---|---|
| Isolation Forest | 0.983 | 0.798 |
| LOF (k=20) | 0.983 | **0.897** |
| One-Class SVM | 0.963 | 0.772 |
| Elliptic Envelope (assumes one Gaussian) | 0.960 | 0.714 |
Reading: all work well on this easy 2-D problem; the single-Gaussian envelope is the worst because the normal data are bimodal (assumption violated); LOF has the best precision here. In real high-dimensional data expect larger gaps — evaluate on your data.
**How to evaluate in practice:** with a small labeled set use PR-AUC / precision@k / recall at fixed alert budget; without labels, inspect top-ranked items with domain experts, track alert precision over time, and simulate injected anomalies.

## 9. Pitfalls
Setting `contamination` arbitrarily (it just picks the threshold) · concept drift (retrain/monitor) · scale and categorical encoding · duplicate/near-duplicate normal patterns · tuning on the same anomalies you evaluate on · alert fatigue (choose the threshold from the cost of review).

## Interview Q&A
- *Why does Isolation Forest work?* Anomalies are few and different ⇒ shorter average isolation paths.
- *LOF vs Isolation Forest?* LOF = local density, handles varying densities but quadratic; IF = global-ish random partitioning, fast, scalable.
- *Metric for imbalanced anomaly data?* PR-AUC, precision@k.
- *Novelty vs outlier detection?* Train on clean normal data vs find outliers inside contaminated data.

## Exercises
Add 50 irrelevant features and re-run the table; implement isolation trees in 30 lines; compare thresholds chosen by contamination vs by cost.

## Further reading
Liu, Ting & Zhou (2008); Breunig et al. (2000); Schölkopf et al. (2001); Chandola, Banerjee & Kumar (2009) "Anomaly detection: a survey"; sklearn "Novelty and outlier detection".
