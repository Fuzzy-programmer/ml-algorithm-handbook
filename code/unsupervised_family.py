"""Clustering, PCA family, t-SNE/UMAP-ish, anomaly detection, association rules, feature selection."""
import time, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, MeanShift, OPTICS, HDBSCAN
from sklearn.datasets import make_blobs, make_moons, make_circles, load_digits, load_breast_cancer
from sklearn.metrics import silhouette_score, adjusted_rand_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA, NMF, FastICA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE, RFECV
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

print("== K-Means: inertia/silhouette vs K on 4 blobs ==")
X, y = make_blobs(800, centers=4, cluster_std=1.1, random_state=3)
for k in range(2, 8):
    m = KMeans(k, n_init=10, random_state=0).fit(X); print(k, round(m.inertia_, 0), round(silhouette_score(X, m.labels_), 3))
bad = KMeans(4, init="random", n_init=1, random_state=2).fit(X).inertia_; good = KMeans(4, n_init=10, random_state=2).fit(X).inertia_
print("single random init inertia", round(bad), "vs k-means++ n_init=10", round(good))

print("\n== Clustering algorithms ARI on 3 shapes ==")
sets = {"blobs": make_blobs(600, centers=3, cluster_std=1.0, random_state=1),
        "moons": make_moons(600, noise=.07, random_state=0), "circles": make_circles(600, noise=.05, factor=.4, random_state=0)}
for name, (Xs, ys) in sets.items():
    Xs = StandardScaler().fit_transform(Xs); k = len(set(ys)); res = {}
    algs = {"KMeans": KMeans(k, n_init=10, random_state=0), "Ward": AgglomerativeClustering(k), "Single": AgglomerativeClustering(k, linkage="single"),
            "DBSCAN": DBSCAN(eps=.3, min_samples=5), "HDBSCAN": HDBSCAN(min_cluster_size=20), "Spectral": SpectralClustering(k, affinity="nearest_neighbors", random_state=0)}
    for an, a in algs.items(): res[an] = round(adjusted_rand_score(ys, a.fit_predict(Xs)), 2)
    print(name, res)

print("\n== PCA: digits (64-d) ==")
Xd, yd = load_digits(return_X_y=True); Xd = Xd / 16.0
p = PCA().fit(Xd); cum = np.cumsum(p.explained_variance_ratio_)
print("components for 90%/95%/99% variance:", [int(np.searchsorted(cum, t) + 1) for t in (.9, .95, .99)])
for nc in [5, 10, 20, 64]:
    pipe = make_pipeline(PCA(nc), LogisticRegression(max_iter=3000)); print(f"PCA({nc}) + logistic CV acc {cross_val_score(pipe, Xd, yd, cv=5).mean():.3f}")
Z = PCA(10).fit(Xd); rec = Z.inverse_transform(Z.transform(Xd)); print("10-comp reconstruction MSE", round(np.mean((Xd - rec) ** 2), 4))
# SVD equivalence
Xc = Xd - Xd.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False); print("SVD var ratio[0:3]", ((S ** 2) / (S ** 2).sum())[:3].round(4), "PCA", p.explained_variance_ratio_[:3].round(4))
Xci, _ = make_circles(500, noise=.05, factor=.3, random_state=0)
for nm, emb in [("linear PCA", PCA(2)), ("RBF Kernel PCA", KernelPCA(2, kernel="rbf", gamma=2))]:
    Zc = emb.fit_transform(Xci); print(f"circles: {nm} -> logistic CV acc", round(cross_val_score(LogisticRegression(), Zc, _, cv=5).mean(), 3))
t = time.time(); T = TSNE(2, perplexity=30, random_state=0, init="pca").fit_transform(Xd); print("t-SNE time", round(time.time() - t, 1), "s; kNN(5) CV acc in 2-D embedding (optimistic, fit on all data):",
      round(cross_val_score(__import__("sklearn.neighbors", fromlist=["x"]).KNeighborsClassifier(5), T, yd, cv=5).mean(), 3), "| PCA-2:", round(cross_val_score(__import__("sklearn.neighbors", fromlist=["x"]).KNeighborsClassifier(5), PCA(2).fit_transform(Xd), yd, cv=5).mean(), 3))
try:
    import umap
    t = time.time(); Uu = umap.UMAP(random_state=0).fit_transform(Xd); print("UMAP time", round(time.time() - t, 1))
except Exception as e: print("umap not installed:", type(e).__name__)

print("\n== Anomaly detection: 1000 normal 2-cluster + 30 outliers, AUC / AP ==")
rng = np.random.default_rng(0)
Xn = np.r_[rng.normal(0, 1, (500, 2)), rng.normal(6, .8, (500, 2))]; Xo = rng.uniform(-6, 12, (30, 2)); Xa = np.r_[Xn, Xo]; ya = np.r_[np.zeros(1000), np.ones(30)]
det = {"IsolationForest": IsolationForest(random_state=0).fit(Xa), "LOF(k=20)": LocalOutlierFactor(20), "OneClassSVM": OneClassSVM(nu=.03, gamma="scale").fit(Xa),
       "EllipticEnvelope(1 Gaussian)": EllipticEnvelope(contamination=.03, random_state=0).fit(Xa)}
for n, d in det.items():
    if n.startswith("LOF"): d.fit(Xa); s = -d.negative_outlier_factor_
    elif n.startswith("Iso"): s = -d.score_samples(Xa)
    elif n.startswith("One"): s = -d.score_samples(Xa)
    else: s = d.mahalanobis(Xa)
    print(f"{n:30s} ROC-AUC {roc_auc_score(ya, s):.3f}  AP {average_precision_score(ya, s):.3f}")

print("\n== Feature selection (breast cancer + 100 noise features), 5-fold CV acc ==")
Xb, yb = load_breast_cancer(return_X_y=True); Xb = np.c_[Xb, np.random.default_rng(1).normal(size=(len(Xb), 100))]
lr = lambda: LogisticRegression(max_iter=5000)
print("all 130:", round(cross_val_score(make_pipeline(StandardScaler(), lr()), Xb, yb, cv=5).mean(), 3))
print("filter ANOVA top10 (inside CV):", round(cross_val_score(make_pipeline(StandardScaler(), SelectKBest(f_classif, k=10), lr()), Xb, yb, cv=5).mean(), 3))
print("filter MI top10:", round(cross_val_score(make_pipeline(StandardScaler(), SelectKBest(mutual_info_classif, k=10), lr()), Xb, yb, cv=5).mean(), 3))
print("wrapper RFE->10:", round(cross_val_score(make_pipeline(StandardScaler(), RFE(lr(), n_features_to_select=10), lr()), Xb, yb, cv=5).mean(), 3))
print("embedded L1 logistic:", round(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(penalty="l1", solver="liblinear", C=.1)), Xb, yb, cv=5).mean(), 3))
sel = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=10)).fit(Xb, yb); print("noise features among top-10 picked:", int((sel[-1].get_support(indices=True) >= 30).sum()))

print("\n== Association rules (Apriori from scratch) on toy baskets ==")
from itertools import combinations
tx = [{"bread", "milk"}, {"bread", "diaper", "beer", "eggs"}, {"milk", "diaper", "beer", "cola"}, {"bread", "milk", "diaper", "beer"}, {"bread", "milk", "diaper", "cola"}]
sup = lambda s: sum(s <= t for t in tx) / len(tx)
items = sorted(set().union(*tx)); L1 = [frozenset([i]) for i in items if sup({i}) >= .4]
L2 = [frozenset(c) for c in combinations(items, 2) if sup(set(c)) >= .4]
print("frequent pairs (min sup 0.4):", [(sorted(c), sup(c)) for c in L2])
a, b = {"diaper"}, {"beer"}; print("diaper->beer support", sup(a | b), "confidence", round(sup(a | b) / sup(a), 3), "lift", round(sup(a | b) / (sup(a) * sup(b)), 3))
