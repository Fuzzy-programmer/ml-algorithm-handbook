"""Causal inference (confounding, adjustment, IPW, matching, DiD), survival (KM, Cox), semi-supervised (self-training, label propagation)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import GradientBoostingRegressor
rng = np.random.default_rng(0)

print("== Causal: confounded treatment, true ATE = 2.0 ==")
n = 20000; x = rng.normal(size=n); ps_true = 1 / (1 + np.exp(-1.5 * x)); t = (rng.random(n) < ps_true).astype(int)
y = 2.0 * t + 3.0 * x + rng.normal(size=n)
print("naive diff in means:", round(y[t == 1].mean() - y[t == 0].mean(), 3))
X = np.c_[t, x]; print("regression adjustment (T + x):", round(LinearRegression().fit(X, y).coef_[0], 3))
e = LogisticRegression().fit(x[:, None], t).predict_proba(x[:, None])[:, 1]
ipw = np.mean(t * y / e) - np.mean((1 - t) * y / (1 - e)); print("IPW (estimated propensity):", round(ipw, 3))
ipw_c = np.mean(np.clip(t / e, 0, 20) * y) - np.mean(np.clip((1 - t) / (1 - e), 0, 20) * y)
print("overlap: propensity range", e.min().round(3), e.max().round(3))
tr, ct = np.where(t == 1)[0], np.where(t == 0)[0]
nn = NearestNeighbors(n_neighbors=1).fit(x[ct, None]); idx = ct[nn.kneighbors(x[tr, None])[1][:, 0]]
print("matching on x (ATT, 1-NN):", round(np.mean(y[tr] - y[idx]), 3))
# omitted confounder
u = rng.normal(size=n); t2 = (rng.random(n) < 1 / (1 + np.exp(-1.5 * u))).astype(int); y2 = 2 * t2 + 3 * u + rng.normal(size=n)
print("unobserved confounder -> regression (T only visible):", round(LinearRegression().fit(t2[:, None], y2).coef_[0], 3), "(biased; no adjustment can fix)")
# DiD
print("\n== Difference-in-differences: true effect 5 ==")
m = 5000; grp = rng.integers(0, 2, m); a0 = 10 + 4 * grp + rng.normal(size=m); a1 = a0 + 3 + 5 * grp + rng.normal(size=m)   # common trend +3, level gap 4, effect 5
print("post-only difference:", round(a1[grp == 1].mean() - a1[grp == 0].mean(), 2), "| DiD:", round((a1[grp == 1] - a0[grp == 1]).mean() - (a1[grp == 0] - a0[grp == 0]).mean(), 2))
# T-learner CATE
print("\n== Heterogeneous effect (T-learner), effect = 1 + 2*x ==")
xs = rng.uniform(-1, 1, 20000); ts = rng.integers(0, 2, 20000); ys = 1 + xs + ts * (1 + 2 * xs) + rng.normal(0, .5, 20000)
m1 = GradientBoostingRegressor(max_depth=2).fit(xs[ts == 1, None], ys[ts == 1]); m0 = GradientBoostingRegressor(max_depth=2).fit(xs[ts == 0, None], ys[ts == 0])
for q in [-0.8, 0.0, 0.8]: print(f"x={q:+.1f}: true {1 + 2*q:+.2f}  est {(m1.predict([[q]]) - m0.predict([[q]]))[0]:+.2f}")

print("\n== Survival ==")
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.datasets import load_rossi
T = np.array([2, 3, 3, 5, 6, 8]); E = np.array([1, 1, 0, 1, 0, 1])
S = 1.0; order = sorted(set(T[E == 1]))
for tt in order:
    at_risk = (T >= tt).sum(); d = ((T == tt) & (E == 1)).sum(); S *= 1 - d / at_risk; print(f"t={tt}: at risk {at_risk}, events {d}, S(t)={S:.4f}")
km = KaplanMeierFitter().fit(T, E); print("lifelines KM S(5) =", round(float(km.predict(5)), 4))
df = load_rossi(); cph = CoxPHFitter().fit(df, "week", "arrest")
print(cph.summary[["coef", "exp(coef)", "p"]].round(3).head(4)); print("concordance index:", round(cph.concordance_index_, 3))

print("\n== Semi-supervised: digits with 30 labels ==")
from sklearn.datasets import load_digits
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
from sklearn.svm import SVC
Xd, yd = load_digits(return_X_y=True); Xd = Xd / 16; r = np.random.default_rng(1); perm = r.permutation(len(Xd)); Xtr, ytr, Xte, yte = Xd[perm[:1300]], yd[perm[:1300]], Xd[perm[1300:]], yd[perm[1300:]]
lab = np.concatenate([r.choice(np.where(ytr == c)[0], 3, replace=False) for c in range(10)]); yp = np.full(len(ytr), -1); yp[lab] = ytr[lab]
print("supervised on 30 labels (SVC):", round(SVC().fit(Xtr[lab], ytr[lab]).score(Xte, yte), 3))
print("self-training SVC:", round(SelfTrainingClassifier(SVC(probability=True), threshold=.8).fit(Xtr, yp).score(Xte, yte), 3))
print("LabelSpreading (knn):", round(LabelSpreading(kernel="knn", n_neighbors=10).fit(Xtr, yp).score(Xte, yte), 3))
print("upper bound, all 1300 labels SVC:", round(SVC().fit(Xtr, ytr).score(Xte, yte), 3))
