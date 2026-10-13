"""Phase 0 demos: MLE, optimizers, bias-variance, leakage, calibration, preprocessing."""
import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, brier_score_loss, accuracy_score)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

rng = np.random.default_rng(0)

print("== MLE / MAP coin: 7 heads in 10 ==")
h, n = 7, 10
print("MLE", h / n, " MAP Beta(2,2):", (h + 1) / (n + 2))

print("\n== Optimizers on ill-conditioned quadratic f=0.5(x^2+25 y^2), start (5,5), 100 steps ==")
H = np.array([1.0, 25.0])
grad = lambda w: H * w
def run(step):
    w = np.array([5.0, 5.0]); st = {}
    for t in range(1, 101): w = step(w, grad(w), st, t)
    return 0.5 * (H * w * w).sum()
def sgd(w, g, st, t, lr=0.07): return w - lr * g
def mom(w, g, st, t, lr=0.02, b=0.9):
    v = b * st.get("v", 0) + g; st["v"] = v; return w - lr * v
def adam(w, g, st, t, lr=0.3, b1=0.9, b2=0.999, e=1e-8):
    m = b1 * st.get("m", 0) + (1 - b1) * g; v = b2 * st.get("v", 0) + (1 - b2) * g * g
    st["m"], st["v"] = m, v
    return w - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + e)
for nm, f in [("GD", sgd), ("Momentum", mom), ("Adam", adam)]: print(f"{nm:9s} final loss {run(f):.2e}")

print("\n== Bias-variance: noisy sine, degree vs train/test MSE (n=30) ==")
f = lambda x: np.sin(2 * np.pi * x)
xtr = rng.uniform(0, 1, 30); ytr = f(xtr) + rng.normal(0, .3, 30)
xte = rng.uniform(0, 1, 2000); yte = f(xte) + rng.normal(0, .3, 2000)
for d in [1, 3, 9]:
    m = make_pipeline(PolynomialFeatures(d), StandardScaler(), Ridge(1e-8)).fit(xtr[:, None], ytr)
    tr = np.mean((m.predict(xtr[:, None]) - ytr) ** 2); te = np.mean((m.predict(xte[:, None]) - yte) ** 2)
    print(f"degree {d:2d}: train {tr:.3f}  test {te:.3f}")
print("noise floor (sigma^2) = 0.090")

print("\n== Leakage: feature selection outside vs inside CV, pure-noise data (n=100,p=2000) ==")
X = rng.normal(size=(100, 2000)); y = rng.integers(0, 2, 100)
Xs = SelectKBest(f_classif, k=20).fit_transform(X, y)
print("WRONG (select on all data, then CV):", cross_val_score(LogisticRegression(max_iter=1000), Xs, y, cv=5).mean().round(3))
pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
print("RIGHT (select inside pipeline):     ", cross_val_score(pipe, X, y, cv=5).mean().round(3), "(truth = 0.5)")

print("\n== Imbalanced metrics: 2% positives ==")
X, y = make_classification(n_samples=20000, weights=[.98], n_informative=5, flip_y=.01, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=0)
rf = RandomForestClassifier(200, n_jobs=-1, random_state=0).fit(Xtr, ytr); p = rf.predict_proba(Xte)[:, 1]
print("accuracy of 'always 0':", round(1 - yte.mean(), 4))
print(f"RF acc {accuracy_score(yte,p>.5):.4f}  ROC-AUC {roc_auc_score(yte,p):.3f}  PR-AUC {average_precision_score(yte,p):.3f}")
for t in [.5, .3, .1]:
    print(f"thr {t}: precision {precision_score(yte,p>=t):.3f} recall {recall_score(yte,p>=t):.3f} F1 {f1_score(yte,p>=t):.3f}")

print("\n== Calibration (Brier, lower=better): RF raw vs isotonic ==")
cal = CalibratedClassifierCV(RandomForestClassifier(200, n_jobs=-1, random_state=0), method="isotonic", cv=3).fit(Xtr, ytr)
print("raw", round(brier_score_loss(yte, p), 4), " calibrated", round(brier_score_loss(yte, cal.predict_proba(Xte)[:, 1]), 4))
