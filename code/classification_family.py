"""Logistic, perceptron, NB, LDA/QDA, kNN, SVM on breast-cancer + moons + text."""
import time, numpy as np
from sklearn.datasets import load_breast_cancer, make_moons, make_classification, fetch_20newsgroups
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

cv = StratifiedKFold(5, shuffle=True, random_state=0)
X, y = load_breast_cancer(return_X_y=True)
def auc(m, X=X, y=y): return cross_val_score(m, X, y, cv=cv, scoring="roc_auc").mean()
def acc(m, X=X, y=y): return cross_val_score(m, X, y, cv=cv, scoring="accuracy").mean()

print("== Breast cancer (569x30), 5-fold CV ==")
rows = {
 "majority baseline": DummyClassifier(),
 "Logistic (C=1)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
 "Perceptron": make_pipeline(StandardScaler(), Perceptron(random_state=0)),
 "GaussianNB": GaussianNB(),
 "LDA": LDA(), "QDA (reg_param=0.1)": QDA(reg_param=0.1),
 "kNN k=5 (scaled)": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
 "kNN k=5 (UNscaled)": KNeighborsClassifier(5),
 "NearestCentroid": make_pipeline(StandardScaler(), NearestCentroid()),
 "LinearSVC": make_pipeline(StandardScaler(), LinearSVC(max_iter=20000)),
 "RBF SVC": make_pipeline(StandardScaler(), SVC()),
}
for n, m in rows.items():
    a = acc(m)
    try: u = f"{auc(m):.3f}"
    except Exception: u = "  n/a"
    print(f"{n:21s} acc {a:.3f}  auc {u}")

print("\n== Logistic C sweep (strength of regularization = 1/C) ==")
for C in [1e-4, 1e-2, 1, 100]:
    print(f"C={C:<7}: acc {acc(make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))):.3f}")

print("\n== Moons (nonlinear), n=500 noise .25 ==")
Xm, ym = make_moons(500, noise=.25, random_state=0)
for n, m in {"Logistic": LogisticRegression(), "LDA": LDA(), "GaussianNB": GaussianNB(), "kNN k=15": KNeighborsClassifier(15),
             "Linear SVC": LinearSVC(), "RBF SVC": SVC(gamma="scale")}.items():
    print(f"{n:11s} acc {acc(m, Xm, ym):.3f}")

print("\n== kNN k sweep on moons (train acc vs CV) ==")
for k in [1, 5, 15, 50, 200]:
    m = KNeighborsClassifier(k).fit(Xm, ym)
    print(f"k={k:3d}: train {m.score(Xm, ym):.3f}  cv {acc(KNeighborsClassifier(k), Xm, ym):.3f}")

print("\n== SVM RBF grid on breast cancer ==")
gs = GridSearchCV(make_pipeline(StandardScaler(), SVC()), {"svc__C": [.1, 1, 10, 100], "svc__gamma": [1e-3, 1e-2, 1e-1, 1]}, cv=cv).fit(X, y)
print("best", gs.best_params_, round(gs.best_score_, 3))
print("extreme overfit (C=1000,gamma=1):", round(acc(make_pipeline(StandardScaler(), SVC(C=1000, gamma=1))), 3))

print("\n== Scaling with n: fit time (s) on 2D-blob-ish data, 20 features ==")
for n in [2000, 8000, 16000]:
    Xs, ys = make_classification(n, 20, random_state=0)
    t = {}
    for nm, m in [("Logistic", LogisticRegression(max_iter=500)), ("RBF SVC", SVC()), ("GaussianNB", GaussianNB())]:
        s = time.time(); m.fit(Xs, ys); t[nm] = time.time() - s
    print(n, {k: round(v, 3) for k, v in t.items()})

print("\n== Text: 4 newsgroups, TF-IDF + linear models ==")
try:
    cats = ["sci.space", "rec.autos", "talk.politics.misc", "comp.graphics"]
    tr = fetch_20newsgroups(subset="train", categories=cats, remove=("headers", "footers", "quotes"))
    te = fetch_20newsgroups(subset="test", categories=cats, remove=("headers", "footers", "quotes"))
    for n, m in {"MultinomialNB": make_pipeline(CountVectorizer(), MultinomialNB()),
                 "TFIDF+Logistic": make_pipeline(TfidfVectorizer(sublinear_tf=True), LogisticRegression(max_iter=2000)),
                 "TFIDF+LinearSVC": make_pipeline(TfidfVectorizer(sublinear_tf=True), LinearSVC())}.items():
        print(f"{n:16s} test acc {m.fit(tr.data, tr.target).score(te.data, te.target):.3f}")
except Exception as e:
    print("20newsgroups unavailable offline:", type(e).__name__)
