"""From-scratch logistic regression (IRLS + GD), perceptron, Gaussian NB, kNN; checked against sklearn."""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sigmoid = lambda z: 1 / (1 + np.exp(-np.clip(z, -35, 35)))

class LogisticScratch:
    def __init__(self, lam=0.0, method="irls", lr=0.1, n_iter=500): self.lam, self.method, self.lr, self.n_iter = lam, method, lr, n_iter
    def fit(self, X, y):
        A = np.c_[np.ones(len(X)), X]; w = np.zeros(A.shape[1]); R = self.lam * np.eye(A.shape[1]); R[0, 0] = 0
        for _ in range(self.n_iter if self.method == "gd" else 25):
            p = sigmoid(A @ w); g = A.T @ (p - y) + R @ w
            if self.method == "gd": w -= self.lr * g / len(y)
            else:                                   # Newton / IRLS
                H = A.T @ (A * (p * (1 - p))[:, None]) + R
                step = np.linalg.solve(H, g); w -= step
                if np.abs(step).max() < 1e-10: break
        self.w = w; return self
    def predict_proba(self, X): return sigmoid(np.c_[np.ones(len(X)), X] @ self.w)
    def predict(self, X): return (self.predict_proba(X) >= .5).astype(int)

class PerceptronScratch:
    def fit(self, X, y, epochs=50):
        s = 2 * y - 1; w = np.zeros(X.shape[1] + 1); A = np.c_[np.ones(len(X)), X]
        for e in range(epochs):
            errs = 0
            for a, t in zip(A, s):
                if t * (a @ w) <= 0: w += t * a; errs += 1
            if errs == 0: break
        self.w, self.epochs_ = w, e + 1; return self
    def predict(self, X): return (np.c_[np.ones(len(X)), X] @ self.w > 0).astype(int)

class GaussianNBScratch:
    def fit(self, X, y):
        self.c = np.unique(y); self.pri = np.array([(y == k).mean() for k in self.c])
        self.mu = np.array([X[y == k].mean(0) for k in self.c]); self.var = np.array([X[y == k].var(0) + 1e-9 for k in self.c]); return self
    def predict(self, X):
        ll = -0.5 * (np.log(2 * np.pi * self.var)[None] + (X[:, None, :] - self.mu[None]) ** 2 / self.var[None]).sum(2) + np.log(self.pri)
        return self.c[ll.argmax(1)]

class KNNScratch:
    def __init__(self, k=5): self.k = k
    def fit(self, X, y): self.X, self.y = X, y; return self
    def predict(self, X):
        d = ((X[:, None, :] - self.X[None]) ** 2).sum(2); idx = np.argsort(d, 1)[:, :self.k]
        return (self.y[idx].mean(1) >= .5).astype(int)

if __name__ == "__main__":
    # worked example: one logistic GD step. x=[1,2,3,4], y=[0,0,1,1]-> hand-check first step
    x = np.array([[1.], [2.], [3.], [4.]]); y = np.array([0, 0, 1, 1])
    A = np.c_[np.ones(4), x]; w = np.zeros(2); p = sigmoid(A @ w); g = A.T @ (p - y) / 4
    print("p0", p, "grad", g, "w after lr=1 step", w - 1.0 * g)
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=0)
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    for n, m in [("Logistic IRLS", LogisticScratch(lam=1.0)), ("Logistic GD", LogisticScratch(method="gd", lr=.5, n_iter=2000, lam=1.0)),
                 ("Perceptron", PerceptronScratch()), ("GaussianNB", GaussianNBScratch()), ("kNN k=5", KNNScratch(5))]:
        print(f"{n:14s} test acc {(m.fit(Xtr, ytr).predict(Xte) == yte).mean():.3f}")
    sk = LogisticRegression(C=1.0, max_iter=5000).fit(Xtr, ytr)   # sklearn: C=1/lambda with sum-loss, same objective as lam=1
    print("sklearn logistic", round(sk.score(Xte, yte), 3), " max |coef diff| vs IRLS:", np.abs(sk.coef_[0] - LogisticScratch(lam=1.0).fit(Xtr, ytr).w[1:]).max().round(5))
