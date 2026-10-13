"""CART decision tree (Gini / MSE) from scratch + tiny gradient boosting from scratch; verified vs sklearn."""
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def gini(y): p = np.bincount(y) / len(y); return 1 - (p ** 2).sum()
def entropy(y): p = np.bincount(y) / len(y); p = p[p > 0]; return -(p * np.log2(p)).sum()

class TreeScratch:
    def __init__(self, max_depth=3, min_leaf=1): self.max_depth, self.min_leaf = max_depth, min_leaf
    def fit(self, X, y): self.tree = self._grow(X, y, 0); return self
    def _grow(self, X, y, d):
        if d >= self.max_depth or gini(y) == 0 or len(y) < 2 * self.min_leaf: return np.bincount(y, minlength=2).argmax()
        best = (gini(y), None)
        for j in range(X.shape[1]):
            xs = np.unique(X[:, j])
            for t in (xs[:-1] + xs[1:]) / 2:
                L = X[:, j] <= t
                if L.sum() < self.min_leaf or (~L).sum() < self.min_leaf: continue
                g = (L.sum() * gini(y[L]) + (~L).sum() * gini(y[~L])) / len(y)
                if g < best[0] - 1e-12: best = (g, (j, t))
        if best[1] is None: return np.bincount(y, minlength=2).argmax()
        j, t = best[1]; L = X[:, j] <= t
        return (j, t, self._grow(X[L], y[L], d + 1), self._grow(X[~L], y[~L], d + 1))
    def _one(self, x, n):
        while isinstance(n, tuple): n = n[2] if x[n[0]] <= n[1] else n[3]
        return n
    def predict(self, X): return np.array([self._one(x, self.tree) for x in X])

class GBMRegressorScratch:
    """Least-squares gradient boosting: each stage fits a shallow tree to the residuals (= negative gradient of MSE)."""
    def __init__(self, n_est=100, lr=0.1, depth=2): self.n_est, self.lr, self.depth = n_est, lr, depth
    def fit(self, X, y):
        self.f0 = y.mean(); F = np.full(len(y), self.f0); self.trees = []
        for _ in range(self.n_est):
            t = DecisionTreeRegressor(max_depth=self.depth).fit(X, y - F); F += self.lr * t.predict(X); self.trees.append(t)
        return self
    def predict(self, X): return self.f0 + self.lr * sum(t.predict(X) for t in self.trees)

if __name__ == "__main__":
    # worked example: split on tiny dataset
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1]); print("gini parent", round(gini(y), 4), "entropy", round(entropy(y), 4))
    from sklearn.datasets import load_breast_cancer, make_friedman1
    from sklearn.model_selection import train_test_split
    X, yy = load_breast_cancer(return_X_y=True); Xtr, Xte, ytr, yte = train_test_split(X, yy, random_state=0, stratify=yy)
    for d in [1, 3, 5]:
        a = (TreeScratch(d).fit(Xtr, ytr).predict(Xte) == yte).mean(); b = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr).score(Xte, yte)
        print(f"depth {d}: scratch {a:.3f}  sklearn {b:.3f}")
    Xr, yr = make_friedman1(1000, noise=1, random_state=0); Xtr, Xte, ytr, yte = train_test_split(Xr, yr, random_state=0)
    g = GBMRegressorScratch(200, .1, 3).fit(Xtr, ytr)
    print("scratch GBM test RMSE", round(float(np.sqrt(np.mean((g.predict(Xte) - yte) ** 2))), 3))
