"""Linear regression from scratch: normal equation, QR/lstsq, and gradient descent."""
import numpy as np


class LinearRegressionScratch:
    def __init__(self, method="lstsq", lr=0.05, n_iter=5000, fit_intercept=True):
        self.method, self.lr, self.n_iter, self.fit_intercept = method, lr, n_iter, fit_intercept

    def _design(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        return np.c_[np.ones(len(X)), X] if self.fit_intercept else X

    def fit(self, X, y):
        A, y = self._design(X), np.asarray(y, dtype=float)
        if self.method == "normal":      # (A^T A)^-1 A^T y  -- textbook, numerically fragile
            w = np.linalg.solve(A.T @ A, A.T @ y)
        elif self.method == "lstsq":     # SVD-based, what you should use
            w, *_ = np.linalg.lstsq(A, y, rcond=None)
        elif self.method == "gd":        # batch gradient descent on MSE
            w = np.zeros(A.shape[1])
            for _ in range(self.n_iter):
                w -= self.lr * (2 / len(y)) * A.T @ (A @ w - y)
        else:
            raise ValueError(self.method)
        if self.fit_intercept:
            self.intercept_, self.coef_ = w[0], w[1:]
        else:
            self.intercept_, self.coef_ = 0.0, w
        return self

    def predict(self, X):
        return self._design(X) @ np.r_[self.intercept_ if self.fit_intercept else [], self.coef_]

    def score(self, X, y):               # R^2
        y = np.asarray(y, dtype=float)
        return 1 - ((y - self.predict(X)) ** 2).sum() / ((y - y.mean()) ** 2).sum()


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 5]); y = np.array([2, 4, 5, 4, 5])
    for m in ["normal", "lstsq", "gd"]:
        lr = LinearRegressionScratch(method=m).fit(x, y)
        print(f"{m:7s} b0={lr.intercept_:.4f} b1={lr.coef_[0]:.4f} R2={lr.score(x, y):.4f}")
    # Why feature scaling matters for GD: lr=0.1 is too big for unscaled x and diverges
    bad = LinearRegressionScratch(method="gd", lr=0.1, n_iter=50).fit(x, y)
    print("lr=0.1 unscaled -> b1 =", f"{bad.coef_[0]:.3g}", "(diverged)")
    xs = (x - x.mean()) / x.std()
    ok = LinearRegressionScratch(method="gd", lr=0.1, n_iter=200).fit(xs, y)
    print(f"lr=0.1 scaled   -> b0={ok.intercept_:.4f} b1={ok.coef_[0]:.4f} (converged)")
