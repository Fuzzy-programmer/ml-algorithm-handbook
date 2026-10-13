"""Polynomial, Ridge, Lasso, ElasticNet, Quantile, Bayesian ridge."""
import numpy as np
from sklearn.datasets import make_regression, load_diabetes
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV,
                                  ElasticNetCV, QuantileRegressor, BayesianRidge, HuberRegressor)
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, SplineTransformer

rng = np.random.default_rng(0)
cv = KFold(5, shuffle=True, random_state=0)
print("== Polynomial on noisy sine, n=40, CV RMSE ==")
x = rng.uniform(0, 1, 40)[:, None]; y = np.sin(2*np.pi*x[:, 0]) + rng.normal(0, .3, 40)
for d in [1, 3, 5, 10, 15]:
    m = make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression())
    print(f"deg {d:2d}: {-cross_val_score(m, x, y, cv=cv, scoring='neg_root_mean_squared_error').mean():.3f}")
m = make_pipeline(PolynomialFeatures(15), StandardScaler(), RidgeCV(np.logspace(-4, 2, 30)))
print("deg 15 + RidgeCV:", round(-cross_val_score(m, x, y, cv=cv, scoring='neg_root_mean_squared_error').mean(), 3))
m = make_pipeline(SplineTransformer(n_knots=6, degree=3), Ridge(1e-2))
print("cubic spline + Ridge:", round(-cross_val_score(m, x, y, cv=cv, scoring='neg_root_mean_squared_error').mean(), 3))

print("\n== Ridge/Lasso/ENet: n=80,p=60 (10 informative), correlated noise features ==")
X, y, coef = make_regression(80, 60, n_informative=10, noise=20, coef=True, random_state=1)
X[:, 10:20] = X[:, 0:10] + 0.1*rng.normal(size=(80, 10))   # decoys correlated with informative features
models = {"OLS": LinearRegression(), "RidgeCV": RidgeCV(np.logspace(-2, 4, 40)),
          "LassoCV": LassoCV(cv=5, random_state=0), "ElasticNetCV": ElasticNetCV(l1_ratio=[.2, .5, .8], cv=5, random_state=0)}
for n, mdl in models.items():
    pipe = make_pipeline(StandardScaler(), mdl)
    r = -cross_val_score(pipe, X, y, cv=cv, scoring='neg_root_mean_squared_error').mean()
    pipe.fit(X, y); w = pipe[-1].coef_
    print(f"{n:13s} CV RMSE {r:6.1f}  nonzero coefs {np.sum(np.abs(w) > 1e-6)}")

print("\n== Ridge shrinkage path (diabetes) ==")
Xd, yd = load_diabetes(return_X_y=True); Xs = StandardScaler().fit_transform(Xd)
for a in [0.01, 1, 10, 100, 1000]:
    print(f"alpha {a:7}: ||w||2 = {np.linalg.norm(Ridge(alpha=a).fit(Xs, yd).coef_):.1f}")

print("\n== Outliers: OLS vs Huber vs Quantile(median); true slope 2 ==")
x = rng.uniform(0, 10, 100)[:, None]; y = 2*x[:, 0] + 1 + rng.normal(0, 1, 100); y[:8] += 40
for n, mdl in [("OLS", LinearRegression()), ("Huber", HuberRegressor()), ("Median(QR)", QuantileRegressor(quantile=.5, alpha=0, solver='highs'))]:
    print(f"{n:11s} slope {mdl.fit(x, y).coef_[0]:.2f}")

print("\n== Quantile intervals under heteroscedastic noise ==")
x = rng.uniform(0, 10, 2000)[:, None]; y = 2*x[:, 0] + rng.normal(0, 0.3 + 0.4*x[:, 0])
lo = QuantileRegressor(quantile=.1, alpha=0, solver='highs').fit(x, y); hi = QuantileRegressor(quantile=.9, alpha=0, solver='highs').fit(x, y)
inside = ((y >= lo.predict(x)) & (y <= hi.predict(x))).mean(); print("coverage of 10-90% band:", round(inside, 3))

print("\n== BayesianRidge predictive std grows away from data ==")
x = rng.uniform(-1, 1, 30)[:, None]; y = 3*x[:, 0] + rng.normal(0, .3, 30)
br = BayesianRidge().fit(x, y)
for q in [0.0, 1.0, 5.0]: print(f"x={q}: std {br.predict([[q]], return_std=True)[1][0]:.3f}")
