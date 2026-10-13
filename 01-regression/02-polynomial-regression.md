# Chapter 1.2 — Polynomial Regression & Basis Expansion

[← Roadmap](../README.md) · Prev [1.1 OLS](01-linear-regression.md) · Next [1.3 Ridge/Lasso/ENet](03-ridge-lasso-elasticnet.md) · Priority 🟠 · Code: [`regression_family.py`](../code/regression_family.py)

## 1. Identity & problem
Family: supervised regression, linear model on **transformed features**. Problem: the relationship between `x` and `y` is curved (dose–response, depreciation, physics-like trends) but you still want OLS's speed and interpretability.

## 2. History & motivation
Fitting polynomials to data predates modern statistics (e.g. Gergonne 1815). OLS (Ch 1.1) cannot bend. The key insight: the model only needs to be linear in the **parameters**, so you can feed it `φ(x) = [1, x, x², …, x^d]`. Splines (Schoenberg 1946; de Boor), GAMs (Hastie & Tibshirani 1986) and kernel methods generalize this idea.

## 3. Intuition
Same machine as OLS, but each original feature is replaced by a menu of shapes. More shapes ⇒ more flexible curve ⇒ more risk of wiggling through noise.

## 4. Math
`ŷ = Σ_{j=0}^d β_j x^j`, design matrix `Φ` with columns `x^j`; solve `β̂ = argmin ‖y − Φβ‖²` exactly as OLS. For multiple features, degree-2 expansion adds squares and pairwise interactions: p features ⇒ `C(p+d, d)` terms (p=10, d=3 ⇒ 286).
**Worked example.** `x=[−1,0,1]`, `y=[1,0,1]`. A line gives slope 0, intercept 2/3 (RMS error 0.47). With `φ=[1,x,x²]` the exact fit is `β=[0,0,1]` (`y=x²`), zero error — but 3 points and 3 parameters is always an exact fit, so this proves nothing about generalization (see §10).
**Numerical issue:** columns `x, x², x³…` are nearly collinear (huge condition number). Standardize first, or use orthogonal polynomials (Legendre/Chebyshev) or B-splines.

## 5. Training
Identical to OLS (QR/SVD). Cost `O(n·m²)` where `m` = number of expanded features, which grows combinatorially with degree and features.

## 6. Evolution
| Method | Fixes | Cost |
|---|---|---|
| Global polynomial | Curvature | Runge oscillation at edges; poor extrapolation |
| Piecewise polynomials / **splines** (`SplineTransformer`) | Local flexibility, stable edges | Knot choice |
| Smoothing splines, **GAMs** | Per-feature smooth shapes with penalty | More machinery |
| Kernel ridge / SVR / GP | Implicit infinite basis | O(n³) |
| Trees/boosting/NNs | Learn interactions automatically | Less interpretable, need data |

## 7. When to use / not
Use: 1–3 features, smooth monotone or single-bend relationships, need an equation, small data, physics-style trends. Avoid: many features (explosion), extrapolation, noisy data with high degree, discontinuities (use trees/splines with knots), periodic data (use Fourier/sin-cos features).

## 8. Assumptions, strengths, weaknesses
Assumes smooth function, OLS error assumptions. Strengths: simple, interpretable, convex. Weaknesses: global (a change at one end affects the other), unstable at high degree, extrapolates wildly (`x^d` explodes).

## 9. Hyperparameters
`degree` (main bias–variance dial), `interaction_only`, `include_bias`, regularization strength (Ridge on top is the standard fix), spline `n_knots`/`degree`.

## 10. Implementation
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, SplineTransformer
from sklearn.linear_model import LinearRegression, RidgeCV
import numpy as np
poly = make_pipeline(PolynomialFeatures(3), StandardScaler(), LinearRegression())
reg_poly = make_pipeline(PolynomialFeatures(15), StandardScaler(), RidgeCV(np.logspace(-4, 2, 30)))
spline = make_pipeline(SplineTransformer(n_knots=6, degree=3), LinearRegression())
```
From scratch: build `Phi = np.vander(x, d+1, increasing=True)` then reuse `LinearRegressionScratch` from Ch 1.1.

## 11. Evaluation (measured; 40 noisy-sine points, 5-fold CV RMSE; noise σ=0.3)
| Model | CV RMSE |
|---|---|
| Degree 1 | 0.615 |
| **Degree 3** | **0.335** |
| Degree 5 | 0.374 |
| Degree 10 | 0.424 |
| Degree 15 | 0.432 |
| Degree 15 + RidgeCV | 0.370 |
| Cubic spline + Ridge(0.01) | 0.390 |
Read: the U-shape is textbook — error is minimized near degree 3, then rises. Regularization recovers much (0.432 → 0.370) but didn't beat the right low degree here; the spline was not tuned (fixed knots/alpha), so don't read it as "splines lose." **Use CV to choose degree; training error alone always favors the highest degree.**

## 12. Case study suggestion
Fit mpg vs horsepower (Auto-MPG) with degrees 1–5 using CV; you will see a curved relationship and diminishing returns past degree 2–3. Plot the fit *and* its extrapolation to show why predictions beyond the data range are dangerous.

## 13. Decision guide
Curved + few features + need interpretability → polynomial/spline + Ridge. Many features/interactions → boosted trees. Periodic → Fourier terms. Need uncertainty → Gaussian process/Bayesian.

## Interview Q&A
- *Is polynomial regression non-linear regression?* It's nonlinear in `x`, linear in parameters, so OLS machinery applies.
- *Why scale before expanding?* Powers differ by orders of magnitude, hurting conditioning and penalties.
- *Why do high-degree fits oscillate?* Runge's phenomenon; the basis is global and ill-conditioned.

## Exercises
Show interaction terms explosion for p=20,d=3; compare degree selection via CV vs AIC; replace polynomials with a Fourier basis on a seasonal series.

## Further reading
ESL Ch 5 (basis expansions and splines); Runge (1901); de Boor, *A Practical Guide to Splines*; sklearn `PolynomialFeatures`, `SplineTransformer` docs.
