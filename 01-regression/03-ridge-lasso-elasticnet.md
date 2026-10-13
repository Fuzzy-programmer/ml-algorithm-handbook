# Chapter 1.3 — Ridge, Lasso and Elastic Net

[← Roadmap](../README.md) · Prev [1.2](02-polynomial-regression.md) · Next [1.4 Quantile & Bayesian](04-quantile-bayesian-regression.md) · Priority 🔴 · Code: [`regression_family.py`](../code/regression_family.py)

## 1. Identity & problem
Penalized linear regression. Problem: OLS has huge variance when features are many (p near/above n) or collinear, and gives no feature selection.

## 2. History
- Tikhonov regularization (1940s–60s) for ill-posed inverse problems.
- **Ridge** — Hoerl & Kennard, *Technometrics* 1970: add `λI` to `XᵀX` to stabilize.
- **Lasso** — Tibshirani, *JRSS B* 1996 (also "basis pursuit" in signal processing, Chen–Donoho–Saunders 1998): L1 penalty gives exact zeros.
- **LARS** (Efron et al. 2004) computes the whole lasso path; coordinate descent (Friedman, Hastie, Tibshirani 2010, `glmnet`) is the practical solver.
- **Elastic Net** — Zou & Hastie 2005: combine penalties to fix Lasso's behavior with correlated features.

## 3. Intuition
Add a "budget" on coefficient size. Ridge shrinks all coefficients smoothly toward 0; Lasso's diamond-shaped constraint touches the loss contours at corners, so some coefficients become exactly 0.
```
 β2 │   Ridge: circle      Lasso: diamond (corners on axes ⇒ zeros)
    │      ___                 /\
    │    /     \              /  \   contours of the loss hit
    │   |   •   |            <  • >  the corner → β1 = 0
    │    \_____/              \  /
    └──────── β1               \/
```

## 4. Math
- Ridge: `min ‖y−Xβ‖² + λ‖β‖₂²` → `β̂ = (XᵀX + λI)⁻¹Xᵀy` (always invertible for λ>0).
- Lasso: `min ½n‖y−Xβ‖² + α‖β‖₁` (no closed form; convex).
- Elastic Net: `α[ρ‖β‖₁ + ½(1−ρ)‖β‖₂²]`, `ρ` = `l1_ratio`.
- Bayesian view: Ridge = MAP with Gaussian prior; Lasso = Laplace prior (Ch 0.2).
- **Bias–variance:** ridge variance `σ²Σ dᵢ²/(dᵢ²+λ)²` (singular values `dᵢ`) — small directions get shrunk most.
- **Degrees of freedom:** ridge `Σ dᵢ²/(dᵢ²+λ)` (≤ p); lasso ≈ number of nonzero coefficients.

**Worked example** (data of Ch 1.1, centered x so intercept separates): `Sxy=6`, `Sxx=10`.
- Ridge: `β₁ = Sxy/(Sxx+λ) = 6/(10+λ)`: λ=0 → 0.60; λ=2 → 0.50; λ=10 → 0.30. Never exactly 0.
- Lasso (soft-threshold, loss `½‖·‖² + λ|β|`): `β₁ = max(Sxy−λ, 0)/Sxx = (6−λ)/10`: λ=3 → 0.30; λ ≥ 6 → exactly **0**.
That contrast — shrink vs. zero out — is the whole difference between the penalties.

## 5. Training
Ridge: closed form `O(p³)` or SVD, or SAG/LSQR for big data. Lasso/ENet: coordinate descent (cycle over features applying soft-thresholding), warm starts along a λ path. **Standardize features first** — penalties depend on scale. Don't penalize the intercept.

## 6. Evolution & variants
| Method | Improvement | Cost |
|---|---|---|
| Ridge | Stable under collinearity; works when p>n | Keeps all features; biased |
| Lasso | Sparsity/selection; interpretable | Picks one of a correlated group arbitrarily; at most n nonzeros when p>n; biased large coefs |
| Elastic Net | Groups correlated features + sparsity | Two hyperparameters |
| Adaptive lasso, relaxed lasso | Reduced bias | Extra stage |
| Group lasso, fused lasso | Structured sparsity | Specialized |
| Stability selection | Reliable selection | Compute |
Note: Lasso is not "better Ridge" — when most features carry small signal, ridge often predicts better.

## 7. When to use / not
Use: p large relative to n, collinearity, need a compact model (Lasso/ENet), strong baseline for sparse high-dimensional data (text TF-IDF, genomics). Avoid: tiny p ≪ n clean data (OLS suffices), strongly nonlinear problems, when inference/p-values are needed (penalized estimates complicate standard errors).

## 8. Assumptions, strengths, weaknesses
Assumes roughly linear signal and standardized inputs; sparse truth for Lasso. Strengths: convex, fast, regularization path, built-in CV estimators. Weaknesses: biased coefficients, λ selection required, selection unstable in correlated settings.

## 9. Hyperparameters
`alpha` (λ): ↑ ⇒ more shrinkage, bias↑ variance↓ (choose by CV on a log grid; `RidgeCV`/`LassoCV`/`ElasticNetCV`). `l1_ratio`: 0 = Ridge, 1 = Lasso. `max_iter`/`tol` (coordinate descent convergence; warnings usually mean "scale features or raise max_iter"). `fit_intercept`, `positive`, `selection` (`cyclic`/`random`).

## 10. Implementation
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
import numpy as np
ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 40)))
lasso = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=0))
enet  = make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[.2, .5, .8], cv=5, random_state=0))
```
From scratch (ridge): `w = np.linalg.solve(X.T@X + lam*np.eye(p), X.T@y)` on centered data. Lasso coordinate update: `w_j ← S(ρ_j, λ)/z_j`, `S(a,λ)=sign(a)·max(|a|−λ,0)`.

## 11. Evaluation (measured)
Synthetic: n=80, p=60 (10 informative + 10 near-copies of them as decoys), noise 20; 5-fold CV RMSE.
| Model | CV RMSE | Nonzero coefficients |
|---|---|---|
| OLS | 377.3 | 60 |
| RidgeCV | 94.1 | 60 |
| **LassoCV** | **62.3** | 15 |
| ElasticNetCV | 88.7 | 54 |
Interpretation: with p≈n OLS is catastrophic; both penalties fix it. Lasso won *because the truth here is sparse*; ElasticNetCV (restricted to l1_ratio ≤ 0.8 in my grid) leaned toward ridge-like solutions and landed between. On a dense-signal problem the ranking can reverse — always CV all three.
Ridge shrinkage on the diabetes data (standardized): ‖w‖₂ = 65.4 (α=0.01) → 57.5 (1) → 42.6 (10) → 34.8 (100) → 17.4 (1000).

## 12. Case study suggestion
Predict diabetes progression (sklearn `load_diabetes`): plot coefficient paths vs α (`lasso_path`), choose α by CV, report which features survive. Compare against Ch 1.1 OLS and boosted trees.

## 13. Decision guide
Collinear/dense → Ridge · want selection → Lasso · correlated groups + selection → Elastic Net · coefficient interpretability with inference → OLS/statsmodels or debiased lasso · nonlinear → trees/NNs (they also benefit from weight decay).

## Interview Q&A
- *Why does L1 produce sparsity?* Constraint region has corners on the axes; optimum at a corner zeroes coefficients (also soft-threshold view).
- *Why standardize?* Penalty treats all coefficients equally; scale would decide who gets penalized.
- *Lasso with correlated features?* Picks one arbitrarily; ENet keeps groups.
- *Ridge when p>n?* Still solvable since `XᵀX+λI` is invertible.

## Exercises
Derive ridge solution; implement coordinate-descent lasso and match `sklearn`; plot regularization paths; test the sparse vs dense-truth reversal.

## Further reading
Hoerl & Kennard (1970); Tibshirani (1996); Zou & Hastie (2005); Friedman, Hastie, Tibshirani (2010) "Regularization paths for GLMs via coordinate descent"; ESL Ch 3.4; Hastie, Tibshirani, Wainwright, *Statistical Learning with Sparsity* (free).
