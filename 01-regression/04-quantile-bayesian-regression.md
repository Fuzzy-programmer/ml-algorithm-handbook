# Chapter 1.4 — Quantile, Robust and Bayesian Regression

[← Roadmap](../README.md) · Prev [1.3](03-ridge-lasso-elasticnet.md) · Next [2.1 Logistic Regression](../02-classification/01-logistic-regression.md) · Priority 🟡 · Code: [`regression_family.py`](../code/regression_family.py)

## 1. Problem
OLS predicts the *conditional mean* with squared loss. Sometimes you need (a) the median or another quantile, (b) robustness to outliers, or (c) honest uncertainty about parameters and predictions.

## 2. History
Laplace's least-absolute-deviation (18th c.) predates least squares; Huber's robust M-estimation (1964); **Koenker & Bassett (1978)** quantile regression; Bayesian linear regression with conjugate priors (Lindley & Smith 1972; MacKay 1992 evidence framework; Tipping 2001 RVM).

## 3. Quantile regression
- **Pinball loss** for quantile τ: `ρ_τ(r) = r(τ − 1[r<0])`, i.e. weight τ on under-prediction, (1−τ) on over-prediction. τ=0.5 ⇒ absolute error (median regression).
- Minimizing it estimates `Q_τ(y|x)`. Solved by linear programming. Different τ give different lines ⇒ prediction intervals without assuming Gaussian noise; handles heteroscedasticity.
- **Worked example:** targets `[1,2,3,4,100]`, constant predictor. Squared loss minimizer = mean = 22; absolute loss minimizer = median = 3; τ=0.9 minimizer = the 90th percentile (≈ 100 for 5 points). The outlier ruins the mean, not the median.
- Measured (`regression_family.py`): 2000 points with noise growing with x; the fitted 10th–90th percentile band covered **0.80** of observations (target 0.80) ✓.

## 4. Robust regression
- **Huber loss:** quadratic for `|r|≤δ`, linear beyond ⇒ outliers get bounded influence. `HuberRegressor`.
- **RANSAC** (Fischler & Bolles 1981): fit on random subsets, keep the consensus set. **Theil–Sen:** median of pairwise slopes.
- Measured: 100 points, true slope 2, 8 outliers (+40): OLS slope 2.07, Huber 2.00, median regression 2.00. The slope barely moved because the outliers were all on the same side and mostly at low x; the big damage was to the intercept/residual scale, so check *both* — and don't assume OLS always breaks visibly.

## 5. Bayesian linear regression
Prior `w~𝒩(0, α⁻¹I)`, noise precision β. Posterior is Gaussian: `S⁻¹ = αI + βXᵀX`, `m = βSXᵀy`. Predictive variance at x: `1/β + xᵀSx` — **it grows away from the training data**. Measured with `BayesianRidge`: predictive std 0.277 at x=0, 0.293 at x=1 (edge of data), **0.546 at x=5** (extrapolation). α, β are learned by maximizing marginal likelihood (evidence). Posterior mean = Ridge with λ=α/β.
Extensions: Gaussian Processes (infinite basis), ARD (automatic relevance determination), full MCMC (PyMC, Stan) for non-conjugate models.

## 6. When to use / not
| Need | Use |
|---|---|
| Median / percentiles / prediction intervals, skewed or heteroscedastic noise | Quantile regression (or quantile gradient boosting) |
| Outliers in y | Huber, RANSAC, median regression |
| Uncertainty with small data, priors from domain knowledge | Bayesian regression |
| Big data, simple mean prediction | OLS/Ridge |
Avoid: quantile regression crossing (separate quantile fits can cross — use monotone fixes/rearrangement); Bayesian methods when you cannot afford MCMC and conjugacy doesn't hold; robust losses when "outliers" are real, valuable extremes (e.g. peak demand).

## 7. Hyperparameters
Quantile: `quantile`, `alpha` (L1 penalty), `solver` (`highs`). Huber: `epsilon` (δ, default 1.35 ≈ 95% efficiency under Gaussian), `alpha`. BayesianRidge: priors `alpha_1/2`, `lambda_1/2`, `n_iter`.

## 8. Implementation
```python
from sklearn.linear_model import QuantileRegressor, HuberRegressor, BayesianRidge
lo = QuantileRegressor(quantile=0.1, alpha=0, solver="highs").fit(X, y)
hi = QuantileRegressor(quantile=0.9, alpha=0, solver="highs").fit(X, y)
huber = HuberRegressor().fit(X, y)
br = BayesianRidge().fit(X, y); mean, std = br.predict(Xnew, return_std=True)
```
Evaluate quantile models with pinball loss (`mean_pinball_loss`) and **coverage** (does the 80% band contain ~80%?). Evaluate Bayesian predictions with log-likelihood / calibration, not just RMSE.

## 9. Comparison
Mean (OLS) vs median (QR): same if noise symmetric; QR safer under skew/outliers, slower. Conformal prediction (Vovk; Angelopoulos & Bates tutorial) is a model-agnostic alternative for distribution-free intervals.

## Interview Q&A
- *Why is median regression robust?* Absolute loss grows linearly so each point has bounded influence.
- *How get a prediction interval from boosted trees?* Train quantile models (loss="quantile") or use conformal prediction.
- *Bayesian vs Ridge?* Same point estimate (given α,β), but Bayesian gives a predictive distribution.

## Exercises
Fit τ=0.1,0.5,0.9 lines on the diabetes data and test crossing; derive the Huber influence function; compute the Bayesian posterior by hand for 2 points.

## Further reading
Koenker & Bassett (1978); Koenker, *Quantile Regression*; Huber & Ronchetti, *Robust Statistics*; Bishop PRML Ch 3; Angelopoulos & Bates (2021) "A gentle introduction to conformal prediction."
