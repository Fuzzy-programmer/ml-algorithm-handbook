# Chapter 3.3 — AdaBoost and Gradient Boosting

[← Roadmap](../README.md) · Prev [3.2](02-bagging-random-forest.md) · Next [3.4 XGBoost/LightGBM/CatBoost/HistGB](04-xgboost-lightgbm-catboost.md) · Priority 🔴 · Code: [`trees_scratch.py`](../code/trees_scratch.py), [`trees_family.py`](../code/trees_family.py)

## 1. Identity & problem
**Boosting** builds an ensemble sequentially: each new weak learner focuses on what the ensemble still gets wrong. Reduces *bias* (and variance) by stage-wise additive modeling.

## 2. History
Kearns & Valiant (1989) asked whether weak learners (barely better than chance) can be boosted into strong ones → Schapire (1990) proved yes → **AdaBoost** (Freund & Schapire 1995/1997; Gödel Prize 2003) → statistical view: "additive logistic regression" (Friedman, Hastie, Tibshirani 2000) showed AdaBoost minimizes exponential loss by stage-wise fitting → **Gradient Boosting Machine** (Friedman 1999/2001, "Greedy function approximation") generalized to any differentiable loss; Stochastic GB (Friedman 2002) added subsampling. Modern implementations: Ch 3.4.

## 3. Intuition
- **AdaBoost:** train a small tree; raise the weight of the points it got wrong; train the next tree on the reweighted data; combine with votes weighted by each tree's accuracy.
- **Gradient boosting:** fit a tree to the current *residuals* (errors), add a fraction of it to the model, repeat — gradient descent in function space.
```
 F0 = mean(y)  →  F1 = F0 + η·tree1(residuals)  →  F2 = F1 + η·tree2(residuals of F1)  → ...
```

## 4. Math
**AdaBoost (binary, labels ±1):** weights `w_i=1/n`. For each round: fit `h_t` on weighted data; error `ε_t = Σ w_i 1[h_t(x_i)≠y_i]`; `α_t = ½ ln((1−ε_t)/ε_t)`; update `w_i ← w_i · exp(−α_t y_i h_t(x_i))`, renormalize; final `H(x)=sign(Σα_t h_t(x))`.
**Worked example.** 5 points, equal weights 0.2; stump misclassifies 1 ⇒ ε=0.2, `α = ½ln(0.8/0.2) = ½ln4 = 0.693`. Misclassified weight ×e^{0.693}=2 ⇒ 0.4; each correct ×e^{−0.693}=0.5 ⇒ 0.1 (four of them = 0.4). Normalize (sum 0.8): mistake **0.5**, others **0.125** each — the one hard point now carries half the weight.
**Gradient boosting:** minimize `Σ L(y_i, F(x_i))`. Stage m: pseudo-residuals `r_im = −∂L/∂F(x_i)`; fit tree `h_m` to `r_im`; (optionally line-search leaf values); `F_m = F_{m−1} + η h_m`.
- Squared error: `r = y − F` (plain residuals). Log-loss: `r = y − p`. Absolute/Huber/quantile: sign-based, robust, quantile.
**Worked example (MSE).** y=[3,5,10], F₀=mean=6. Residuals [−3,−1,4]. Suppose a stump predicts [−2,−2,4] (splits {first two} vs {third}); with η=0.5: F₁ = 6+0.5·[−2,−2,4] = [5,5,8]; new residuals [−2,0,2]; squared error fell from 9+1+16=26 to 4+0+4=**8**. Each stage chips away at the error.
Assumptions: none distributional; needs enough weak-learner capacity; sensitive to label noise (AdaBoost's exponential loss especially).

## 5. Training & prediction
Sequential (can't parallelize across trees; parallelize inside a tree). Cost `O(M · tree cost)`, prediction `O(M·depth)`. Regularization levers: learning rate η (shrinkage), number of trees M, tree depth, subsampling, early stopping on a validation set.

## 6. Evolution
| Step | Improvement | Cost |
|---|---|---|
| Bagging → AdaBoost | Reduces bias via reweighting | Sensitive to noisy labels/outliers |
| AdaBoost → Gradient boosting | Any loss (regression, ranking, quantile, count) | More hyperparameters |
| GBM → Stochastic GB | Row subsampling: faster, regularizes | One more knob |
| GBM → XGBoost/LightGBM/CatBoost/HistGB | Regularization, speed, histograms, categoricals (Ch 3.4) | Library-specific tuning |

## 7. When to use / not
Use: tabular prediction where accuracy matters, nonlinear/interactions, mixed data, custom losses. Avoid: very noisy labels with AdaBoost, tiny datasets (overfit), need for extrapolation, strict training-time limits without histogram implementations, when a logistic baseline is within noise.

## 8. Strengths / weaknesses
+ Top accuracy on tabular data, flexible loss, little preprocessing. − Many hyperparameters, sequential training, overfit with large M and η, less interpretable, probabilities may need calibration.

## 9. Hyperparameters
`learning_rate` η (↓ needs ↑ trees; 0.01–0.1 typical), `n_estimators` M (pick via early stopping), `max_depth` (3–8; depth = interaction order), `subsample` (0.5–1), `max_features`, `min_samples_leaf`, `loss`, AdaBoost `estimator` (stump) and `learning_rate`.

## 10. Implementation
From scratch (least-squares GBM; checked against sklearn):
```python
class GBM:
    def fit(self, X, y, n=200, lr=0.1, depth=3):
        self.f0 = y.mean(); F = np.full(len(y), self.f0); self.trees = []
        for _ in range(n):
            t = DecisionTreeRegressor(max_depth=depth).fit(X, y - F)   # residuals = negative gradient of MSE
            F += lr * t.predict(X); self.trees.append(t)
        return self
    def predict(self, X): return self.f0 + lr * sum(t.predict(X) for t in self.trees)
```
Friedman-1 synthetic data (n=1000): scratch test RMSE **1.490** vs sklearn `GradientBoostingRegressor` **1.497** (single depth-3 tree: 3.39).
Library: `GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.1, subsample=0.8)`, `AdaBoostClassifier`.

## 11. Evaluation & results (measured)
California housing test RMSE: AdaBoost(100, default depth-3 base) **0.950** (worse than even a single tree 0.709 — AdaBoost.R2 with default settings is a weak regressor here; use gradient boosting for regression), sklearn GBM(300, depth 3, η 0.1) **0.499**. A deeper run (1500 trees, η 0.1) kept improving up to the last tree: RMSE 0.468 — no overfitting turning point within 1500 trees at this depth/η, so "more trees always overfit" isn't a rule; always check the validation curve.
Breast-cancer 5-fold AUC: AdaBoost 0.993, GBM 0.991, RF 0.991 — all statistically indistinguishable on this small easy set.
**Always use early stopping** (`n_iter_no_change`, `validation_fraction`) rather than a fixed M.

## 12. Case study
Churn/credit scoring: logistic baseline → GBM with `learning_rate=0.05`, depth 4, subsample 0.8, early stopping; compare AUC/PR-AUC and calibration; explain with SHAP.

## 13. Comparison
vs RF: boosting is usually more accurate but more sensitive to tuning/noise; RF is parallel and safer. vs deep nets: GBMs remain the default on tabular data of moderate size.

## Interview Q&A
- *Bagging vs boosting?* Bagging: parallel, reduces variance. Boosting: sequential, reduces bias (and some variance).
- *Why is the residual the "gradient"?* For squared loss, −∂/∂F ½(y−F)² = y−F.
- *What does the learning rate do?* Shrinks each step; smaller η + more trees generalizes better but costs compute.
- *Why is AdaBoost noise-sensitive?* Exponential loss keeps upweighting mislabeled points.

## Exercises
Implement AdaBoost with stumps by hand; add log-loss gradients to the scratch GBM for classification; draw train/validation curves vs M for different η.

## Further reading
Freund & Schapire (1997) JCSS; Friedman, Hastie, Tibshirani (2000) *Annals of Statistics*; Friedman (2001) "Greedy function approximation: a gradient boosting machine"; ESL Ch 10.
