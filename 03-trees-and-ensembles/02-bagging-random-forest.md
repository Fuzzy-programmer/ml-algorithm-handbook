# Chapter 3.2 — Bagging, Random Forest, Extra Trees

[← Roadmap](../README.md) · Prev [3.1](01-decision-trees.md) · Next [3.3 AdaBoost & Gradient Boosting](03-adaboost-gradient-boosting.md) · Priority 🔴 · Code: [`trees_family.py`](../code/trees_family.py)

## 1. Identity & problem
Variance-reduction ensembles of decorrelated trees. A single deep tree has low bias but high variance; averaging many different trees keeps the low bias and cuts the variance.

## 2. History
**Bagging** (bootstrap aggregating) — Breiman 1996, built on Efron's bootstrap (1979). Random subspace (Ho 1995) and random split selection (Amit & Geman 1997) → **Random Forest** — Breiman 2001 (bagging + random feature subset at each split; OOB error; importance). **Extremely Randomized Trees** — Geurts, Ernst & Wehenkel 2006 (random thresholds, usually no bootstrap).

## 3. Intuition
Ask 200 slightly different experts (each saw a random sample and a random subset of clues) and average their answers. Individual mistakes cancel if the experts aren't all making the same mistake — hence the random feature subsets (to decorrelate).

## 4. Math
Average of B predictors with variance σ², pairwise correlation ρ: `Var = ρσ² + (1−ρ)σ²/B`. More trees kill the second term; only lower ρ reduces the first ⇒ random feature subsampling.
Bootstrap sample of size n leaves out each row with probability `(1−1/n)ⁿ → 1/e ≈ 36.8%` ⇒ **out-of-bag (OOB)** rows give a free validation estimate.
**Worked example.** Five trees predict 4.0, 6.0, 5.0, 7.0, 3.0 for one house. Forest = mean **5.0**. If each tree's error variance is 1 and trees are independent, ensemble variance = 1/5 = 0.2; with ρ=0.5: 0.5 + 0.5/5 = **0.6** — correlation limits the gain.
Assumptions: none about distributions. Bagging reduces variance, **not** bias.

## 5. Training & prediction
Trees are trained independently ⇒ trivially parallel. Cost `O(B · p'·n log n)` (p' = features tried per split); prediction `O(B·depth)`; memory `O(B·nodes)` — can be GBs for deep forests on large n (I hit memory limits on a 5.8 GB laptop with 100 deep trees and `n_jobs=-1`; use `min_samples_leaf`, fewer trees, or fewer jobs). Defaults: regression `max_features=1.0` in current sklearn, classification `sqrt(p)`.

## 6. Evolution
| Step | Improvement | Cost |
|---|---|---|
| Tree → Bagging | Large variance reduction | Less interpretable, memory |
| Bagging → Random Forest | Decorrelated trees, OOB, importances | Another hyperparameter (`max_features`) |
| RF → Extra Trees | Even lower variance, faster training | Slightly more bias; often competitive |
| RF → Boosting | Reduces bias too, often more accurate | Sequential, more tuning, can overfit |
| RF variants | Quantile RF (prediction intervals), Isolation Forest (Ch 4.5), Random survival forests, causal forests | Specialized |

## 7. When to use / not
Use: strong default for tabular data with little tuning, noisy data, mixed features, OOB estimates, importance screening, robust baseline. Avoid: need for extrapolation (flat), very high-dimensional sparse text (linear models win), tight memory/latency, when boosting gives clearly better accuracy and you can tune it.

## 8. Strengths / weaknesses
+ Robust, few hyperparameters, parallel, handles nonlinearity/interactions, OOB, hard to badly overfit by adding trees. − Big models, slow prediction for huge forests, piecewise-constant, biased impurity importance, probabilities not calibrated, usually behind tuned boosting.

## 9. Hyperparameters
`n_estimators` (more is never worse in expectation, diminishing returns), `max_features` (↓ = more decorrelation, more bias), `max_depth`/`min_samples_leaf` (memory + smoothness), `bootstrap`, `max_samples`, `oob_score`, `class_weight`, `n_jobs`, `random_state`.

## 10. Implementation
```python
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
rf = RandomForestRegressor(n_estimators=200, max_features=0.6, min_samples_leaf=1, oob_score=True, n_jobs=2, random_state=0).fit(Xtr, ytr)
print(rf.oob_score_)
# from scratch idea: for b in range(B): idx = rng.integers(0,n,n); tree_b = Tree(max_features=...).fit(X[idx], y[idx]); predict = mean over b
```

## 11. Evaluation & results (measured; California housing, test RMSE, lower is better)
| Model | Train | Test |
|---|---|---|
| Linear regression | 0.720 | 0.746 |
| Single deep tree | 0.000 | 0.709 |
| Bagging (50 deep trees) | 0.193 | 0.507 |
| **Random forest (200)** | 0.186 | **0.502** |
| Extra Trees (200) | 0.000 | **0.501** |
Forest `max_features` sweep (test RMSE / OOB R²): 0.33 → 0.495 / 0.819; **0.6 → 0.492** / 0.819; 1.0 → 0.502 / 0.810. A single tree → forest cuts RMSE 29% (0.709 → 0.502). Bagging alone already captures most of the gain; feature subsampling adds a little. OOB R² (0.819) lands in the same ballpark as test performance without any holdout.
**Importance (RF):** impurity ranks MedInc 0.525, AveOccup 0.138, Latitude 0.089, Longitude 0.089, while *permutation* importance gives MedInc 0.733, Latitude 0.456, Longitude 0.341, AveOccup 0.204 — location matters much more than impurity suggests (impurity importance is biased toward features with many split points like AveOccup). Prefer permutation/SHAP.

## 12. Case study
California housing: RF captures the nonlinear latitude/longitude geography that linear regression (Ch 1.1) could not, cutting test RMSE from 0.746 to 0.502 (≈33%). Next chapters show boosting pushes to ~0.44.

## 13. Comparison
vs single tree: accuracy ↑, interpretability ↓. vs boosting: RF is easier and more robust to defaults; boosting is usually more accurate when tuned. vs Extra Trees: ET faster and sometimes smoother; same family.

## Interview Q&A
- *Why random feature subsets?* Decorrelate trees; averaging correlated trees helps less.
- *What is OOB?* Rows not in a tree's bootstrap sample (≈36.8%) give an unbiased-ish validation score.
- *Can a forest overfit by adding trees?* No (it converges), but deep trees can still overfit noise.
- *Why can't RF extrapolate?* Leaves hold averages of training targets.

## Exercises
Plot OOB error vs `n_estimators`; implement bagging by hand and verify variance reduction vs one tree; compare impurity vs permutation importance with a random noise feature added.

## Further reading
Breiman (1996) "Bagging predictors"; Breiman (2001) "Random forests," *Machine Learning*; Geurts et al. (2006); ESL Ch 15; sklearn "Ensemble methods".
