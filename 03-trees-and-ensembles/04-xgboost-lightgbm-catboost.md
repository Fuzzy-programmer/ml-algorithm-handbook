# Chapter 3.4 — XGBoost, LightGBM, CatBoost and HistGradientBoosting

[← Roadmap](../README.md) · Prev [3.3](03-adaboost-gradient-boosting.md) · Next [3.5 GMM/HMM/Bayes nets](05-gmm-hmm-bayesnet.md) · Priority 🔴 · Code: [`trees_family.py`](../code/trees_family.py)

## 1. Identity & problem
Engineering- and algorithm-level upgrades to gradient boosting that made it fast, regularized and practical at scale. Same core idea as Ch 3.3; different engineering trade-offs.

## 2. History
- **XGBoost** — Chen & Guestrin, KDD 2016 (open-sourced 2014; dominated Kaggle tabular competitions).
- **LightGBM** — Ke et al., NeurIPS 2017 (Microsoft): histogram splits, leaf-wise growth, GOSS, EFB.
- **CatBoost** — Prokhorenkova et al., NeurIPS 2018 (Yandex): ordered boosting, ordered target statistics, symmetric trees.
- **HistGradientBoosting** — scikit-learn (0.21+; inspired by LightGBM): native histograms, NaN support, categorical support.
Not a strict ranking — they are siblings with different strengths.

## 3. What each contributes
| Library | Key ideas | Strength | Watch out |
|---|---|---|---|
| **XGBoost** | 2nd-order (Newton) objective, explicit leaf regularization (γ, λ), sparsity-aware splits (native missing values), column/row subsampling, approximate & exact split finders, out-of-core, GPU | Robust, flexible objectives, mature | More knobs; level-wise default is slower than leaf-wise |
| **LightGBM** | Histogram binning, **leaf-wise** (best-first) growth, GOSS (keep large-gradient samples), EFB (bundle exclusive sparse features), native categoricals | Very fast on large/high-dim data | Leaf-wise overfits small data; tune `num_leaves`, `min_child_samples` |
| **CatBoost** | **Ordered boosting** (reduces prediction-shift bias), ordered target statistics for categoricals (reduces target leakage), symmetric (oblivious) trees | Best out-of-the-box with high-cardinality categoricals; strong defaults | Slower on pure numeric data; fewer deployment targets |
| **HistGB (sklearn)** | Histograms, NaN-aware, early stopping built in | Zero extra dependencies | Fewer features/objectives |

## 4. Math (XGBoost)
Objective at stage t: `L = Σ l(y_i, ŷ_i^{(t−1)} + f_t(x_i)) + Ω(f_t)`, `Ω = γT + ½λΣ w_j²`.
Second-order Taylor: with `g_i=∂l/∂ŷ`, `h_i=∂²l/∂ŷ²`: for a leaf with sums `G=Σg`, `H=Σh`:
optimal leaf weight `w* = −G/(H+λ)`; leaf score `−½ G²/(H+λ) + γ`.
Split **gain** = `½[ G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ) ] − γ`. Split only if gain > 0 ⇒ γ is a built-in pruning threshold.
**Worked example (squared error ⇒ g = ŷ − y, h = 1).** Four rows with g = [−2,−2,+2,+2]. Parent: G=0 ⇒ term 0. Split {−2,−2} | {+2,+2}: G_L=−4, H_L=2; G_R=4, H_R=2; λ=1:
`G_L²/(H_L+λ)=16/3=5.333` (same for right) ⇒ gain `= ½(5.333+5.333−0) = 5.333` (γ=0). Leaf weights `w_L = 4/3 = 1.333`, `w_R = −1.333` (pull predictions up for the left rows whose errors were negative). With λ=0 the weights would be ±2 (full correction); λ shrinks them — regularization in action.
LightGBM leaf-wise: always split the leaf with largest gain, regardless of depth ⇒ lower loss per split but deeper, lopsided trees. Histogram: bin each feature into ≤255 buckets; split search `O(#bins)` rather than `O(n)`; subtraction trick (sibling histogram = parent − child).

## 5. Training & cost
Per tree: histogram building `O(n·p)` once per level/leaf, split search `O(bins·p)`. Orders of magnitude faster than classic GBM on large n. GPU variants exist (XGBoost `device="cuda"`, LightGBM/CatBoost GPU). Prediction cost `O(M·depth)`; models are compact vs forests.

## 6. When to use / not
Use: any structured/tabular supervised problem (the default winner), ranking (LambdaRank/`rank:ndcg`), counts/quantiles/survival objectives, missing data, large n. Avoid: images/audio/text without embeddings, tiny data where simpler models match, when extrapolation is needed, strict interpretability mandates (use GAM/EBM or constrained monotonic GBMs).

## 7. Hyperparameters (cheat sheet)
| Concept | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| Learning rate | `learning_rate` | `learning_rate` | `learning_rate` |
| Trees | `n_estimators` + `early_stopping_rounds` | `n_estimators` + callbacks | `iterations` + `early_stopping_rounds` |
| Complexity | `max_depth`, `min_child_weight`, `gamma` | `num_leaves` (< 2^depth), `max_depth`, `min_child_samples` | `depth` |
| Regularization | `reg_lambda`, `reg_alpha` | `lambda_l2/l1`, `min_gain_to_split` | `l2_leaf_reg` |
| Subsampling | `subsample`, `colsample_bytree` | `bagging_fraction/freq`, `feature_fraction` | `subsample`, `rsm` |
| Categorical | one-hot / `enable_categorical` | `categorical_feature` | `cat_features` (native) |
| Imbalance | `scale_pos_weight` | `scale_pos_weight`/`is_unbalance` | `auto_class_weights` |
**Tuning order that works:** fix η≈0.05–0.1 and use early stopping → tune tree size (`max_depth`/`num_leaves`, `min_child_*`) → subsampling → regularization → lower η and add trees at the end. Use random search/Optuna rather than large grids.

## 8. Implementation
```python
import xgboost as xgb, lightgbm as lgb, catboost as cb
from sklearn.ensemble import HistGradientBoostingRegressor
xg = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8)
lg = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
cbm = cb.CatBoostRegressor(iterations=500, learning_rate=0.08, depth=6, verbose=0)
hg = HistGradientBoostingRegressor()                 # early_stopping='auto' for n>10k
# early stopping:  xg.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)  (set early_stopping_rounds in the constructor)
```

## 9. Evaluation & results (measured; California housing, test RMSE in $100k; 4-core laptop)
| Model | Train | Test | Fit time |
|---|---|---|---|
| Linear regression | 0.720 | 0.746 | 0.0 s |
| Random forest (200) | 0.186 | 0.502 | 19.2 s |
| sklearn GBM (300, d3) | 0.444 | 0.499 | 15.9 s |
| HistGradientBoosting | 0.400 | 0.462 | **0.7 s** |
| CatBoost (500, d6) | 0.380 | 0.459 | 3.5 s |
| XGBoost (500, η .05, d6) | 0.278 | 0.445 | 3.6 s |
| **LightGBM (500, 31 leaves)** | 0.323 | **0.437** | **1.0 s** |
All parameters were set once, **not tuned**; differences of ~0.01–0.02 among the top four are within what tuning/seed changes can move, so treat them as a tie. The robust conclusions: (1) all modern boosters beat the forest (0.502) and linear model (0.746); (2) histogram-based ones train 5–20× faster than classic GBM/RF.
**Leaf-wise overfitting (LightGBM, only 300 training rows, test RMSE):** `num_leaves` 4 → 0.649; 15 → 0.658; **63 → 0.692** — capacity beyond what the data supports hurts.
**Missing values:** HistGB trained and tested with 20% of all entries randomly set to NaN: RMSE 0.589 (vs 0.462 with complete data) — works with no imputation, at an expected accuracy cost.
Breast cancer 5-fold AUC: XGB 0.994, LGBM 0.993, RF 0.991 — saturated; the ranking is noise on this small easy set.

## 10. Case study
Fraud/churn with categorical IDs: CatBoost with `cat_features` vs LightGBM with ordinal-encoded categories vs target-encoded logistic. Evaluate on a **time-based split**, report PR-AUC, calibrate, tune the threshold, and explain with SHAP. Check train/serve skew.

## 11. Comparison & decision guide
Default start: LightGBM or HistGB for speed; XGBoost for maturity, custom objectives, GPU, distributed; CatBoost for many categorical features or robust defaults. Always compare to a regularized linear model and a random forest; and confirm gains hold on time-ordered validation.

## 12. Pitfalls
Tuning on the test set · early stopping on the same validation set used to report · leaking via target statistics · ignoring monotonic constraints when business requires them (`monotone_constraints`) · treating feature importance (gain/split count) as causal · different library defaults making "default vs default" comparisons unfair.

## Interview Q&A
- *What did XGBoost add over GBM?* 2nd-order optimization, explicit regularization, sparsity-aware splits, systems engineering (parallel/histogram/out-of-core).
- *Level-wise vs leaf-wise?* Level-wise grows balanced trees; leaf-wise chooses the highest-gain leaf, faster loss reduction but overfits small data.
- *Why does CatBoost have "ordered" boosting?* Standard target statistics and gradients use the row's own label, causing prediction shift/leakage; ordering avoids it.
- *Which hyperparameters first?* learning rate + early stopping, then depth/leaves and min child weight.

## Exercises
Reproduce the table with random search per model; add monotone constraints and measure the cost; compare gain, split-count and SHAP importance on the same model.

## Further reading
Chen & Guestrin (2016) "XGBoost: A scalable tree boosting system"; Ke et al. (2017) "LightGBM"; Prokhorenkova et al. (2018) "CatBoost: unbiased boosting with categorical features"; Grinsztajn, Oyallon & Varoquaux (2022) "Why do tree-based models still outperform deep learning on typical tabular data?"; official XGBoost/LightGBM/CatBoost docs.
