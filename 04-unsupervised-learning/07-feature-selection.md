# Chapter 4.7 — Feature Selection: Filter, Wrapper, Embedded

[← Roadmap](../README.md) · Prev [4.6](06-association-rules.md) · Next [5.1 MLP & backprop](../05-deep-learning/01-mlp-backprop.md) · Priority 🟠 · Code: [`unsupervised_family.py`](../code/unsupervised_family.py)

## 1. Problem
Choose a subset of features to improve generalization, speed, interpretability and cost-to-collect. Distinct from dimensionality *reduction* (PCA creates new features; selection keeps original ones).

## 2. Taxonomy (Guyon & Elisseeff 2003)
| Family | Idea | Examples | Pros | Cons |
|---|---|---|---|---|
| **Filter** | Score features independent of the model | variance threshold, correlation, ANOVA F, χ², mutual information, ReliefF | Fast, model-agnostic | Ignores interactions & redundancy |
| **Wrapper** | Search subsets using model performance | forward/backward selection, **RFE** (Guyon et al. 2002, SVM-RFE), genetic search | Model-aware, captures interactions | Expensive, overfits selection |
| **Embedded** | Selection happens during training | **L1/Lasso**, ElasticNet, tree importances, regularized boosting, group lasso | Efficient, principled | Model-specific |
| Other | Boruta (Kursa & Rudnicki 2010; compares to shadow features), permutation importance, SHAP-based pruning, stability selection (Meinshausen & Bühlmann 2010) | Robustness | Compute |

## 3. Math notes
Mutual information `I(X;Y) = Σ p(x,y) log[p(x,y)/(p(x)p(y))]` captures nonlinear dependence (estimated by k-NN estimators). ANOVA F: between-group over within-group variance. **Redundancy:** two highly correlated informative features both score high individually (mRMR trades relevance against redundancy). Lasso embedded selection: coefficients driven to exactly 0 (Ch 1.3).
**Worked example.** Feature X has correlation 0.5 with y, feature Z has 0.5 and `corr(X,Z)=0.9`. A filter ranks both top; a wrapper/embedded method keeps one because the second adds little given the first.

## 4. The cardinal rule
Selection is part of the model. **Do it inside cross-validation** (Pipeline). Selecting on the full data first leaks label information — Ch 0.5 showed noise features producing 0.86 "accuracy" on random labels.

## 5. When to use / not
Use: p ≫ n, collection costs, regulation/interpretability, noisy irrelevant features, linear/SVM/kNN models (sensitive to junk). Often unnecessary for boosted trees/forests with moderate p (they ignore junk), but can reduce latency and cost. Avoid: selecting by univariate p-values then claiming inference; keeping selection stability unexamined.

## 6. Hyperparameters
`k` / threshold (choose by CV, e.g. `RFECV`), scoring function, estimator for RFE, step size, L1 strength `C`/`alpha`, stability-selection subsample count.

## 7. Implementation
```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE, RFECV, SelectFromModel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
filt = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=10), LogisticRegression(max_iter=5000))
wrap = make_pipeline(StandardScaler(), RFE(LogisticRegression(max_iter=5000), n_features_to_select=10), LogisticRegression(max_iter=5000))
emb  = make_pipeline(StandardScaler(), LogisticRegression(penalty="l1", solver="liblinear", C=0.1))
```

## 8. Evaluation & results (measured)
Breast cancer (30 real features) + 100 pure-noise features (130 total); 5-fold CV accuracy, selection inside the pipeline:
| Approach | Accuracy |
|---|---|
| All 130 features | 0.947 |
| Filter: ANOVA top-10 | 0.951 |
| Filter: mutual information top-10 | 0.951 |
| Wrapper: RFE → 10 | 0.968 |
| **Embedded: L1 logistic (C=0.1)** | **0.972** |
All three families beat "keep everything" (0.947), with wrapper/embedded ahead of simple filters. The ANOVA filter's top-10 contained **0** noise features (so its gap to the others reflects the *redundancy/interaction* handling, not noise leakage). For reference, the clean 30-feature logistic model scores 0.979 (Ch 2.1): the 100 noise columns cost ~3 points that selection recovers most of. With more noise columns or fewer rows the gap widens.

## 9. Pitfalls
Selection outside CV · unstable selections (report frequency across folds/bootstraps) · treating importance as causation · dropping features needed for fairness audits · correlated groups (Lasso picks arbitrarily; use ElasticNet/group methods) · impurity importance bias (use permutation).

## Interview Q&A
- *Filter vs wrapper vs embedded?* Score independently vs search with the model vs selection inside training.
- *Why not select features on the entire dataset first?* Information leakage inflates validation scores.
- *Do trees need feature selection?* Less so; they down-weight irrelevant features, but selection can cut cost and variance.

## Exercises
Add more noise columns and plot accuracy vs number of noise features for each family; implement forward selection with CV; run stability selection with Lasso.

## Further reading
Guyon & Elisseeff (2003) *JMLR*; Guyon et al. (2002) SVM-RFE; Kursa & Rudnicki (2010) Boruta; Meinshausen & Bühlmann (2010); Kuhn & Johnson, *Feature Engineering and Selection*.
