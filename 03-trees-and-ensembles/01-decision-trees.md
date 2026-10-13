# Chapter 3.1 — Decision Trees (ID3 → C4.5 → CART) and Pruning

[← Roadmap](../README.md) · Prev [2.6 SVM](../02-classification/06-svm.md) · Next [3.2 Bagging & Random Forest](02-bagging-random-forest.md) · Priority 🔴 · Code: [`trees_scratch.py`](../code/trees_scratch.py)

## 1. Identity & problem
Recursive partitioning model: a flowchart of if/else questions on features, ending in leaves that predict a class or number. "Approve this loan?" — yes/no questions a human could follow.

## 2. History (a family, not a line)
- **AID** (Morgan & Sonquist 1963) and **CHAID** (Kass 1980): statistical-test-based splitting for survey data.
- **ID3** (Quinlan 1986; earlier CLS 1966/1979): categorical features, **information gain**, no pruning, no missing values.
- **C4.5** (Quinlan 1993): **gain ratio** (corrects ID3's bias toward many-valued features), continuous thresholds, missing-value handling, error-based pruning. C5.0 is the proprietary successor.
- **CART** (Breiman, Friedman, Olshen & Stone 1984): binary splits, **Gini** (classification) / variance (regression), **cost-complexity pruning**, surrogate splits. scikit-learn implements an optimized CART.
- Later: conditional inference trees (Hothorn et al. 2006; unbiased split selection), oblique trees. The real step-change came by *combining* trees (Ch 3.2–3.4).

## 3. Intuition
Ask the question that best separates the classes; repeat on each side. Stop when groups are pure or small.
```
              Income > 50k?
              /          \
           yes            no
    Age > 30?          Default risk: HIGH (leaf)
     /     \
  LOW      MEDIUM
```

## 4. Math
Impurity of a node with class proportions `p_k`: **Gini** `G = 1 − Σp_k²`; **entropy** `H = −Σp_k log₂p_k`; regression: variance/MSE.
Split quality = parent impurity − weighted child impurity (information gain for entropy). Gain ratio = IG / SplitInfo (C4.5).
CART cost-complexity pruning: minimize `R(T) + α|T|` (error plus α per leaf); α chosen by CV (`ccp_alpha`).
**Worked example.** Labels `[0,0,0,1,1,1,1,1]`: Gini = 1−(3/8)²−(5/8)² = **0.4688**; entropy = **0.9544** bits. Candidate split → left `[0,0,0,1]`, right `[1,1,1,1]`:
Left Gini = 1−0.75²−0.25² = 0.375; right = 0 ⇒ weighted = (4·0.375+4·0)/8 = **0.1875**; Gini gain = 0.4688−0.1875 = **0.2813**. Entropy: left H=0.8113 ⇒ weighted 0.4056; information gain = 0.9544−0.4056 = **0.5488**. The algorithm evaluates every feature/threshold this way and takes the best (greedy, no look-ahead).
Assumptions: none about distributions or linearity; axis-aligned splits; features can be mixed scale; invariant to monotonic transformations.

## 5. Training & prediction
Greedy top-down; stop at `max_depth`, `min_samples_leaf/split`, no gain, or purity. Cost: sorting makes it `O(p·n log n)` per level; prediction `O(depth)`, tiny memory. Optimal tree is NP-complete, so greedy heuristics are used. Regression leaves predict the mean; classification leaves predict class frequencies (often poorly calibrated).

## 6. Evolution & trade-offs
| Step | Improvement | Cost |
|---|---|---|
| ID3 → C4.5 | Numeric features, missing values, pruning, gain ratio | More complexity |
| C4.5 → CART | Regression, binary splits, principled pruning, surrogate splits | Gini bias toward multi-valued/high-cardinality features |
| Single tree → Random Forest / boosting | Removes the high variance of one tree | Interpretability, memory |
| Tree → Optimal/oblique/soft trees | Better splits / smoother | Compute, complexity |

## 7. When to use / not
Use: interpretable rules, mixed feature types with no preprocessing, nonlinearity/interactions, quick baselines, as building blocks. Avoid as final model on noisy data (unstable), smooth/additive relations (staircase approximation), extrapolation (flat outside range), high-cardinality categoricals (bias), when stability matters.

## 8. Strengths / weaknesses
+ Interpretable, no scaling, handles interactions, fast prediction, minimal preprocessing. − High variance (small data change ⇒ different tree), overfits when deep, piecewise-constant output, greedy, biased toward high-cardinality features, cannot extrapolate.

## 9. Hyperparameters
`max_depth` (↑ complexity), `min_samples_leaf` (↑ smoother, regularizes), `min_samples_split`, `max_leaf_nodes`, `min_impurity_decrease`, `ccp_alpha` (pruning), `criterion` (gini/entropy/log_loss; MSE/MAE for regression), `max_features`, `class_weight`, `splitter`.

## 10. Implementation
From scratch CART (Gini) in [`trees_scratch.py`](../code/trees_scratch.py):
```python
def gini(y): p = np.bincount(y)/len(y); return 1 - (p**2).sum()
# for each feature j, each midpoint threshold t: score = (nL*gini(yL) + nR*gini(yR))/n ; keep the minimum, then recurse
```
Library:
```python
from sklearn.tree import DecisionTreeClassifier, export_text
tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, ccp_alpha=0.001, random_state=0).fit(X, y)
print(export_text(tree, feature_names=list(names)))
```

## 11. Evaluation & results (measured)
Breast cancer test split (stratified): scratch vs sklearn accuracy — depth 1: 0.888 / 0.888; depth 3: 0.916 / 0.916; depth 5: 0.909 / 0.902 (tiny difference from tie-breaking in equal-gain splits).
California housing: unpruned tree **train RMSE 0.000, test 0.709**; depth 6: train 0.656, test **0.705** — shallow beats memorizing, but both lose badly to ensembles (0.44–0.50). The train/test gap of the unpruned tree is the signature of variance.

## 12. Case study
Credit-risk rules: fit a depth-3/4 tree with `min_samples_leaf≥50`, export rules for the compliance team, then compare its AUC with a boosted model to quantify the explainability cost. Check stability by refitting on bootstrap samples and noting how top splits change.

## 13. Comparison
vs logistic/linear: trees capture interactions/nonlinearity automatically; linear models extrapolate and are smoother. vs ensembles: single trees explain; ensembles predict.

## Interview Q&A
- *Gini vs entropy?* Very similar in practice; entropy slightly more sensitive to balanced splits, Gini cheaper (no log).
- *Why do trees overfit?* Can isolate every training point; fix with depth limits, min leaf size, pruning, ensembling.
- *Do trees need scaling?* No; thresholds only use order.
- *Feature importance caveat?* Impurity importance favors high-cardinality/continuous features; use permutation importance.

## Exercises
Hand-compute the best split for a 6-row dataset; implement regression-tree splitting by variance reduction; run cost-complexity pruning with `cost_complexity_pruning_path` and CV `ccp_alpha`.

## Further reading
Breiman et al. (1984) *Classification and Regression Trees*; Quinlan (1986) "Induction of decision trees," (1993) *C4.5*; Hothorn, Hornik, Zeileis (2006); ESL Ch 9.2; sklearn "Decision Trees" guide.
