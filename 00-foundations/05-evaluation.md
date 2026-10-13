# Chapter 0.5 — Evaluation: Metrics, CV, Leakage, Calibration, Thresholds

[← Roadmap](../README.md) · Prev [0.4](04-bias-variance-regularization.md) · Next [0.6](06-feature-engineering.md) · Priority 🔴

## 1. Principle
A model is only as trustworthy as its evaluation. Decide **metric, split strategy and baseline before** training.

## 2. Splits
| Scheme | Use | Notes |
|---|---|---|
| Train/val/test | Large data | Test touched once |
| K-fold CV | Small/medium data | Use stratified for classification |
| GroupKFold | Rows share an entity (patient, user) | Prevents identity leakage |
| TimeSeriesSplit / rolling origin | Temporal data | Train strictly before validate |
| Nested CV | Unbiased estimate when also tuning | Costly |

## 3. Metrics
**Regression:** MAE, RMSE, R², MAPE (unsafe near 0), quantile/pinball loss.

**Classification** (confusion matrix TP, FP, FN, TN):
`precision=TP/(TP+FP)`, `recall=TP/(TP+FN)`, `F1=2PR/(P+R)`, `accuracy=(TP+TN)/N`, `specificity=TN/(TN+FP)`.
- **ROC-AUC:** probability a random positive scores above a random negative; insensitive to class balance (can look rosy when positives are rare).
- **PR-AUC / average precision:** focuses on the positive class; preferred for rare positives.
- **Log loss / Brier score:** judge probability quality.
- Multiclass: macro (equal per class) vs micro (equal per sample) vs weighted averages.

**Ranking/retrieval:** precision@k, recall@k, MRR, NDCG. **Clustering:** silhouette, ARI (with labels). **Generation:** task-specific evals (Ch 6.11).

**Worked example.** TP=40, FP=10, FN=60, TN=890 (N=1000). Accuracy 0.93, precision 0.80, recall 0.40, F1 0.533. A model predicting "negative" always scores 0.90 accuracy — accuracy hides the failure.

## 4. Imbalance — measured (2% positives, random forest)
Always-negative accuracy **0.975**; RF accuracy 0.986, ROC-AUC 0.884, PR-AUC 0.725.
Moving the threshold trades precision for recall:
| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.5 | 1.000 | 0.440 | 0.611 |
| 0.3 | 0.911 | 0.576 | 0.706 |
| 0.1 | 0.461 | 0.760 | 0.574 |
Choose the threshold from business costs (e.g. cost of a missed fraud vs a false alert), on validation data — not 0.5 by default. Class weights/resampling change the scores; thresholding is often enough.

## 5. Data leakage (the #1 silent killer)
Information from validation/test (or the future, or the target) leaks into training.
**Measured:** on pure-noise data (true accuracy 0.5), selecting the 20 "best" features on *all* data and then cross-validating reports **0.86**; doing the selection inside a `Pipeline` gives **0.50**.
Sources: preprocessing fit on all data, target encoding without folds, duplicates across splits, time leakage, features recorded after the outcome, group leakage. Cure: `Pipeline`, correct splitter, adversarial review of every feature ("would I know this at prediction time?").

## 6. Calibration
A calibrated model's "0.8" means 80% of such cases are positive. Check with reliability diagrams and Brier score; fix with Platt (sigmoid) or isotonic regression (`CalibratedClassifierCV`). Random forests and boosted trees are often miscalibrated; logistic regression usually isn't. Measured Brier on the imbalanced demo: raw RF 0.0119 → isotonic 0.0112 (a small gain; bigger when scores are far off).

## 7. Reporting checklist
Baseline (mean/majority/last-value/seasonal-naive) · metric + CI (bootstrap or CV spread) · same split for all candidates · error analysis by segment · calibration · latency/memory · distribution-shift check.

## 8. Code
[`code/foundations_demos.py`](../code/foundations_demos.py). Template:
```python
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
cv = StratifiedKFold(5, shuffle=True, random_state=0)
res = cross_validate(pipe, X, y, cv=cv, scoring=["roc_auc", "average_precision", "f1"])
```

## Interview Q&A
- *ROC-AUC vs PR-AUC?* With rare positives PR-AUC reveals false-positive cost that ROC-AUC dilutes.
- *95% accuracy, is it good?* Compare to majority baseline and look at per-class metrics.
- *How do you detect leakage?* Suspiciously high scores, one dominant feature, drop at deployment.

## Exercises
Compute precision/recall by hand from a confusion matrix; build a `TimeSeriesSplit` evaluation; bootstrap a CI for AUC.

## Further reading
scikit-learn "Model evaluation" and "Cross-validation" guides; Saito & Rehmsmeier (2015) on PR vs ROC; Kaufman et al. (2012) "Leakage in data mining."
