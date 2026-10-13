# Project 1 — End-to-End Tabular ML: Income Classification (OpenML "adult")

[← Roadmap](../README.md) · Code: [`project_adult_income.py`](../code/project_adult_income.py) · Uses: Ch 0.5, 0.6, 2.1, 3.4

A realistic pipeline you can reuse: baseline → pipeline → tuning → calibration → threshold → single final test → error analysis → model card. Data: 20,000-row random sample of OpenML `adult` (fetched with `fetch_openml`), 14 mixed features, 23.9% positives. Split: 12,000 train / 4,000 validation / 4,000 test (stratified; test touched once).

## 1. Problem framing
Predict whether annual income exceeds $50K from census attributes. (A standard benchmark; **do not** use such a model to make real decisions about people. The data are from 1994 and encode historical inequities.) Define upfront: primary metric PR-AUC (positive class is the minority), secondary ROC-AUC, log loss/Brier for probability quality; baseline = regularized logistic regression.

## 2. Steps and results (all measured)
| Step | Validation result |
|---|---|
| Baseline: standardized numerics + one-hot categoricals + logistic | AUC 0.905, PR-AUC 0.767, logloss 0.319, Brier 0.1028 |
| LightGBM, near-default (300 trees, lr 0.05, 15 leaves) | AUC 0.924, PR-AUC 0.821, logloss 0.283, Brier 0.0901 |
| + Randomized search (20 trials × 4-fold CV on train, scoring PR-AUC) | best CV PR-AUC 0.822; validation AUC 0.925, PR-AUC 0.823 |
| + Isotonic calibration (4-fold) | AUC 0.926, PR-AUC 0.825, logloss 0.280, Brier 0.0893 |
Best parameters found: `num_leaves 15, lr 0.05, n_estimators 400, min_child_samples 10, subsample 1.0, colsample 0.8, reg_lambda 1`.

**Lessons:** (1) the model *family* (+0.054 PR-AUC from logistic → boosted trees) mattered far more than *tuning* (+0.002); (2) calibration helped probabilities slightly (Brier 0.0894 → 0.0893 — a tiny gain because LightGBM was already well calibrated; check, don't assume); (3) without a baseline you couldn't say the gain is real.

## 3. Threshold selection
On validation predictions: F1-optimal threshold **0.36** (F1 0.730). With an example cost ratio (missed positive costs 5× a false alarm), the cost-optimal threshold is **0.16**. The "right" threshold is a business decision, not 0.5.

## 4. Final evaluation (test set, once)
| Model | AUC | PR-AUC | logloss | Brier |
|---|---|---|---|---|
| Baseline logistic (refit on train+val) | 0.906 | 0.774 | 0.319 | 0.1014 |
| **Final: tuned LightGBM + isotonic** (refit on train+val) | **0.926** | **0.827** | **0.287** | **0.0883** |
| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.50 | 0.774 | 0.650 | 0.707 |
| 0.36 (F1-opt) | 0.690 | 0.780 | 0.732 |
| 0.16 (cost-opt) | 0.532 | 0.924 | 0.675 |
Test results match validation closely (no sign of overfitting to validation).

## 5. Error and subgroup analysis (test)
| Slice | n | AUC | Actual positive rate | Mean predicted |
|---|---|---|---|---|
| Male | 2,647 | 0.906 | 0.308 | 0.302 |
| Female | 1,353 | 0.946 | 0.106 | 0.111 |
| Age < 30 | 1,214 | 0.948 | 0.046 | 0.051 |
| Age 30–50 | 1,904 | 0.904 | 0.311 | 0.316 |
| Age ≥ 50 | 882 | 0.887 | 0.350 | 0.324 |
Calibration is good within slices (mean prediction ≈ actual rate), but base rates differ greatly across groups (e.g. selection rate at 0.5: 26.3% for males vs 8.1% for females), so any use of this model for decisions requires fairness analysis, legal review and a clear definition of acceptable disparity—not a purely technical fix. Discrimination is weakest for ages ≥ 50 (AUC 0.887).

## 6. Explainability
Permutation importance (drop in validation PR-AUC): capital-gain 0.247, education-num 0.062, age 0.047, marital-status 0.043, capital-loss 0.039, relationship 0.025. Capital-gain dominates because it is highly informative (and zero for most people). Permutation importance on correlated features can understate shared signal (e.g. marital-status/relationship); use SHAP/partial dependence for deeper analysis.

## 7. Deliverables checklist (a real project)
- [ ] Problem statement, metric, cost matrix, success threshold agreed with stakeholders
- [ ] Data audit: leakage, duplicates, drift, missingness, label quality
- [ ] Baseline + single pipeline object (preprocessing + model) saved with versioned config
- [ ] Time/group-aware split documented
- [ ] Tuning with a budget; report tuned vs default honestly
- [ ] Calibration + threshold policy; fairness and slice analysis
- [ ] Model card (intended use, data, metrics, limits), monitoring plan (input drift, performance by slice, calibration), rollback plan
- [ ] Serving test: latency, memory, train/serve parity

## 8. Minimal serving sketch
```python
import joblib, pandas as pd
joblib.dump(final, "model.joblib")                    # pipeline includes preprocessing
model = joblib.load("model.joblib")
proba = model.predict_proba(pd.DataFrame([row]))[0, 1]
decision = proba >= 0.36                              # threshold from the business cost analysis
```

## Exercises
Re-run with a time-based split if you add a date; replace LightGBM with logistic + interactions and see how much gap remains; add monotone constraints on `education-num`; run a drift simulation by shifting `age`.
