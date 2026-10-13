# Chapter 6.4 — Causal Inference: Potential Outcomes, Propensity Scores, IPW, Matching, DiD, Causal Forests

[← Roadmap](../README.md) · Prev [6.3](03-reinforcement-learning.md) · Next [6.5 Survival analysis](05-survival-analysis.md) · Priority 🟡 · Code: [`causal_survival_semisup.py`](../code/causal_survival_semisup.py)

## 1. Identity & problem
Predictive ML answers "what will happen?"; causal inference answers "what would happen **if we intervened**?" — does the discount increase retention, does the drug work, which customers are persuadable? Observational data are confounded: people who received treatment differ from those who did not.

## 2. History
Neyman (1923) potential outcomes for experiments; Fisher's randomization; **Rubin (1974)** potential-outcomes framework; Wright (1920s path analysis) and **Pearl** (graphical models, do-calculus, 1995; *Causality* 2000); **Rosenbaum & Rubin (1983)** propensity score; Horvitz–Thompson IPW (1952); instrumental variables (Wright 1928; Angrist, Imbens & Rubin 1996 LATE); regression discontinuity (Thistlethwaite & Campbell 1960); difference-in-differences (Ashenfelter & Card 1985; Card & Krueger 1994); doubly robust estimators (Robins et al. 1994); **Double/Debiased ML** (Chernozhukov et al. 2018); meta-learners (Künzel et al. 2019); **causal forests** (Wager & Athey 2018); uplift modeling in marketing. Two main schools (Rubin potential outcomes, Pearl graphs) are complementary.

## 3. Intuition
For each person there are two potential outcomes, `Y(1)` treated and `Y(0)` untreated, but only one is observed — the **fundamental problem of causal inference**. Randomization (or a convincing substitute) makes the untreated group a valid stand-in for what the treated group would have experienced.

## 4. Math
Estimands: ATE `E[Y(1)−Y(0)]`, ATT, CATE `τ(x)=E[Y(1)−Y(0)|X=x]`.
Identification assumptions: **unconfoundedness** `(Y(0),Y(1)) ⫫ T | X`; **overlap** `0<e(x)<1` where `e(x)=P(T=1|X=x)`; **SUTVA** (no interference, consistent treatment). Randomized experiments satisfy the first two by design.
Estimators: regression adjustment; **IPW** `τ̂ = mean(TY/e) − mean((1−T)Y/(1−e))`; matching; stratification; **doubly robust (AIPW)**; meta-learners — S, T (separate models per arm), X, R, DR; double ML; causal forests.
**DiD:** `τ̂ = (Ȳ_T,post − Ȳ_T,pre) − (Ȳ_C,post − Ȳ_C,pre)`, valid under *parallel trends*.
**Worked example (measured).** Simulated confounded data: x → treatment and outcome; true ATE **2.0**.
| Estimator | Estimate |
|---|---|
| Naive difference in means | **5.18** (heavily biased by confounding) |
| Regression adjustment (T + x) | 2.034 |
| IPW (estimated propensity) | 2.068 |
| Matching on x (1-NN, ATT) | 2.033 |
| Unobserved confounder, regress on T only | **5.15** (no method using only observed variables can fix this) |
Propensity scores ranged 0.002–0.997 — near-violations of overlap make IPW weights huge and unstable (clip/trim in practice).
**DiD (simulated, true effect 5, common trend +3, baseline gap 4):** post-only comparison = **9.06** (wrong), DiD = **5.04**.
**Heterogeneous effects (T-learner with boosted trees; true τ(x)=1+2x):** at x=−0.8: true −0.60, est −0.56; x=0: 1.00 vs 1.06; x=+0.8: 2.60 vs 2.70.

## 5. Training & cost
Standard supervised models inside estimators (logistic/GBM for propensity, outcome models), cross-fitting to avoid overfitting bias. Cost is modest; the real cost is design, data collection, and assumption checking.

## 6. When to use / not
Use: decisions about interventions; policy evaluation; A/B tests with heterogeneity analysis; observational studies with good covariates, natural experiments (DiD, RDD, IV). Avoid: causal claims from predictive feature importance; adjusting for colliders/mediators; conditioning on post-treatment variables; poor overlap; strong unobserved confounding. **Always prefer randomized experiments when feasible.**

## 7. Hyperparameters & design choices
Covariate set (guided by a DAG, not by predictive power), propensity model/trimming thresholds, matching caliper, learner types, cross-fitting folds, number of trees (causal forest), clustering of standard errors, placebo/falsification tests.

## 8. Implementation
```python
e = LogisticRegression().fit(X, t).predict_proba(X)[:, 1]
ate_ipw = np.mean(t*y/e) - np.mean((1-t)*y/(1-e))
m1 = GradientBoostingRegressor().fit(X[t==1], y[t==1]); m0 = GradientBoostingRegressor().fit(X[t==0], y[t==0])
cate = m1.predict(X) - m0.predict(X)                      # T-learner
```
Libraries: `DoWhy`, `EconML`, `CausalML`, `statsmodels`, `grf` (R) for causal forests. (Full simulation code in `causal_survival_semisup.py`.)

## 9. Evaluation
No ground truth in real data: use **randomized experiments as the gold standard**, covariate balance checks (standardized mean differences), overlap plots, placebo tests, sensitivity analysis to unobserved confounding (E-values, Rosenbaum bounds), uplift/Qini curves for CATE ranking, and agreement across estimators. Simulations (like above) validate method behavior.

## 10. Case study (template)
Marketing coupon: run an A/B test; estimate ATE with covariate adjustment (CUPED); fit uplift models on the experiment to target persuadable customers; evaluate with Qini curve on a held-out randomized sample; deploy with a holdout group to monitor lift.

## 11. Comparison
| Method | Needs | Handles unobserved confounding? |
|---|---|---|
| RCT | Randomization | Yes (by design) |
| Adjustment/IPW/matching | Unconfoundedness + overlap | No |
| DiD | Panel/repeated data, parallel trends | Time-invariant ones |
| IV | Valid instrument | Yes (if valid) |
| RDD | Assignment cutoff | Locally |

## Interview Q&A
- *Correlation vs causation?* Confounders create associations without causal effect (naive estimate 5.18 vs true 2.0).
- *What is a propensity score?* P(treatment | covariates); balancing score.
- *Why DiD?* Removes time-invariant group differences and common shocks; needs parallel trends.
- *Pitfall of "controlling for everything"?* Conditioning on colliders/mediators introduces bias.
- *ATE vs CATE?* Average vs covariate-specific effect (uplift).

## Exercises
Draw a DAG with a collider and show bias from adjusting for it by simulation; implement doubly robust estimation; run a placebo-outcome test on the simulated data.

## Further reading
Hernán & Robins, *Causal Inference: What If* (free); Pearl, Glymour & Jewell, *Causal Inference in Statistics: A Primer*; Imbens & Rubin, *Causal Inference*; Chernozhukov et al. (2018); Wager & Athey (2018); Facure, *Causal Inference for the Brave and True* (free); Kohavi, Tang & Xu, *Trustworthy Online Controlled Experiments*.
