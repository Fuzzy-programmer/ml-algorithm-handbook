# Chapter 6.5 — Survival Analysis: Kaplan–Meier, Cox Proportional Hazards, Parametric & ML Models

[← Roadmap](../README.md) · Prev [6.4](04-causal-inference.md) · Next [6.6 Semi-supervised & weak supervision](06-semi-weak-supervised.md) · Priority 🟡 · Code: [`causal_survival_semisup.py`](../code/causal_survival_semisup.py)

## 1. Identity & problem
Model **time until an event** (churn, failure, death, default, relapse) when some subjects are **censored** (the event hasn't happened by the end of observation, or they dropped out). Ordinary regression/classification on "did it happen" or on observed times is biased because it ignores censoring.

## 2. History
Life tables (Graunt 1662; Halley); **Kaplan & Meier (1958)** product-limit estimator; log-rank test (Mantel 1966); **Cox (1972)** proportional hazards (semi-parametric); parametric models (Weibull, exponential, log-normal; accelerated failure time); competing risks (Fine & Gray 1999); frailty models; **random survival forests** (Ishwaran et al. 2008); gradient-boosted survival (XGBoost `survival:cox`/AFT); neural: DeepSurv (Katzman 2018), DeepHit (Lee 2018); discrete-time classification formulations. Concordance index (Harrell 1982) for evaluation.

## 3. Intuition
At each time an event occurs, look at who is still "at risk" and ask what fraction experienced the event; multiply the survival fractions. Censored subjects contribute information up to the time they leave.

## 4. Math
Survival `S(t)=P(T>t)`; hazard `h(t)=lim P(t≤T<t+dt | T≥t)/dt`; cumulative hazard `H(t)=∫h`, `S(t)=exp(−H(t))`.
**Kaplan–Meier:** `Ŝ(t)=Π_{t_i≤t} (1 − d_i/n_i)` (`d_i` events, `n_i` at risk at time `t_i`).
**Worked example (computed).** Times `[2,3,3,5,6,8]`, event indicator `[1,1,0,1,0,1]` (0 = censored):
- t=2: at risk 6, 1 event ⇒ S = 5/6 = 0.8333
- t=3: at risk 5, 1 event (one subject censored at 3 stays in the risk set until 3) ⇒ S = 0.8333·4/5 = 0.6667
- t=5: at risk 3, 1 event ⇒ S = 0.6667·2/3 = 0.4444
- t=8: at risk 1, 1 event ⇒ S = 0.
`lifelines` gives the same S(5)=0.4444.
**Cox model:** `h(t|x)=h₀(t)·exp(βᵀx)`; `β` estimated by the **partial likelihood** `Π_{events} exp(βᵀx_i)/Σ_{j∈R(t_i)} exp(βᵀx_j)` without specifying `h₀`. `exp(β_j)` = hazard ratio per unit of feature j. **Assumption:** proportional hazards (ratios constant over time) — check Schoenfeld residuals; fix with stratification, time-varying covariates, or other models. Other assumptions: non-informative censoring, independent subjects, log-linear covariate effects.
**Parametric/AFT:** `log T = βᵀx + σε` (Weibull etc.) — gives full survival curves and extrapolation beyond observed time.

## 5. Training & cost
KM is O(n log n). Cox fits by Newton–Raphson on the partial likelihood (O(n·p²)-ish with ties handling). Random survival forests and GBMs cost like their non-survival analogues. Prediction: risk score / survival curve per subject.

## 6. When to use / not
Use: churn timing, predictive maintenance, clinical trials, credit default timing, time-to-conversion, any censored duration data. Avoid: plain regression on durations with censored rows dropped or treated as events; classification at a fixed horizon if censoring is heavy (or use IPCW); proportional-hazards model when hazards clearly cross (use AFT, stratified, or ML models).

## 7. Hyperparameters
Cox: penalizer (ridge/lasso), tie method (Efron), strata, time-varying covariates; AFT: distribution family; forests: trees, min node size, `max_features`; GBM: usual boosting knobs with survival objective; discretization of time for classification-style approaches.

## 8. Implementation
```python
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.datasets import load_rossi
km = KaplanMeierFitter().fit(durations, event_observed)
cph = CoxPHFitter(penalizer=0.01).fit(df, duration_col="week", event_col="arrest"); cph.print_summary(); cph.check_assumptions(df)
# scikit-survival: RandomSurvivalForest, GradientBoostingSurvivalAnalysis, concordance_index_censored
```

## 9. Evaluation & results (measured)
Rossi recidivism data (432 released prisoners, time to rearrest, censored at 52 weeks) — Cox model: `fin` (financial aid) coef −0.379 ⇒ hazard ratio **0.684** (≈32% lower hazard, p=0.047); `age` coef −0.057 ⇒ HR **0.944** per year (≈5.6% lower hazard per year older, p=0.009); `race` (HR 1.369, p=0.308) and `wexp` (HR 0.861, p=0.480) not significant at conventional levels. **Concordance index 0.64** (0.5 = random, 1 = perfect) — modest discrimination, typical for behavioral outcomes. (Treat hazard ratios as associations, not causal effects, unless the design supports it—Ch 6.4.)
Metrics: concordance index (ranking), time-dependent AUC, **integrated Brier score** (calibration + discrimination over time, with IPCW for censoring), calibration plots at chosen horizons, log-rank test for group comparison. Validate on out-of-time/held-out patients.

## 10. Case study (template)
Subscription churn: KM curves by plan; Cox with features and time-varying usage; compare with random survival forest and a monthly discrete-time LightGBM; evaluate C-index and calibration at 3/6/12 months; use predicted survival curves to target retention offers (and causal designs to test them).

## 11. Comparison
| Model | Pros | Cons |
|---|---|---|
| Kaplan–Meier | Nonparametric, easy | No covariates (except by strata) |
| Cox PH | Interpretable HRs, no baseline assumption | Proportional hazards, linear |
| AFT/Weibull | Full curves, extrapolation | Distribution assumption |
| Random survival forest / GBM | Nonlinear, interactions | Less interpretable |
| Deep survival | Complex inputs | Data hungry, tuning |

## Interview Q&A
- *What is censoring?* Event not observed within follow-up; right-censoring is most common. Dropping or treating censored as events biases estimates.
- *Hazard vs survival function?* Instantaneous risk given survival so far vs probability of surviving past t.
- *Proportional hazards assumption?* Ratio of hazards between subjects is constant over time.
- *Interpret HR=0.68?* 32% lower instantaneous event rate at any time, holding others fixed.
- *What does the C-index measure?* Probability that, for a comparable pair, the one with higher predicted risk fails first.

## Exercises
Hand-compute KM for a new dataset; fit Cox and test proportional hazards; compare Weibull AFT vs Cox predicted median survival; build a discrete-time hazard model with logistic regression.

## Further reading
Kaplan & Meier (1958); Cox (1972) *JRSS B*; Klein & Moeschberger, *Survival Analysis*; Harrell, *Regression Modeling Strategies*; `lifelines` and `scikit-survival` documentation; Kvamme, Borgan & Scheel (2019) on Cox-based neural nets.
