# Chapter 2.1 — Logistic Regression

[← Roadmap](../README.md) · Prev [1.4](../01-regression/04-quantile-bayesian-regression.md) · Next [2.2 Perceptron](02-perceptron.md) · Priority 🔴 · Code: [`classification_scratch.py`](../code/classification_scratch.py), [`classification_family.py`](../code/classification_family.py)

## 1. Identity & problem
Linear model for **probabilities of a binary (or multiclass) outcome**. "Will this customer churn? With what probability?" Despite the name, it's classification.

## 2. History
Logistic curve: Verhulst (1838, population growth). Logit as a statistical model: Berkson (1944); Cox (1958) regression for binary sequences; generalized linear models (Nelder & Wedderburn 1972) unified it with OLS, Poisson etc. Fitting by IRLS/Newton; modern L-BFGS/SAGA, and regularized versions (glmnet). It is also the single-neuron output layer of every classification net (softmax = multiclass generalization).
**Why not OLS on 0/1?** Predictions leave [0,1], residuals are non-Gaussian and heteroscedastic, and squared loss is a poor fit for probabilities.

## 3. Intuition
Compute a linear score `z = wᵀx + b` (any real number), then squash into (0,1) with the sigmoid. The score is the **log-odds**: each unit of a feature multiplies the odds by `e^{w_j}`.
```
 p │          ______ 1
   │        /
   │      /       sigmoid(z) = 1/(1+e^-z)
   │ ____/ 
 0 └────────────── z   (z=0 → p=0.5 → decision boundary is the hyperplane wᵀx+b=0)
```

## 4. Math
`p(y=1|x) = σ(wᵀx+b)`, `log(p/(1−p)) = wᵀx+b`.
Bernoulli likelihood ⇒ loss (cross-entropy / log loss): `J = −Σ[yᵢ log pᵢ + (1−yᵢ) log(1−pᵢ)] + (λ/2)‖w‖²`.
Gradient: `∇J = Xᵀ(p−y) + λw` (beautifully the same form as OLS with residual `p−y`). Hessian `XᵀSX + λI`, `S=diag(p(1−p))` ⇒ convex ⇒ Newton/IRLS converges in a handful of steps.
Multiclass: **softmax** `p_k = e^{z_k}/Σe^{z_j}` with cross-entropy.
Assumptions: log-odds linear in features; independent observations; no perfect separation (else unregularized weights → ∞); little collinearity for interpretation. Does **not** assume normality of features.

**Worked example.** `x=[1,2,3,4]`, `y=[0,0,1,1]`, start `w=b=0` ⇒ all `p=0.5`. Gradient (mean) = `Xᵀ(p−y)/n`: intercept `0.5(1)+0.5(1)−0.5−0.5 = 0` → 0; slope `(0.5·1+0.5·2−0.5·3−0.5·4)/4 = −0.5`. Step with lr=1: `b=0, w=0.5` ⇒ new probabilities σ(0.5x) = 0.62, 0.73, 0.82, 0.88; the class-1 points move up (and class-0 too — one step isn't enough, the intercept will go negative as training continues). This data is **perfectly separable**: unregularized MLE has no finite solution (weights grow forever) — the reason sklearn defaults to L2 (C=1).
**Interpretation:** coefficient 0.5 ⇒ odds ×e^0.5 ≈ 1.65 per unit x.

## 5. Training
Solvers: `lbfgs` (default, dense, smooth L2), `newton-cholesky` (n≫p), `liblinear` (small, L1), `saga` (large, L1/ElasticNet), SGD (streaming). Stop on gradient/loss tolerance or `max_iter`. Cost per iteration `O(np)`; model `O(p)`; prediction `O(p)`.

## 6. Evolution
| Method | Improvement | Cost |
|---|---|---|
| Perceptron (hard threshold) → logistic | Probabilistic output, convex smooth loss | — |
| + L1/L2/ElasticNet | Stability, selection | Tune C |
| Multinomial/softmax | True multiclass | — |
| Ordinal logistic, Poisson GLM | Ordered / count targets | Specialized |
| GAMs, trees, boosting, NNs | Nonlinear boundaries, interactions | Less interpretable, less calibrated (trees) |
Logistic remains the standard strong baseline for tabular classification and the calibration reference.

## 7. When to use / not
Use: need calibrated probabilities, interpretability (odds ratios), regulated domains (credit, medicine), high-dimensional sparse data (text), fast serving, baseline. Avoid: strongly nonlinear boundaries or interactions without engineered features; perfect separation without regularization; ultra-rare events without class weights/threshold tuning; unstructured data (images) without learned features.

## 8. Strengths / weaknesses
+ Convex, fast, probabilistic, interpretable, well-calibrated, scales to huge sparse data. − Linear boundary, sensitive to scaling when regularized, collinearity muddles coefficients, outlier leverage.

## 9. Hyperparameters
`C = 1/λ` (smaller C ⇒ stronger regularization), `penalty` (l1/l2/elasticnet), `class_weight`, `solver`, `max_iter`, `multi_class`, decision threshold (not a model parameter but critical). Measured on breast cancer (5-fold CV acc): C=1e-4 → **0.645** (underfit ≈ majority 0.627), 1e-2 → 0.949, **1 → 0.979**, 100 → 0.965.

## 10. Implementation
From scratch (IRLS + GD in [`classification_scratch.py`](../code/classification_scratch.py)):
```python
sigmoid = lambda z: 1/(1+np.exp(-np.clip(z,-35,35)))
def fit_irls(X, y, lam=1.0, iters=25):
    A = np.c_[np.ones(len(X)), X]; w = np.zeros(A.shape[1]); R = lam*np.eye(A.shape[1]); R[0,0] = 0
    for _ in range(iters):
        p = sigmoid(A@w); g = A.T@(p-y) + R@w
        H = A.T@(A*(p*(1-p))[:,None]) + R
        w -= np.linalg.solve(H, g)
    return w
```
Library:
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
```
Check: scratch IRLS vs sklearn coefficients agree to ~0.01 (max abs diff 0.0086; solver tolerance), test accuracy both 0.958.

## 11. Evaluation
Log loss/Brier (probability quality), ROC-AUC, PR-AUC, F1 at a chosen threshold, calibration curve, confusion matrix by cost (Ch 0.5). Stratified CV. Baseline: majority class. Breast cancer 5-fold CV: **acc 0.979, AUC 0.995** vs majority 0.627 / 0.500.

## 12. Case study
Breast-cancer diagnosis (569×30). Logistic with standardized features reaches 0.979 CV accuracy, equaling or beating the more flexible RBF-SVM (0.977), NB (0.939), kNN (0.965), LDA (0.954). Lesson: with small, nearly linearly separable tabular data a regularized linear model is hard to beat. In medicine, tune the threshold for recall (missed cancers cost more than false alarms).

## 13. Comparison
vs LDA/NB: logistic is discriminative, fewer distributional assumptions, usually better with large n; generative models can win with tiny n (Ng & Jordan 2001). vs SVM: logistic gives probabilities, SVM can use kernels cheaply. vs trees/GBMs: they capture nonlinearity/interactions; logistic is more stable and interpretable.

## 14. Pitfalls
Forgetting scaling · interpreting coefficients under collinearity · separation · using accuracy on imbalanced data · assuming 0.5 threshold · class_weight distorting probabilities (recalibrate).

## 15. Further reading
Cox (1958); Nelder & Wedderburn (1972); Hosmer, Lemeshow & Sturdivant, *Applied Logistic Regression*; ESL Ch 4; Ng & Jordan (2001) "On discriminative vs. generative classifiers"; sklearn "Logistic regression" guide.

## Interview Q&A
- *Why not use MSE for logistic?* Non-convex in w with sigmoid, weak gradients when confidently wrong; cross-entropy is the Bernoulli MLE and convex.
- *Interpret a coefficient.* Change in log-odds per unit (odds ratio `e^w`), others fixed.
- *Linear boundary?* Yes: `wᵀx+b=0`. Use interactions/kernel/trees for non-linear.
- *Handle imbalance?* Class weights or threshold tuning, PR-AUC; recalibrate if probabilities matter.

## Exercises
Implement softmax regression; show separation blow-up with C=∞; compare calibration curves of logistic vs random forest; derive the gradient.
