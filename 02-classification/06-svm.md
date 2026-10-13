# Chapter 2.6 — Support Vector Machines (Linear, Kernel, SVR)

[← Roadmap](../README.md) · Prev [2.5](05-knn.md) · Next [3.1 Decision Trees](../03-trees-and-ensembles/01-decision-trees.md) · Priority 🟠 · Code: [`classification_family.py`](../code/classification_family.py)

## 1. Identity & problem
Margin-maximizing classifier/regressor. Finds the separating hyperplane with the widest gap; with kernels it separates non-linear data. SVR applies the idea to regression.

## 2. History
Vapnik & Chervonenkis: statistical learning theory, VC dimension, optimal hyperplane (1960s–70s) → **Boser, Guyon & Vapnik (1992)** kernel trick → **Cortes & Vapnik (1995)** soft margin ("support-vector networks") → SVR (Drucker et al. 1997), SMO training (Platt 1998), LIBSVM (Chang & Lin). Dominated classification in the late 1990s–2000s (text, bioinformatics, digits) until large-scale deep learning and boosting took over. Relation to perceptron: both linear separators; SVM picks the *unique max-margin* one.

## 3. Intuition
Among all separating lines, pick the one with the widest "street" between classes. Only the points on the curbs (**support vectors**) matter; the rest could be deleted without changing the model.
```
   + +   |      margin = 2/‖w‖
  + +  --|-- ← support vectors on the margin
        ---|
   - -   |  -
```
Kernel trick: map to a higher-dimensional feature space where a line works, without ever computing the mapping.

## 4. Math
Hard margin: `min ½‖w‖²  s.t. y_i(wᵀx_i+b) ≥ 1`. Soft margin: `min ½‖w‖² + C Σξ_i  s.t. y_i(wᵀx_i+b) ≥ 1−ξ_i, ξ_i≥0` ⇔ minimize **hinge loss** `Σ max(0, 1−y_i f(x_i)) + (1/2C)‖w‖²`.
Dual: `max Σα_i − ½ΣΣ α_iα_j y_iy_j K(x_i,x_j)  s.t. 0≤α_i≤C, Σα_iy_i=0`; prediction `f(x)=Σ α_i y_i K(x_i,x)+b` over support vectors (α_i>0).
Kernels: linear `xᵀx'`; polynomial `(γxᵀx'+r)^d`; **RBF** `exp(−γ‖x−x'‖²)`; Mercer condition = PSD kernel matrix.
SVR: ε-insensitive loss `max(0,|y−f|−ε)`.
Assumes: scale-sensitive; margin is a meaningful notion; kernel suits the data.

**Worked example (hard margin).** Points x=−1 (class −), x=+1 (class +), 1-D. Constraints: `w·(−1)+b ≤ −1`, `w·1+b ≥ 1` ⇒ minimal `|w|=1`, `b=0`, margin width `2/|w|=2`; both points are support vectors. Adding a far point at x=3 (+) changes nothing (not a support vector). **Kernel example:** 1-D data − at x=0, + at x=−2 and +2 isn't linearly separable, but mapping `φ(x)=x²` gives −→0, +→4, separable at 2.

## 5. Training
Convex quadratic program. SMO/libsvm: `O(n²)`–`O(n³)` time, `O(n²)` kernel cache; prediction `O(#SV·p)`. Linear SVM via `liblinear`/SGD scales `O(n·p)`. Rule of thumb: kernel SVM comfortable up to ~tens of thousands of rows; beyond that use linear SVM, kernel approximation (Nystroem, random Fourier features) or other models. **Scale features.** SVC probabilities (Platt scaling, 5-fold internally) are costly and approximate.

## 6. Evolution
| Step | Gain | Cost |
|---|---|---|
| Perceptron → hard-margin SVM | Unique max-margin solution, generalization bound | Needs separability |
| → soft margin (C) | Handles noise | Tune C |
| → kernel SVM | Non-linear boundaries | Quadratic+ scaling, kernel tuning |
| → SVR, one-class SVM | Regression, novelty detection | — |
| → Linear SVM at scale (liblinear, Pegasos) | Millions of sparse features | Linear boundary |
| SVM → boosted trees/deep nets | Better on large/structured data | Different tuning/compute |

## 7. When to use / not
Use: small–medium n, high-dimensional features (text with linear SVM), clean margin, need nonlinear boundary with strong regularization control. Avoid: very large n (kernel), heavy noise/overlap without careful C, need for calibrated probabilities, many categorical features, interpretability needs.

## 8. Strengths / weaknesses
+ Strong generalization in high dimension, convex (global optimum), sparse solution (support vectors), flexible kernels. − Slow on big data, needs scaling and tuning (C, γ), no native probabilities, opaque kernel models.

## 9. Hyperparameters
`C` (↑ narrower margin, fewer violations, overfit risk), `kernel`, `gamma` (RBF width; ↑ = wigglier boundary, overfit; `scale` default = 1/(p·Var(X))), `degree`/`coef0` (poly), `epsilon` (SVR tube), `class_weight`, `tol`. Tune C and γ on a log grid.

## 10. Implementation
```python
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
pipe = make_pipeline(StandardScaler(), SVC())
gs = GridSearchCV(pipe, {"svc__C": [.1,1,10,100], "svc__gamma": [1e-3,1e-2,1e-1,1]}, cv=5).fit(X, y)
lin = make_pipeline(StandardScaler(), LinearSVC())      # n large / sparse text
```
From scratch: sub-gradient descent on hinge loss (Pegasos): `if y(w·x)<1: w ← (1−ηλ)w + ηyx else w ← (1−ηλ)w`.

## 11. Evaluation & results (measured)
Breast cancer, 5-fold CV: LinearSVC acc 0.968 (AUC 0.993); **RBF SVC 0.977 (AUC 0.996)**; grid search best C=100, γ=0.001 → **0.981** (CV-selected on the same folds, so slightly optimistic; use nested CV for an honest figure). Over-flexible RBF (C=1000, γ=1) collapses to **0.631** ≈ majority baseline (0.627): memorized training data. Moons: RBF-SVC 0.932 vs LinearSVC 0.856.
Fit time on synthetic 20-feature data (seconds): n=2k / 8k / 16k → Logistic 0.01/0.02/0.06, GaussianNB 0.00/0.01/0.01, **RBF SVC 0.09/2.7/5.6** — superlinear growth, the practical limit of kernel SVMs.
Text (4 newsgroups): TF-IDF + LinearSVC **0.858** vs Logistic 0.844 vs MultinomialNB 0.842 — linear SVM is a strong, cheap text baseline.

## 12. Case study
Text classification: `TfidfVectorizer(sublinear_tf=True)` + `LinearSVC` — default winner among classical text models for years. Digits/small images: RBF SVM until CNNs.

## 13. Comparison
vs logistic: similar linear performance; SVM margin vs probabilities. vs k-NN: SVM compact, trained boundary. vs boosted trees: trees handle mixed types/missing values/interactions out of the box and scale better; SVM can win on small high-dimensional numeric data.

## Interview Q&A
- *What are support vectors?* Training points with α>0 — on/inside the margin; they alone define the boundary.
- *What does C do?* Trades margin width against training errors.
- *Kernel trick?* Compute inner products in feature space via K(x,x') without mapping.
- *Why scale?* Margin and RBF distance depend on feature units.
- *Why not for huge data?* O(n²–n³) training.

## Exercises
Derive the dual; implement Pegasos; visualize how γ changes the RBF boundary; compare SVR with Ridge on a nonlinear regression.

## Further reading
Cortes & Vapnik (1995) *Machine Learning*; Boser, Guyon, Vapnik (1992); Platt (1998) SMO; Schölkopf & Smola, *Learning with Kernels*; Chang & Lin (2011) LIBSVM; sklearn SVM guide.
