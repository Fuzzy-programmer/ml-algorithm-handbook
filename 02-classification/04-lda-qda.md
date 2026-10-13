# Chapter 2.4 — Linear & Quadratic Discriminant Analysis (LDA / QDA)

[← Roadmap](../README.md) · Prev [2.3](03-naive-bayes.md) · Next [2.5 k-NN](05-knn.md) · Priority 🟡 · Code: [`classification_family.py`](../code/classification_family.py)

## 1. Identity & problem
Generative classifiers that model each class as a Gaussian. LDA shares one covariance across classes (linear boundary); QDA gives each class its own (quadratic boundary). LDA also acts as a **supervised dimensionality reducer**.

## 2. History
**Fisher (1936)** "The use of multiple measurements in taxonomic problems" (iris data): find the projection that maximizes between-class over within-class scatter. Rao (1948) extended to multiple classes; the Bayes/Gaussian derivation gives the same direction under equal covariances. Later: regularized discriminant analysis (Friedman 1989), shrinkage covariance (Ledoit–Wolf 2004), sparse LDA. Relationship to logistic regression: same linear form of log-odds, different estimation (generative vs discriminative, Efron 1975).

## 3. Intuition
Fit a bell-shaped cloud to each class. A new point belongs to whichever cloud makes it most probable (weighted by class frequency). Shared shape ⇒ boundary is a straight line; separate shapes ⇒ curved.

## 4. Math
Class density `p(x|k)=𝒩(μ_k, Σ_k)`, prior `π_k`. Discriminant: `δ_k(x) = log π_k + log p(x|k)`.
- **LDA** (`Σ_k=Σ`): `δ_k(x) = xᵀΣ⁻¹μ_k − ½μ_kᵀΣ⁻¹μ_k + log π_k` — linear in x.
- **QDA**: adds `−½log|Σ_k| − ½(x−μ_k)ᵀΣ_k⁻¹(x−μ_k)` — quadratic.
- **Fisher view:** maximize `J(w)=wᵀS_B w / wᵀS_W w`; at most `min(K−1, p)` discriminant directions.
- Parameter count: LDA ≈ `Kp + p(p+1)/2`; QDA ≈ `Kp + K·p(p+1)/2` ⇒ QDA needs far more data.
Assumptions: Gaussian classes; common covariance (LDA); no singular covariances.

**Worked example (1-D).** Class means 0 and 4, shared variance σ²=1. Equal priors ⇒ boundary at the midpoint **2**. Priors 0.8 (class 0) / 0.2 (class 1): `x* = (μ₀+μ₁)/2 + σ²/(μ₁−μ₀)·ln(π₀/π₁) = 2 + (1/4)ln 4 = 2.347` — the boundary shifts toward the rarer class, so more space goes to the common class.

## 5. Training
Closed form: estimate class means, pooled covariance, priors. `O(np² + p³)`. Solvers: `svd` (default, stable), `lsqr`/`eigen` (support shrinkage). Prediction `O(pK)`.

## 6. Evolution & trade-offs
| Method | Gain | Cost |
|---|---|---|
| Naive Bayes → LDA | Models feature correlations | Estimates covariance (p² params) |
| LDA → QDA | Class-specific shapes, curved boundary | Many parameters, unstable when n small |
| QDA/LDA → Regularized DA / shrinkage | Stable covariance when p ~ n | Tuning |
| LDA → logistic regression | Fewer assumptions, robust to non-Gaussian | Slower fit |
| LDA → kernel/flexible DA, SVM | Non-Gaussian, nonlinear | More tuning |

## 7. When to use / not
Use: small samples with roughly Gaussian classes, multi-class problems (compact, fast), dimensionality reduction before another classifier, interpretable discriminant axes, low-latency scoring. Avoid: heavy-tailed/multimodal classes, categorical features, p ≥ n without shrinkage, QDA with scarce data.

## 8. Strengths / weaknesses
+ Closed form, fast, multiclass-native, low variance (LDA), built-in projection. − Gaussian assumption, sensitive to outliers, singular covariance with collinearity (QDA fails), shared-covariance assumption.

## 9. Hyperparameters
LDA: `solver`, `shrinkage` (`'auto'` Ledoit–Wolf or float 0–1), `priors`, `n_components` (projection), `tol`. QDA: `reg_param` (blend toward identity), `priors`.

## 10. Implementation
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
lda  = LDA(solver="lsqr", shrinkage="auto")        # stable when p is large relative to n
qda  = QDA(reg_param=0.1)
Z = LDA(n_components=2).fit_transform(X, y)         # supervised projection
```
From scratch: pooled covariance `Σ = Σ_k (n_k−1)Σ̂_k/(n−K)`; compute `δ_k(x)` and take argmax.

## 11. Evaluation & results (measured)
Breast cancer, 5-fold CV: LDA acc **0.954** (AUC 0.990); QDA with `reg_param=0.1` acc 0.956 (AUC 0.989). **Default QDA failed to fit** (`covariance matrix of class 0 is not full rank`) because the 30 features are strongly collinear — the textbook QDA failure; regularization fixed it. Moons dataset (nonlinear): LDA 0.852 vs logistic 0.854 vs RBF-SVC 0.932 — LDA's linear boundary can't bend.

## 12. Comparison
Close in accuracy to logistic regression when classes are near-Gaussian; logistic wins as data departs from Gaussianity. LDA wins on tiny samples or when you want a projection. In practice: LDA as baseline/feature extractor, not the final model.

## Interview Q&A
- *LDA vs PCA?* PCA is unsupervised (max variance); LDA uses labels (max class separation). Max `K−1` components.
- *LDA vs logistic regression?* Same linear log-odds; LDA estimates generatively under Gaussian assumptions, logistic discriminatively.
- *When does QDA beat LDA?* Clearly different class covariances and plenty of data per class.

## Exercises
Derive the Fisher criterion solution; plot LDA projection of iris; compare LDA with shrinkage vs without as p→n.

## Further reading
Fisher (1936) *Annals of Eugenics*; Friedman (1989) "Regularized discriminant analysis"; ESL Ch 4.3; Ledoit & Wolf (2004).
