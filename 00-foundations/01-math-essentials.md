# Chapter 0.1 — Linear Algebra & Calculus You Actually Need

[← Roadmap](../README.md) · Next: [0.2 Probability, MLE/MAP](02-probability-mle-map.md) · Priority 🔴

Goal: the minimum math to read every later chapter. Each idea is tied to where it reappears.

## 1. Vectors, matrices, shapes
- A dataset is a matrix `X ∈ ℝ^{n×p}` (n rows = examples, p columns = features); targets `y ∈ ℝ^n`.
- **Dot product** `a·b = Σ aᵢbᵢ = ‖a‖‖b‖cosθ` → similarity (k-NN, attention, cosine retrieval in RAG).
- **Matrix multiply** `(n×p)(p×m) → n×m`. Most "model code" is shape bookkeeping; **always write shapes next to equations.**
- **Norms:** `‖w‖₂ = √Σwᵢ²` (L2, Ridge), `‖w‖₁ = Σ|wᵢ|` (L1, Lasso), `‖w‖₀` = count of non-zeros.
- **Transpose, inverse:** `(AB)ᵀ = BᵀAᵀ`; `A⁻¹` exists only if A is square and full rank. In code never form an inverse — solve (`np.linalg.solve`, `lstsq`).

## 2. Eigen-things and SVD
- **Eigenvector:** `Av = λv` — a direction A only stretches. For a symmetric matrix (e.g. a covariance), eigenvectors are orthogonal and eigenvalues real.
- **Covariance** `Σ = XcᵀXc/(n−1)` (Xc centered): eigenvectors = principal axes, eigenvalues = variance along them → **PCA** (Ch 4.3).
- **SVD:** `X = UΣVᵀ`. Gives best low-rank approximation (Eckart–Young), stable least squares, PCA, latent-factor recommenders (Ch 6.2).
- **Condition number** `κ = σ_max/σ_min`: large κ ⇒ unstable solves and slow gradient descent (collinearity in Ch 1.1).
- **Positive (semi)definite:** `vᵀAv ≥ 0` for all v. Covariances and kernel matrices are PSD; a PSD Hessian means a convex quadratic.

## 3. Calculus for learning
- **Derivative** = slope; **gradient** `∇f` = vector of partials, points uphill; we step against it.
- **Chain rule** `d f(g(x))/dx = f'(g)·g'(x)` — backpropagation (Ch 5.1) is just this, applied layer by layer.
- **Hessian** `H` = matrix of second derivatives; its eigenvalues describe curvature. Newton step `w ← w − H⁻¹∇f`.
- **Convexity:** every local minimum is global (OLS, logistic, SVM, Lasso). Neural nets are non-convex but train well in practice.
- Useful gradients (memorize): `∇_w ‖y−Xw‖² = −2Xᵀ(y−Xw)`; `∇_w wᵀAw = (A+Aᵀ)w`; `σ'(z)=σ(z)(1−σ(z))`; `d/dz log σ(z) = 1−σ(z)`.

## 4. Worked example — gradient of a tiny model
Model `ŷ = w·x`, loss `L = (ŷ−y)²`, with `x=2, y=3, w=0.5`.
`ŷ = 1`, `L = 4`. `dL/dw = 2(ŷ−y)·x = 2(−2)(2) = −8`. With lr 0.1: `w ← 0.5 − 0.1(−8) = 1.3` ⇒ `ŷ = 2.6`, `L = 0.16`. One step reduced the loss 25×; a larger lr (e.g. 0.3 → w = 2.9 → ŷ = 5.8, L = 7.84) would overshoot — the learning-rate trade-off.

## 5. Information-theory bits you will meet
- **Entropy** `H(p) = −Σ p log p` (uncertainty; trees, Ch 3.1). **Cross-entropy** `H(p,q) = −Σ p log q` (classification and LLM loss). **KL** `D(p‖q) = H(p,q) − H(p) ≥ 0` (VAE, distillation, RLHF penalty).
- Worked: fair coin `H = 1` bit; a 90/10 coin `H = −0.9log₂0.9 − 0.1log₂0.1 ≈ 0.469` bits.

## 6. Code
```python
import numpy as np
X = np.random.default_rng(0).normal(size=(200, 3)); X[:, 2] = X[:, 0] + 0.1*np.random.default_rng(1).normal(size=200)
Xc = X - X.mean(0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
print("variance per axis:", (S**2/(len(X)-1)).round(3))   # third axis ~ 0 : redundant feature
print("condition number:", S[0]/S[-1])
```

## 7. Common mistakes
Forgetting to center before PCA · inverting matrices explicitly · mixing row/column conventions · confusing gradient (vector) with derivative (scalar) · believing "non-convex = untrainable".

## Interview Q&A
- *Why is the gradient the direction of steepest ascent?* Among unit directions u, `∇f·u` is maximized when u aligns with ∇f.
- *Relation between PCA and SVD?* Right singular vectors of centered X are the covariance eigenvectors; `σᵢ²/(n−1)` are the eigenvalues.
- *Why does a high condition number hurt GD?* Steps good for the steep direction are tiny for the flat one → slow or oscillating.

## Exercises
1. Compute ∇ of `L = ‖y − Xw‖² + λ‖w‖²` and solve for w (gives Ridge).
2. Show cross-entropy with one-hot p reduces to `−log q_correct`.
3. Verify the SVD/PCA equivalence numerically.

## Further reading
Strang, *Introduction to Linear Algebra* (and MIT 18.06 lectures); Deisenroth et al., *Mathematics for Machine Learning* (free); 3Blue1Brown "Essence of linear algebra/calculus".
