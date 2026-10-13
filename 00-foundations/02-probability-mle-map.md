# Chapter 0.2 — Probability, MLE and MAP

[← Roadmap](../README.md) · Prev [0.1](01-math-essentials.md) · Next [0.3 Optimization](03-optimization.md) · Priority 🔴

## 1. Why this matters
Almost every loss function is a negative log-likelihood in disguise: squared error ⇔ Gaussian noise, cross-entropy ⇔ Bernoulli/categorical, L2 penalty ⇔ Gaussian prior, L1 penalty ⇔ Laplace prior. Knowing this lets you *derive* the right loss instead of memorizing.

## 2. Essentials
- **Conditional probability** `P(A|B) = P(A,B)/P(B)`; **Bayes' rule** `P(θ|D) = P(D|θ)P(θ)/P(D)` (posterior ∝ likelihood × prior).
- **Independence:** `P(A,B)=P(A)P(B)`. **Expectation** `E[X]`, **variance** `E[(X−μ)²]`, **covariance**.
- Key distributions: Bernoulli/Binomial (binary events), Categorical/Multinomial, Gaussian `𝒩(μ,σ²)`, Poisson (counts), Exponential/Weibull (waiting times), Beta (prior over a probability), Laplace.
- **Law of large numbers / CLT:** sample means concentrate and become Gaussian → why averages (and ensembles) are stable.

## 3. Maximum Likelihood Estimation (MLE)
Pick parameters that make the observed data most probable: `θ̂ = argmax_θ Π p(xᵢ|θ) = argmax Σ log p(xᵢ|θ)` (logs turn products to sums and avoid underflow).

**Worked example — coin, 7 heads in 10 flips.** `L(θ)=θ⁷(1−θ)³`, `ℓ=7logθ+3log(1−θ)`, `ℓ'=7/θ−3/(1−θ)=0 ⇒ θ̂=0.7`.

**Gaussian ⇒ least squares.** If `yᵢ=wᵀxᵢ+ε`, `ε~𝒩(0,σ²)`: `−log L = (1/2σ²)Σ(yᵢ−wᵀxᵢ)² + const` ⇒ MLE = OLS.
**Bernoulli ⇒ cross-entropy.** `−log L = −Σ[y log p + (1−y) log(1−p)]` ⇒ logistic regression loss.

## 4. MAP and regularization
Maximum a posteriori adds a prior: `θ̂ = argmax [log p(D|θ) + log p(θ)]`.
- Gaussian prior `w~𝒩(0,τ²)` ⇒ penalty `(1/2τ²)‖w‖²` = **Ridge**, with `λ = σ²/τ²`.
- Laplace prior ⇒ `‖w‖₁` = **Lasso**.

**Worked example (continued):** Beta(2,2) prior ("mild belief the coin is fair") ⇒ posterior Beta(2+7, 2+3), MAP = `(h+1)/(n+2) = 8/12 ≈ 0.667` vs MLE 0.7. With little data the prior pulls the estimate; with more data it washes out. Full Bayesian inference keeps the whole posterior (Bayesian regression, Ch 1.4).

## 5. Frequentist vs Bayesian (practical view)
Frequentist: parameters fixed, data random → confidence intervals, p-values. Bayesian: parameters have distributions → credible intervals, uncertainty propagation, priors. Engineers mix them: MLE/MAP for fitting, Bayesian ideas for uncertainty and tuning.

## 6. Expectation–Maximization (EM) in one paragraph
When latent variables make the likelihood intractable (Gaussian mixtures, HMMs): **E-step** compute soft assignments given current parameters; **M-step** re-estimate parameters using those soft assignments. Each iteration never decreases the likelihood; it can converge to a local optimum (Dempster, Laird & Rubin 1977). Used in Ch 3.5.

## 7. Code (verified: MLE 0.7, MAP 0.667)
```python
h, n = 7, 10
print("MLE", h/n, "MAP Beta(2,2):", (h+1)/(n+2))
```
See [`code/foundations_demos.py`](../code/foundations_demos.py).

## 8. Pitfalls
Zero counts (Naive Bayes → Laplace smoothing = a prior) · overconfident MLE on tiny data · confusing likelihood with probability of parameters · assuming Gaussian noise when errors are heavy-tailed (use Huber/Student-t).

## Interview Q&A
- *Why log-likelihood?* Sums are stable, differentiable, and the argmax is unchanged.
- *Ridge as a Bayesian statement?* MAP with a zero-mean Gaussian prior on weights.
- *MLE vs MAP with infinite data?* They converge (the prior's influence vanishes).

## Exercises
1. Derive the MLE of a Gaussian's mean and variance (note the biased `1/n` variance).
2. Derive the logistic loss from Bernoulli likelihood.
3. Show Laplace prior ⇒ L1 penalty.

## Further reading
Murphy, *Probabilistic ML: An Introduction*, Ch 2–4; Bishop PRML Ch 1–2; Dempster et al. (1977) EM paper.
