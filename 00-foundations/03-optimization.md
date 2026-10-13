# Chapter 0.3 — Optimization: GD, SGD, Momentum, Adam/AdamW

[← Roadmap](../README.md) · Prev [0.2](02-probability-mle-map.md) · Next [0.4](04-bias-variance-regularization.md) · Priority 🔴

## 1. Problem
Minimize a loss `J(θ)` (average over data plus optional penalty). Convex losses (OLS, logistic, SVM) have a global optimum; deep nets don't, but gradient methods still work.

## 2. Lineage
Gradient descent (Cauchy 1847) → stochastic approximation (Robbins–Monro 1951) → momentum (Polyak 1964) / Nesterov (1983) → per-parameter rates: AdaGrad (Duchi 2011) → RMSProp (Hinton, lecture 2012) → Adam (Kingma & Ba 2014) → AdamW (Loshchilov & Hutter 2017, decoupled weight decay).

## 3. Methods and equations
| Method | Update | Idea | Cost / catch |
|---|---|---|---|
| Batch GD | `θ ← θ − η∇J` | exact gradient | O(n) per step |
| SGD / mini-batch | gradient on a batch of size B | noisy but cheap, noise helps escape plateaus | needs lr schedule |
| Momentum | `v←βv+g; θ←θ−ηv` | accumulates velocity; damps oscillation in narrow valleys | extra state |
| Nesterov | gradient at look-ahead point | slightly better correction | — |
| AdaGrad | `θ ← θ − η g/√(Σg²+ε)` | big steps for rare features | rate decays to zero |
| RMSProp | EMA of g² instead of sum | fixes AdaGrad decay | — |
| Adam | `m←β₁m+(1−β₁)g`, `v←β₂v+(1−β₂)g²`, `θ←θ−η m̂/(√v̂+ε)` (bias-corrected) | momentum + per-parameter scale | can generalize slightly worse than tuned SGD on some vision tasks; L2-in-Adam ≠ weight decay |
| AdamW | Adam + `−ηλθ` applied directly | decoupled weight decay | standard for Transformers |

Defaults: Adam `β₁=0.9, β₂=0.999, ε=1e-8`; typical lr 1e-3 (small nets), 1e-4–3e-4 (Transformers, with warmup).

## 4. Learning-rate practice
Most important hyperparameter. Use a range test, **warmup** then **cosine/linear decay**, gradient clipping for RNNs/Transformers. Batch size ↑ ⇒ usually lr ↑ (roughly linear scaling, with warmup).

## 5. Worked example
`f(x,y)=½(x²+25y²)`, start (5,5), `∇f=(x,25y)`. Stability needs `η<2/25=0.08` (curvature of steep axis). With `η=0.07`: x factor `1−0.07=0.93`, y factor `1−1.75=−0.75` per step (oscillates but shrinks); the *flat* axis x only shrinks 7% per step ⇒ slow. That mismatch is the condition-number problem. Measured after 100 steps:
```
GD (lr 0.07)    final loss 6.2e-06
Momentum        final loss 2.8e-03
Adam (lr 0.3)   final loss 6.0e-03
```
**Honest reading:** on a clean 2-D quadratic, *well-tuned plain GD wins*; my momentum/Adam settings were not tuned. Adaptive methods earn their keep on noisy, badly scaled, high-dimensional problems where one global lr can't fit all parameters — don't generalize from a toy.

## 6. When to use what
| Situation | Choice |
|---|---|
| Small convex problem | L-BFGS / closed form |
| Large linear/logistic | SGD, SAGA, or coordinate descent |
| Deep nets (default) | AdamW + warmup + cosine decay |
| Vision CNNs, long training | SGD+momentum (often competitive/better with tuning) |
| Sparse gradients (embeddings) | Adam/AdaGrad family |

## 7. Failure modes
Divergence (lr too high, unscaled features) · plateaus/saddles · exploding/vanishing gradients (clip, normalize, residuals) · lr scheduling forgotten · evaluating a noisy final iterate (use EMA/Polyak averaging).

## 8. Code
From-scratch Adam is in [`code/foundations_demos.py`](../code/foundations_demos.py); the linear-regression GD with a divergence demo is in [Ch 1.1](../01-regression/01-linear-regression.md). In PyTorch: `torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)`.

## Interview Q&A
- *Why does SGD generalize?* Noise acts as implicit regularization and finds flatter minima (an empirical/heuristic argument, not a theorem).
- *Adam vs SGD?* Adam adapts per-parameter, converges faster early; SGD+momentum often needs more tuning but can generalize well.
- *Why AdamW?* L2 added to the gradient gets rescaled by Adam's denominator; decoupling applies true weight decay.

## Exercises
Implement RMSProp; find the lr at which GD diverges on the quadratic above; add warmup to Adam and test on a noisy loss.

## Further reading
Ruder (2016) "An overview of gradient descent optimization algorithms"; Kingma & Ba (2014); Loshchilov & Hutter (2017); Boyd & Vandenberghe, *Convex Optimization* (free).
