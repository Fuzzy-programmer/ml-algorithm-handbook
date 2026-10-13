# Chapter 0.4 — Bias–Variance, Regularization, Generalization

[← Roadmap](../README.md) · Prev [0.3](03-optimization.md) · Next [0.5](05-evaluation.md) · Priority 🔴

## 1. The core problem
We fit on training data but care about unseen data. **Generalization error = bias² + variance + irreducible noise** (for squared loss).
- **Bias:** error from wrong/too-simple assumptions (underfitting).
- **Variance:** sensitivity to the particular training sample (overfitting).
- **Noise σ²:** can't be removed by any model.

```
error
 │\                       /   test
 │ \_                   __/
 │   \_     sweet     _/
 │     \___ spot ____/
 │  train ──────────────────  (keeps falling)
 └──────────────────────────  model complexity →
   high bias        high variance
```

## 2. Derivation (squared loss)
For `y=f(x)+ε`, `E[ε]=0`, `Var(ε)=σ²`, predictor `f̂` trained on random data:
`E[(y−f̂)²] = (E[f̂]−f)² + Var(f̂) + σ²`.
Cross terms vanish because ε is independent of `f̂`.

## 3. Measured example (noisy sine, n=30, σ²=0.09)
| Polynomial degree | Train MSE | Test MSE | Diagnosis |
|---|---|---|---|
| 1 | 0.370 | 0.327 | Underfit: both high |
| 3 | 0.099 | 0.107 | Near the noise floor (0.09) |
| 9 | 0.040 | 0.143 | Overfit: train ≪ test |
Source: `code/foundations_demos.py`. Train error always falls with complexity; only held-out error shows the U.

## 4. Diagnosing from learning curves
- Train and validation both high and close ⇒ **high bias** → more capacity/features, less regularization.
- Train low, validation high, large gap ⇒ **high variance** → more data, regularize, simpler model, ensembling (bagging), early stopping.
- More data helps variance, not bias.

## 5. Regularization toolbox
| Tool | Mechanism | Where |
|---|---|---|
| L2 (Ridge) | shrinks weights smoothly | linear, NN (weight decay) |
| L1 (Lasso) | sparsity | linear, embedded selection |
| Elastic net | both | correlated features |
| Early stopping | stop before fitting noise | GBMs, NNs |
| Dropout | random unit masking ≈ ensemble | NNs |
| Data augmentation | more (synthetic) variety | vision/audio/text |
| Pruning / depth / min-leaf limits | restrict tree capacity | trees |
| Bagging / averaging | reduce variance | forests, ensembles |
| Larger C⁻¹ / smaller C | margin softness | SVM |

## 6. Modern nuance
Large neural nets can interpolate training data and still generalize, and test error can show **double descent** (error falls, rises near the interpolation threshold, then falls again; Belkin et al. 2019). The classic U-curve is a good guide for classical models, not a law for overparameterized ones.

## 7. Practical rules
Start simple, add capacity until validation stops improving, then regularize · tune regularization with CV, never on the test set · compare to a baseline · variance is worst when n/p is small.

## Interview Q&A
- *High train accuracy, low test accuracy?* Variance: regularize, more data, simplify.
- *Why does bagging reduce variance but not bias?* Averaging B i.i.d. predictions with variance σ² gives σ²/B; the mean (bias) is unchanged.
- *Why is ridge biased yet often better?* It trades a little bias for a large variance drop (MSE = bias² + var).

## Exercises
Reproduce the table with 10 different training sets and estimate bias² and variance separately; plot learning curves with `sklearn.model_selection.learning_curve`.

## Further reading
ESL Ch 7; Geman, Bienenstock & Doursat (1992); Belkin et al. (2019) "Reconciling modern ML practice and the bias–variance trade-off."
