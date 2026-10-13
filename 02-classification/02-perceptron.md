# Chapter 2.2 — The Perceptron

[← Roadmap](../README.md) · Prev [2.1](01-logistic-regression.md) · Next [2.3 Naive Bayes](03-naive-bayes.md) · Priority 🟡 · Code: [`classification_scratch.py`](../code/classification_scratch.py)

## 1. Identity & problem
Online linear classifier with a hard threshold. Historic importance: first trainable "neuron," ancestor of neural networks (not of logistic regression, which came from statistics).

## 2. History
McCulloch–Pitts neuron (1943) → **Rosenblatt's perceptron** (1957 hardware at Cornell; 1958 paper) → convergence theorem (Novikoff 1962) → **Minsky & Papert, *Perceptrons* (1969)** proved a single layer cannot learn XOR/non-linearly-separable functions, contributing to the first "AI winter" for connectionism → multi-layer nets + backprop (Rumelhart, Hinton, Williams 1986; Ch 5.1) resolve this. Related: Widrow–Hoff ADALINE (1960), margin ideas → SVM (Ch 2.6), voted/averaged perceptron (Freund & Schapire 1999), structured perceptron (Collins 2002).

## 3. Intuition
Keep a line. For every misclassified point, nudge the line toward that point's correct side. Stop when nothing is misclassified.

## 4. Math
Prediction `ŷ = sign(wᵀx+b)`, labels `t∈{−1,+1}`. On mistake (`t(wᵀx+b) ≤ 0`): `w ← w + t·x`, `b ← b + t`.
**Convergence (Novikoff):** if data are linearly separable with margin γ and `‖x‖≤R`, the number of mistakes is at most `(R/γ)²` — independent of n and dimension. If not separable, it never converges (cycles).
Equivalent to SGD on the loss `max(0, −t·(wᵀx))` with lr 1.

**Worked example (AND gate).** Points (0,0),(0,1),(1,0) → −1, (1,1) → +1; start `w=(0,0), b=0`; update when `t·score ≤ 0`. Running it: converges after 9 passes (the last is a clean pass) to `w=(3,2), b=−4`. Check: (1,1)→3+2−4=1>0 ✓; (1,0)→−1, (0,1)→−2, (0,0)→−4 ✓. **XOR** (labels +,−,−,+ on the corners) is not linearly separable ⇒ never converges.

## 5. Training
Cycle through data, update on mistakes; shuffle each epoch. Cost `O(n·p)` per epoch, memory `O(p)`. Stop when an epoch has no mistakes or at `max_iter`. Not scale-invariant in speed (mistake bound depends on R), so standardize.

## 6. Evolution & trade-offs
| Step | Gain | Cost |
|---|---|---|
| Perceptron → logistic regression | Probabilities, convex smooth loss, converges on non-separable data | Needs gradient/loss |
| Perceptron → averaged/voted | Better generalization, reduces order dependence | Slight overhead |
| Perceptron → SVM | Max-margin solution, kernels | Optimization cost |
| Single layer → MLP | Non-linear boundaries | Needs backprop, non-convex |

## 7. When to use / not
Use: teaching, streaming/online learning with cheap updates, very large sparse problems (e.g. structured prediction in NLP with averaged perceptron). Avoid: non-separable data, when you need probabilities or margins — use logistic regression or SVM.

## 8. Strengths / weaknesses
+ Trivial, online, mistake-driven (updates only on errors), fast. − Result depends on data order, no margin maximization, no probabilities, no convergence on non-separable data.

## 9. Hyperparameters
`max_iter`, `eta0` (scaling only matters with a nonzero initial w), `penalty`/`alpha` (regularization), `shuffle`, `early_stopping`, `class_weight`, averaging.

## 10. Implementation
```python
class Perceptron_:
    def fit(self, X, y, epochs=50):
        s = 2*y - 1; A = np.c_[np.ones(len(X)), X]; w = np.zeros(A.shape[1])
        for e in range(epochs):
            errs = 0
            for a, t in zip(A, s):
                if t*(a@w) <= 0: w += t*a; errs += 1
            if errs == 0: break
        self.w = w; return self
```
Library: `sklearn.linear_model.Perceptron` (SGD-based) or `SGDClassifier(loss="perceptron")`.

## 11. Evaluation & results
Breast cancer, scaled, 5-fold CV: accuracy **0.967**, AUC 0.993 (logistic 0.979 / 0.995) — close, as expected on near-separable data. From-scratch version: test acc 0.944 vs logistic 0.958 on one split (single-split difference, not a significant gap). AUC for a perceptron uses its raw score, which is not a probability.

## 12. Comparison
Learns the same family of boundaries as logistic/linear SVM; differs only in loss (0-1-like mistake loss vs log vs hinge). Prefer logistic/SVM in production.

## Interview Q&A
- *Why can't a perceptron learn XOR?* XOR is not linearly separable; one hyperplane can't split it. Add a hidden layer or features like `x₁x₂`.
- *Guarantee?* Finite mistakes if separable, bound `(R/γ)²`.
- *Difference to logistic regression?* Hard threshold, mistake-driven updates, no probabilities.

## Exercises
Show the mistake bound empirically; implement the averaged perceptron; map XOR with feature `x₁x₂` and re-train.

## Further reading
Rosenblatt (1958) *Psychological Review*; Minsky & Papert (1969); Freund & Schapire (1999) "Large margin classification using the perceptron algorithm."
