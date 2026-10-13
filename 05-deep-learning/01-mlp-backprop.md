# Chapter 5.1 — Neural Networks: Perceptron → MLP → Backpropagation

[← Roadmap](../README.md) · Prev [4.7](../04-unsupervised-learning/07-feature-selection.md) · Next [5.2 CNNs](02-cnn-lineage.md) · Priority 🔴 · Code: [`deep_learning_numpy.py`](../code/deep_learning_numpy.py)

> **Code note:** NumPy code in this phase was executed and its numbers are quoted. PyTorch snippets are shown for practice but were **not executed** in the authoring environment (no PyTorch installed); treat them as templates and verify before use.

## 1. Identity & problem
A multilayer perceptron (MLP) stacks linear maps with nonlinear activations to learn nonlinear functions from data. Backpropagation computes all gradients efficiently so gradient descent (Ch 0.3) can train it. Foundation of all deep learning.

## 2. History
McCulloch–Pitts neuron (1943) → Rosenblatt perceptron (1958) → Minsky & Papert (1969) show single-layer limits (XOR) → reverse-mode differentiation (Linnainmaa 1970), applied to nets by Werbos (1974) → **Rumelhart, Hinton & Williams (1986)** popularize backprop with learned internal representations → universal approximation (Cybenko 1989; Hornik 1991) → second winter (SVMs/boosting dominate small data) → renaissance: GPUs, ReLU (Nair & Hinton 2010), better init (Glorot 2010; He 2015), dropout (2014), BatchNorm (2015), Adam (2014), big data (ImageNet) → deep learning.

## 3. Intuition
Each layer re-describes the input in a more useful way; the last layer is just logistic/linear regression on those learned features. Backprop answers "how much did each weight contribute to the error?" by passing blame backwards with the chain rule.

## 4. Math
Layer: `a⁰=x`, `z^l = W^l a^{l−1} + b^l`, `a^l = φ(z^l)`. Output loss `L` (cross-entropy for classification with softmax: `p=softmax(z^L)`, `L=−log p_y`).
**Backprop:** output error `δ^L = p − y_onehot`; propagate `δ^{l} = (W^{l+1})ᵀ δ^{l+1} ⊙ φ'(z^l)`; gradients `∂L/∂W^l = δ^l (a^{l−1})ᵀ`, `∂L/∂b^l = δ^l`.
Activations: sigmoid `1/(1+e^{−z})` (saturates; `σ'≤0.25`), tanh, **ReLU** `max(0,z)` (default; dead-unit risk), Leaky ReLU, GELU/SiLU (Transformers). Universal approximation says a wide enough single hidden layer *can* approximate any continuous function; depth makes it efficient.
**Worked example.** `x=1`, `h=ReLU(w₁x)`, `ŷ=w₂h`, target `y=1`, `w₁=0.5`, `w₂=−1`, `L=(ŷ−y)²`.
Forward: `h=0.5`, `ŷ=−0.5`, `L=2.25`. Backward: `dL/dŷ=2(ŷ−y)=−3`; `dL/dw₂ = −3·h = −1.5`; `dL/dh = −3·w₂ = 3`; ReLU' = 1 (z>0) ⇒ `dL/dw₁ = 3·x = 3`. Step lr=0.1: `w₂=−0.85`, `w₁=0.2`. New forward: `h=0.2`, `ŷ=−0.17`, `L=1.369` (down from 2.25).
**Vanishing gradients:** a product of many sigmoid derivatives shrinks geometrically — `0.25¹⁰ ≈ 9.5×10⁻⁷`, `0.25³⁰ ≈ 8.7×10⁻¹⁹` — which is why ReLU, good initialization, normalization and residual connections matter. **Initialization:** He `N(0, 2/fan_in)` for ReLU, Glorot for tanh.
Assumptions: differentiable model/loss; i.i.d. data; no convexity — local minima/saddles exist but wide nets train well in practice.

## 5. Training
Mini-batch SGD/AdamW; epochs; learning-rate schedule; weight decay; dropout; early stopping; batch/layer norm; gradient clipping. Cost: forward+backward ≈ 3× forward FLOPs; parameters `Σ(fan_in·fan_out + fan_out)`. Memory: activations stored for backward pass (checkpointing trades compute for memory).

## 6. Evolution & trade-offs
| Step | Gain | Cost |
|---|---|---|
| Perceptron → MLP | Non-linear boundaries (XOR) | Non-convex training |
| Sigmoid/tanh → ReLU/GELU | Faster, no saturation | Dead units; need init care |
| + Dropout/BatchNorm/LayerNorm/Residuals | Regularize and train deeper | Extra hyperparameters |
| MLP → CNN/RNN/Transformer | Exploit structure (grid/sequence/set) | Architecture-specific |
For plain tabular data, tuned gradient boosting usually matches or beats MLPs (Grinsztajn et al. 2022; Shwartz-Ziv & Armon 2022).

## 7. When to use / not
Use: large data, unstructured inputs via embeddings, multi-output tasks, differentiable pipelines, a head on pretrained features. Avoid: small tabular data (use GBMs/linear), when interpretability/latency/simplicity dominate, no GPU budget for big models.

## 8. Hyperparameters
Depth/width, activation, learning rate (most important), batch size, optimizer, weight decay, dropout rate, init, normalization, epochs/early stopping, LR schedule. Typical order: LR → architecture size → regularization.

## 9. Implementation
From scratch (full MLP with softmax cross-entropy and backprop is in `deep_learning_numpy.py`, class `MLP`):
```python
def grads(self, X, Y):
    P = self.forward(X); n = len(X); d = P.copy(); d[np.arange(n), Y] -= 1; d /= n      # dL/dz_out = p - onehot
    for i in reversed(range(len(self.W))):
        gW.insert(0, self.as_[i].T @ d); gb.insert(0, d.sum(0))
        if i > 0: d = (d @ self.W[i].T) * self._df(self.zs[i-1])                         # chain rule through activation
```
PyTorch template (not executed here):
```python
import torch, torch.nn as nn
net = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 10))
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2); lossf = nn.CrossEntropyLoss()
for xb, yb in loader: opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
```
**Always gradient-check a new layer:** compare backprop with finite differences. Measured: max |analytic − numeric| = **2.5×10⁻¹⁰** (✓ correct).

## 10. Evaluation & results (measured)
- **XOR:** network with no hidden layer → accuracy **0.5** (cannot learn); one hidden layer (4 tanh units) → **1.0**.
- **Digits (8×8, 1,347 train / 450 test):** scratch NumPy MLP (64-64-10, 4,810 parameters, 60 epochs) test accuracy **0.973**; scikit-learn MLP 0.973; multinomial logistic regression **0.969**. On this small, nearly linear problem the net adds little over the linear baseline — a recurring lesson: always report the linear baseline.
Diagnostics: training/validation loss curves, gradient norms, activation statistics, learning-rate range test, overfit a tiny batch first.

## 11. Case study
Tabular churn with embeddings for categorical IDs: compare logistic, LightGBM and an MLP with embeddings on a time-based split; typically the GBM wins or ties on moderate data, the MLP helps with very high-cardinality IDs and multi-task learning.

## 12. Comparison
vs logistic regression: nonlinear but needs tuning and more data. vs trees/GBMs: better for unstructured data, worse default on tabular. vs SVM: scale better with n, no convex guarantee.

## Interview Q&A
- *Why nonlinear activations?* Without them any depth collapses to one linear map.
- *Explain backprop in two sentences.* Cache forward activations; apply the chain rule from the loss backwards, reusing the downstream gradient at every layer.
- *Why ReLU over sigmoid?* No saturation for positive inputs, cheaper, mitigates vanishing gradients.
- *What does BatchNorm do?* Normalizes activations per batch, stabilizing and speeding training (the exact mechanism is debated).
- *Overfitting remedies?* More data/augmentation, weight decay, dropout, early stopping, smaller model.

## Exercises
Re-derive the softmax cross-entropy gradient `p − y`; implement momentum/Adam in the scratch MLP; train with sigmoid vs ReLU at depth 10 and compare gradient norms.

## Further reading
Rumelhart, Hinton & Williams (1986) *Nature*; Goodfellow, Bengio & Courville, *Deep Learning* (Ch 6–8); Karpathy "micrograd" and "The spelled-out intro to neural networks and backpropagation"; He et al. (2015) init; Glorot & Bengio (2010).
