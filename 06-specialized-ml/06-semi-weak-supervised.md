# Chapter 6.6 — Semi-Supervised Learning and Weak Supervision

[← Roadmap](../README.md) · Prev [6.5](05-survival-analysis.md) · Next [7 Comparisons](../07-comparisons/README.md) · Priority 🟡 · Code: [`causal_survival_semisup.py`](../code/causal_survival_semisup.py)

## 1. Identity & problem
Labels are expensive. **Semi-supervised learning (SSL)** uses a small labeled set plus abundant unlabeled data. **Weak supervision** replaces hand labels with noisy, programmatic or partial labels (heuristics, knowledge bases, crowd, other models). Related: self-supervised pretraining (Ch 5.7), active learning, transfer learning.

## 2. History and methods
| Approach | Idea | Landmarks |
|---|---|---|
| **Self-training / pseudo-labeling** | Train → predict unlabeled → add confident predictions → retrain | Scudder (1965); Yarowsky (1995); Lee (2013) for deep nets |
| **Co-training** | Two views teach each other | Blum & Mitchell (1998) |
| **Graph / label propagation** | Labels diffuse over a similarity graph | Zhu & Ghahramani (2002); Label Spreading (Zhou et al. 2004) |
| **Generative / EM** | Model p(x,y) with unlabeled x via EM | Nigam et al. (2000) |
| **Consistency regularization** | Predictions invariant to perturbation | Π-model / Temporal ensembling (Laine & Aila 2016), Mean Teacher (Tarvainen 2017), UDA (Xie 2019) |
| **Hybrid** | Pseudo-labels + consistency + strong augmentation | MixMatch (2019), **FixMatch** (Sohn et al. 2020) |
| **Weak supervision** | Label functions combined by a generative label model | Distant supervision (Mintz 2009), **Snorkel/data programming** (Ratner et al. 2016/2017) |
| **Noisy-label learning** | Find/handle label errors | Confident learning (Northcutt et al. 2021), co-teaching, robust losses |
| **Active learning** | Choose which points to label | Uncertainty / query-by-committee sampling (Settles) |
Modern practice: **pretrain self-supervised/foundation model → fine-tune on few labels**, which often dominates classic SSL; zero/few-shot LLM labeling is now a common weak-supervision source (with verification).

## 3. Intuition
Unlabeled data reveal the *shape* of the data (clusters, manifolds). If decision boundaries should lie in low-density gaps (cluster/smoothness assumption), unlabeled points tell you where the gaps are.

## 4. Math
Assumptions (without them SSL can hurt): **smoothness** (nearby points share labels), **cluster/low-density separation**, **manifold** (data lie on low-dim manifold).
Pseudo-labeling: `L = L_sup + λ(t)·Σ_{u: max p(y|u) ≥ τ} CE(p(y|u), argmax p(y|u))`.
Consistency: `L_u = ‖f(Aug₁(u)) − f(Aug₂(u))‖²`. FixMatch: pseudo-label from a *weak* augmentation (if confidence ≥ τ) supervises prediction on a *strong* augmentation.
Label propagation: build affinity `W`, normalized `S=D^{-1/2}WD^{-1/2}`, iterate `F ← αSF + (1−α)Y` to a fixed point `F* = (1−α)(I−αS)^{-1}Y`.
**Worked mini-example (label propagation intuition).** Chain A—B—C with A labeled +, C labeled −, B unlabeled equidistant: B receives equal pulls from both ⇒ ≈0 (uncertain). Add D connected only to A: D inherits + through A. Structure carries the labels.
**Weak supervision label model:** each labeling function (LF) votes/abstains; a generative model estimates LF accuracies and correlations without ground truth and outputs probabilistic labels to train a discriminative model.

## 5. Training & cost
Typically 1–3× the supervised training cost (extra forward passes, graph construction `O(n²)` or kNN `O(n log n)`). Label spreading needs a graph over all points (memory!). Weak supervision shifts cost from labeling to writing/debugging labeling functions.

## 6. When to use / not
Use: many unlabeled, few labeled; labels expensive/delayed; class structure is cluster-like; domain knowledge easily encoded as rules. Avoid: when unlabeled data come from a different distribution; when classes overlap heavily (assumptions fail); when labels are cheap (just label more); when pseudo-label errors would reinforce confirmation bias; without a clean labeled validation set.

## 7. Hyperparameters
Confidence threshold τ, unlabeled/labeled ratio, loss weight ramp-up, augmentations, number of neighbors/kernel width (graph), α (propagation), iterations, label-model settings, LF coverage/accuracy thresholds.

## 8. Implementation
```python
from sklearn.semi_supervised import SelfTrainingClassifier, LabelSpreading
y_partial = np.full(len(y), -1); y_partial[labeled_idx] = y[labeled_idx]           # -1 means unlabeled
st = SelfTrainingClassifier(SVC(probability=True), threshold=0.8).fit(X, y_partial)
ls = LabelSpreading(kernel="knn", n_neighbors=10).fit(X, y_partial)
```
Weak supervision: `snorkel` library (labeling functions + `LabelModel`); `cleanlab` for label-error detection.

## 9. Evaluation & results (measured)
Digits (1,300 training images, **30 labels = 3 per class**, 497 test images):
| Method | Test accuracy |
|---|---|
| Supervised SVC, 30 labels only | 0.785 |
| Self-training SVC (threshold 0.8) | **0.785** (no gain) |
| **Label Spreading (kNN graph, k=10)** | **0.899** |
| Upper bound: SVC with all 1,300 labels | 0.986 |
Label propagation recovers ~57% of the gap between 30 labels and full supervision using only the data's geometry (digits form tight clusters—assumption holds). **Self-training gave no improvement here**: with few, possibly mislabeled confident predictions it mostly reinforces the initial classifier — the classic confirmation-bias failure; it needs a reasonably accurate base model and careful thresholds. Always evaluate on a held-out *labeled* set and compare with the supervised-only baseline.

## 10. Case study (template)
Document classification with 200 labeled and 50k unlabeled texts: baseline TF-IDF+logistic on labeled; add embeddings from a pretrained sentence encoder (Ch 5.7) → logistic (often the largest jump); then pseudo-label with high confidence, add weak labeling functions (keywords, metadata), review disagreement with cleanlab, track accuracy on a small gold set, and watch per-class drift.

## 11. Comparison
Self-supervised pretraining + fine-tuning (most scalable) > classic SSL (consistency/pseudo-label) > label propagation (small data, strong cluster structure) > pure weak supervision (fast, noisy). Active learning complements all: spend the labeling budget where the model is uncertain.

## Interview Q&A
- *Why can SSL hurt?* If the cluster/smoothness assumption is false or unlabeled data are off-distribution, pseudo-labels inject errors.
- *What is confirmation bias in pseudo-labeling?* The model reinforces its own mistakes; mitigate with thresholds, ramp-up, ensembling, strong augmentation.
- *Weak vs semi-supervised?* Weak: noisy/programmatic labels for many points; semi: few true labels plus unlabeled data.
- *How do you validate?* A small trusted labeled validation set; never trust the pseudo-labels themselves.

## Exercises
Vary the number of labels (10, 30, 100, 300) and plot self-training vs label spreading vs supervised; implement FixMatch's weak/strong augmentation idea on digits; write 5 labeling functions for a text task and measure coverage and conflict.

## Further reading
Chapelle, Schölkopf & Zien, *Semi-Supervised Learning*; Zhu & Goldberg, *Introduction to Semi-Supervised Learning*; Lee (2013); Sohn et al. (2020) FixMatch; Ratner et al. (2017) Snorkel; Northcutt, Jiang & Chuang (2021) "Confident learning"; Settles, *Active Learning*.
