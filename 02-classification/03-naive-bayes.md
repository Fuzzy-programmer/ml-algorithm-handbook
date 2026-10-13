# Chapter 2.3 — Naive Bayes

[← Roadmap](../README.md) · Prev [2.2](02-perceptron.md) · Next [2.4 LDA/QDA](04-lda-qda.md) · Priority 🟠 · Code: [`classification_scratch.py`](../code/classification_scratch.py), [`classification_family.py`](../code/classification_family.py)

## 1. Identity & problem
Generative probabilistic classifier using Bayes' rule with a "naive" conditional-independence assumption. Classic for text (spam, topic), also fast baselines for high-dimensional data.

## 2. History
Bayes' theorem (1763; Laplace); used for text categorization from the early 1960s (Maron 1961); popularised by spam filtering (Graham 2002 "A Plan for Spam"); Domingos & Pazzani (1997) explained why NB classifies well even when independence is false (it needs correct *ranking* of class posteriors, not correct probabilities). Direct predecessor idea: full Bayes classifier (intractable in many dimensions) → NB factorization → TAN / Bayesian networks (Ch 3.5) relax independence.

## 3. Intuition
Score each class by prior × product of how typical each feature is for that class. Pick the highest.
Why "naive": features are assumed independent *given the class*, so the joint likelihood factors into one-dimensional pieces — easy to estimate even with few samples in many dimensions.

## 4. Math
`P(c|x) ∝ P(c) Π_j P(x_j|c)`; prediction `argmax_c [log P(c) + Σ_j log P(x_j|c)]` (logs avoid underflow).
Variants: **Gaussian** (`x_j|c ~ 𝒩(μ_cj, σ²_cj)`), **Multinomial** (word counts: `θ_cj = (N_cj+α)/(N_c+α·V)`), **Bernoulli** (word presence/absence), **Complement NB** (imbalanced text), **Categorical**.
**Laplace/Lidstone smoothing** α>0 prevents zero probability for unseen words (equivalent to a Dirichlet prior, MAP).

**Worked example (spam).** `P(spam)=0.4`. "free": appears in 50% of spam, 5% of ham; "meeting": 5% of spam, 40% of ham. Email contains both.
Spam score `0.4×0.5×0.05 = 0.010`; ham score `0.6×0.05×0.40 = 0.012`. Posterior spam = `0.010/0.022 = 0.4545` → classified **ham**, with probability only ~45% spam. Note how two moderately strong cues cancel; if "meeting" were absent from the model, spam would win.
Independence assumption consequence: two highly correlated words both counted ⇒ evidence double-counted ⇒ overconfident probabilities (often near 0/1) even when ranking is fine.

## 5. Training & prediction
Training = counting (one pass), `O(n·p)`; closed-form MLE; no iterations, trivially online/incremental (`partial_fit`) and parallel. Prediction `O(p·C)`. Memory `O(p·C)`.

## 6. Evolution
| Step | Improvement | Cost |
|---|---|---|
| Full Bayes → NB | Tractable | Independence assumption |
| NB → TAN / Bayesian networks | Model dependencies | Structure learning |
| NB → logistic regression | Discriminative; same linear form for NB in log-space but weights fit jointly | Slower to train, better with large n |
| NB features → TF-IDF + linear / transformers | Context, long-range semantics | Heavier |
Ng & Jordan (2001): NB reaches its (higher) asymptotic error faster than logistic regression, so it can win on very small data while logistic wins as n grows.

## 7. When to use / not
Use: tiny labeled sets, very high-dimensional sparse data, need training in seconds, streaming/online, text baselines, as a sanity benchmark. Avoid: strongly correlated features where calibrated probabilities matter, numeric features with non-Gaussian shapes (Gaussian NB), interactions mattering, final-accuracy-critical tasks where discriminative models are available.

## 8. Strengths / weaknesses
+ Extremely fast, tiny data requirement, handles many features, simple, interpretable log-odds per word. − Independence assumption, poor probability calibration, zero-frequency problem (fix with smoothing), Gaussian assumption.

## 9. Hyperparameters
`alpha` (smoothing; 1 = Laplace; tune on CV, often smaller works better for text), `var_smoothing` (GaussianNB, adds to variances), `priors` / `fit_prior`, `binarize` (Bernoulli), `norm` (Complement).

## 10. Implementation
From scratch (Gaussian NB, see [`classification_scratch.py`](../code/classification_scratch.py)):
```python
mu  = np.array([X[y==k].mean(0) for k in classes]); var = np.array([X[y==k].var(0)+1e-9 for k in classes])
ll  = -0.5*(np.log(2*np.pi*var)[None] + (X[:,None,:]-mu[None])**2/var[None]).sum(2) + np.log(priors)
pred = classes[ll.argmax(1)]
```
Library:
```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
text_clf = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0))
```

## 11. Evaluation & results (measured)
Breast cancer (5-fold CV): GaussianNB acc **0.939**, AUC 0.989 (logistic 0.979 / 0.995) — ranks well, slightly lower accuracy. From-scratch Gaussian NB: 0.909 on one test split. Text, 4 categories of 20-newsgroups (headers/footers/quotes removed, test set): MultinomialNB **0.842**, TF-IDF+Logistic 0.844, TF-IDF+LinearSVC 0.858 — NB is essentially tied with far less training time.

## 12. Case study
Spam/topic filter: CountVectorizer → MultinomialNB trains in milliseconds and is a legitimate production baseline; inspect `feature_log_prob_` to list top words per class for explanation. Recalibrate (isotonic/Platt) if probabilities feed decisions.

## 13. Comparison & decision
Little data/text/speed → NB. More data, want better accuracy/calibration → logistic regression or linear SVM on TF-IDF. Context-heavy language tasks → fine-tuned transformers (Ch 5.4, 6.2).

## Interview Q&A
- *Why does NB work despite false independence?* Classification needs correct arg-max of posteriors; errors from dependence often affect all classes similarly.
- *Zero-frequency problem?* A word unseen in a class zeros the product; fix with additive smoothing.
- *Generative vs discriminative?* NB models P(x|c)P(c); logistic regression models P(c|x) directly.

## Exercises
Reproduce the spam calculation with α-smoothed counts; compare NB and logistic learning curves as n grows; check calibration of NB with a reliability curve.

## Further reading
Domingos & Pazzani (1997) *Machine Learning*; McCallum & Nigam (1998) multinomial vs Bernoulli NB for text; Manning, Raghavan & Schütze, *Introduction to Information Retrieval* (Ch 13, free); sklearn Naive Bayes guide.
