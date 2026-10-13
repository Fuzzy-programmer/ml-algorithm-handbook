# Chapter 6.1 — Generative vs. Discriminative Modeling

[← Section index](README.md) · Next [6.2 Language models → LLMs](02-language-models-llms.md) · Priority 🔴 · Code: [`classification_family.py`](../code/classification_family.py), [`genai_demos.py`](../code/genai_demos.py)

## 1. The distinction
- **Discriminative** models learn `p(y|x)` (or a decision boundary): *given the input, what is the label?* Logistic regression, SVM, trees/GBMs, most neural classifiers.
- **Generative** models learn `p(x)` or `p(x,y)=p(x|y)p(y)`: *how is the data produced?* They can score likelihood, sample new data, handle missing inputs, and classify via Bayes' rule. Naive Bayes, LDA/QDA, GMM, HMM, VAEs, GANs, diffusion models, autoregressive language models.
Terminology caution: "generative AI" (models that produce text/images/audio) overlaps with, but is not identical to, "generative classifiers" (modeling `p(x|y)` to classify). LLMs are generative models of `p(text)`, yet are used discriminatively too (prompted classification).

## 2. Math
Bayes: `p(y|x) = p(x|y)p(y)/p(x)`. A generative classifier models the numerator; a discriminative one models the left side directly. For Gaussian class-conditionals with shared covariance, the generative posterior is exactly logistic in `x` (Ch 2.4) — the same *form* as logistic regression, different *fitting* (joint vs conditional likelihood).
Generative model families by how they represent `p(x)`:
| Family | Representation | Likelihood | Examples |
|---|---|---|---|
| Autoregressive | `Πp(x_t|x_{<t})` | Exact | n-grams, RNN/Transformer LMs, PixelCNN |
| Latent variable | `∫p(x|z)p(z)dz` | Bound (ELBO) | VAE, GMM, HMM |
| Implicit | Sampler only | None | GAN |
| Score/diffusion | Learned noise/score | Bound/ODE | DDPM, score SDE (Ch 6.7) |
| Flow | Invertible transform | Exact | RealNVP, Glow |
**Worked example (generative classification).** Priors `p(spam)=0.4`; word likelihoods give joint scores 0.010 (spam) vs 0.012 (ham) → posterior spam 0.4545 (Ch 2.3). The generative model also assigns `p(x)=0.022` to the message — a score unavailable to a purely discriminative model.

## 3. When each wins
| Situation | Prefer |
|---|---|
| Plenty of labeled data, prediction accuracy is the goal | Discriminative |
| Very small labeled sets (Ng & Jordan 2001: NB approaches its asymptotic error faster) | Generative may win early |
| Missing features at test time, semi-supervised learning, anomaly detection (low `p(x)`) | Generative |
| Need to sample/synthesize data, simulate, augment | Generative |
| Language, image, audio, code generation | Generative (autoregressive/diffusion) |
| Calibrated class probabilities on tabular data | Usually discriminative (logistic/GBM) |

## 4. Evidence from this handbook (measured)
- Breast cancer 5-fold CV accuracy: Gaussian NB **0.939**, LDA 0.954, QDA 0.956 vs Logistic **0.979**, RBF-SVM 0.977 (Ch 2.x) — discriminative models ahead with ~450 training points per fold.
- 20-newsgroups (4 classes, test accuracy): MultinomialNB **0.842** vs TF-IDF+Logistic **0.844** vs LinearSVC 0.858 — a tie in accuracy, NB far cheaper.
- Anomaly detection (Ch 4.5) uses generative-style density/isolation scoring because anomalies lack labels.
- Autoencoder vs PCA (Ch 5.5): the small nonlinear model did not beat the linear one with limited data — generative flexibility needs data.

## 5. Cost and risk
Generative modeling of high-dimensional data is harder than classification (it must model *everything* about `x`, including irrelevant detail). Generated data can be wrong yet fluent (hallucination), memorize training data, and carry biases; evaluation is subjective and costly (Ch 6.11). Using a generative model as a *classifier* may underperform a dedicated discriminative model.

## 6. Interview Q&A
- *Generative vs discriminative?* Model of how data arise vs model of the boundary/label given data.
- *Can a generative model classify?* Yes via Bayes' rule; may be less accurate with large data but more sample-efficient with small data.
- *Is an LLM generative or discriminative?* Generative autoregressive model of text; can be applied discriminatively by prompting or adding a head.
- *Why does Naive Bayes lose to logistic regression with more data?* Its independence assumption introduces asymptotic bias.

## Exercises
Fit NB and logistic regression on increasing subsets of 20-newsgroups and plot learning curves; derive the logistic form of LDA's posterior; score held-out points under a GMM and use low-density points as anomalies.

## Further reading
Ng & Jordan (2001) "On discriminative vs. generative classifiers"; Bishop PRML Ch 4; Murphy PML Ch 20–22; Goodfellow et al., *Deep Learning* Ch 20; Lilian Weng's generative-model posts.
