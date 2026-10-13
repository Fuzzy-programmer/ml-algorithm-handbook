# Chapter 3.5 — Probabilistic Models: Gaussian Mixtures, Hidden Markov Models, Bayesian Networks

[← Roadmap](../README.md) · Prev [3.4](04-xgboost-lightgbm-catboost.md) · Next [4.1 K-Means](../04-unsupervised-learning/01-kmeans.md) · Priority 🟡 · Code: [`probabilistic_models.py`](../code/probabilistic_models.py)

(Placed here because all three are likelihood-based models trained with MLE/EM from Ch 0.2; GMMs reappear in clustering, Ch 4.1–4.2.)

## Part A — Gaussian Mixture Models (GMM)

### Identity & problem
Soft clustering and density estimation: data come from K Gaussian "sources"; learn their means, covariances and mixing weights, and the probability each point belongs to each.

### History
Pearson (1894) fitted two-component mixtures by moments; **EM** (Dempster, Laird & Rubin 1977) made maximum-likelihood fitting practical. K-Means is the hard-assignment, spherical, equal-variance limit of a GMM. Extensions: Bayesian/Dirichlet-process mixtures, mixtures of experts, mixtures of factor analyzers.

### Intuition
Each point has a probability of coming from each cluster; clusters can be elliptical and different sizes.

### Math
`p(x) = Σ_k π_k 𝒩(x|μ_k, Σ_k)`. Log-likelihood `Σ_i log Σ_k π_k 𝒩(x_i|μ_k,Σ_k)` has no closed form.
**EM:** E-step responsibilities `r_ik = π_k 𝒩(x_i|μ_k,Σ_k) / Σ_j π_j 𝒩(x_i|μ_j,Σ_j)`; M-step `N_k=Σ_i r_ik`, `π_k=N_k/n`, `μ_k = Σ r_ik x_i / N_k`, `Σ_k = Σ r_ik (x_i−μ_k)(x_i−μ_k)ᵀ / N_k`. Likelihood never decreases; converges to a local optimum (initialize with K-Means, multiple restarts).
**Worked example (1-D, one E-step).** Two components: 𝒩(0,1) and 𝒩(4,1), equal weights. Point x=1: densities ∝ exp(−½·1²)=0.607 and exp(−½·9)=0.011 ⇒ responsibilities 0.607/(0.618)=**0.982** vs **0.018**. A point at x=2 (midpoint) gets 0.5/0.5.
Model selection: **BIC** `= p·ln n − 2 lnL̂` (lower is better); covariance types: full, tied, diag, spherical. Singularities: a component collapsing onto one point gives infinite likelihood ⇒ `reg_covar`.

### Training, cost
Per iteration `O(n·K·d²)` (full covariance, plus `d³` for inversions). Prediction `O(K·d²)`. Memory `O(K·d²)`.

### When to use / not
Use: overlapping/elliptical clusters, need soft probabilities, density estimation, anomaly scores (low `score_samples`), generative sampling, speaker/background modeling. Avoid: non-Gaussian/non-convex clusters (DBSCAN/spectral), high dimensions with full covariance (parameter explosion; use diag/tied or PCA first), huge K without BIC check.

### Hyperparameters
`n_components`, `covariance_type`, `n_init`, `init_params` (kmeans / k-means++ / random), `reg_covar`, `max_iter`, `tol`.

### Implementation & results (measured)
Two elongated, overlapping Gaussians (correlation ~0.85), 600 points:
| Method | Adjusted Rand Index vs true labels |
|---|---|
| K-Means | 0.236 |
| **GMM (scratch EM)** | **0.980** |
| GMM (sklearn) | 0.980 |
Average log-likelihood per point: scratch −4.213 = sklearn −4.213 (the from-scratch EM matches). K-Means fails because it assumes spherical clusters of equal spread. BIC over K=1…6 for 3 true blobs: 6531, 5007, **4782 (K=3)**, 4808, 4838, 4875 — minimum at the truth.
```python
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(n_components=3, covariance_type="full", n_init=5, random_state=0).fit(X)
resp = gm.predict_proba(X); bic = gm.bic(X); anomaly_score = gm.score_samples(X)
```
Scratch EM: see `em_gmm` in the code file.

---

## Part B — Hidden Markov Models (HMM)

### Identity & problem
Sequence model with a hidden discrete state evolving as a Markov chain and observations emitted from the state. Used for speech (classic), part-of-speech tagging, gene finding, regime detection, activity recognition.

### History
Markov chains (1906); Baum–Welch/EM training (Baum et al., late 1960s–70s); Viterbi (1967); Rabiner's 1989 tutorial made HMMs the workhorse of speech recognition until neural sequence models (Ch 5.3–5.4) replaced them. Successors: CRFs (Lafferty et al. 2001), RNN/LSTM, Transformers; state-space models/Kalman filters are the continuous-state analogue.

### Math
Parameters: initial `π`, transitions `A_ij=P(s_{t+1}=j|s_t=i)`, emissions `B_i(o)`. Markov: `P(s_t|s_{1:t−1}) = P(s_t|s_{t−1})`; output independence.
Three problems: **evaluation** (forward algorithm, `O(T·K²)`), **decoding** (Viterbi: most likely state sequence), **learning** (Baum–Welch EM).
**Worked example.** States {Rainy, Sunny}; π=(0.6, 0.4); transitions Rainy→(0.7 Rainy, 0.3 Sunny), Sunny→(0.4, 0.6); emissions Rainy: walk 0.1, shop 0.4, clean 0.5; Sunny: walk 0.6, shop 0.3, clean 0.1. Observations: walk, shop, clean.
Forward: α₁ = (0.6·0.1, 0.4·0.6) = (0.06, 0.24); α₂ = ((0.06·0.7+0.24·0.4)·0.4, (0.06·0.3+0.24·0.6)·0.3) = (0.0552, 0.0486); α₃ = (0.02904, 0.00457) ⇒ **P(observations) = 0.0336**.
Viterbi: most likely hidden path **Sunny → Rainy → Rainy** (prob 0.01344), confirmed by brute force over all 8 paths. (Both are in `probabilistic_models.py`.)

### Cost, when to use
Forward/Viterbi `O(T·K²)`. Use for modest discrete-state sequences, interpretable regime switching, small data. Avoid for long-range dependencies, rich observations (use RNN/Transformer), or continuous latent dynamics (Kalman).
Library: `hmmlearn` (`GaussianHMM`, `CategoricalHMM`); `pomegranate`.

---

## Part C — Bayesian Networks

### Identity & problem
Directed acyclic graph encoding a factorization of a joint distribution: `P(X₁…X_n) = Π P(X_i | Parents(X_i))`. Gives interpretable causal-ish structure, handling missing evidence and reasoning under uncertainty (diagnosis, risk).

### History
Pearl (1980s; *Probabilistic Reasoning in Intelligent Systems*, 1988); belief propagation, junction-tree algorithm (Lauritzen & Spiegelhalter 1988); structure learning (PC algorithm, score-based search such as BIC/BDeu; Spirtes–Glymour–Scheines). Naive Bayes (Ch 2.3) is the simplest Bayesian network; HMMs are dynamic Bayesian networks; modern causal inference (Ch 6.4) builds on Pearl's graphs.

### Math & worked example (inference by enumeration)
Network: Rain → Sprinkler, Rain → WetGrass, Sprinkler → WetGrass. `P(R)=0.2`; `P(S|R)=0.01`, `P(S|¬R)=0.4`; `P(W|R,S)=0.99`, `P(W|R,¬S)=0.8`, `P(W|¬R,S)=0.9`, `P(W|¬R,¬S)=0`.
`P(W=1) = 0.4484`. `P(R|W) = 0.3577` (up from the 0.20 prior). **Explaining away:** adding evidence that the sprinkler ran gives `P(R|W,S) = 0.0068`. The big drop is partly built into this CPT (the sprinkler almost never runs when it rains); the qualitative lesson is that once one cause is known, the other becomes less probable.
Inference: exact (variable elimination, junction tree; NP-hard in general), approximate (sampling, variational). Learning parameters: MLE/Bayesian counts; structure: constraint- or score-based.

### When to use / not
Use: small–medium problems with expert knowledge, need explanations and what-if reasoning, missing data. Avoid: large raw-data prediction tasks (boosting/NNs win), automatic structure discovery claims of causality without assumptions. Tools: `pgmpy`, `pomegranate`, `bnlearn` (R).

## Comparison
GMM = unsupervised density/clustering; HMM = sequences; Bayes nets = structured reasoning. All are generative (Ch 6.1 contrasts with discriminative models) and trade accuracy for interpretability and principled uncertainty.

## Interview Q&A
- *EM intuition?* Alternate guessing the hidden assignments (E) and refitting parameters given them (M).
- *GMM vs K-Means?* Soft vs hard, full covariance vs spherical.
- *Forward vs Viterbi?* Sum over paths (likelihood) vs max over paths (best sequence).
- *Explaining away?* Observing one cause lowers belief in competing causes of a shared effect.

## Exercises
Fit a GMM on 2-D Old Faithful-style data and choose K by BIC; run Baum–Welch with `hmmlearn` on a synthetic regime-switching series; compute P(S|W) in the sprinkler net by hand.

## Further reading
Dempster, Laird, Rubin (1977); Rabiner (1989) "A tutorial on HMMs and selected applications in speech recognition"; Pearl (1988); Koller & Friedman, *Probabilistic Graphical Models*; Bishop PRML Ch 9, 13, 8.
