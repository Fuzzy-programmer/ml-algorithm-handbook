# Section 7 — Comparisons, Decision Matrices and Benchmarks

[← Roadmap](../README.md) · Code: [`tabular_benchmark.py`](../code/tabular_benchmark.py)

## 1. A fair benchmark protocol
Same data, same folds, preprocessing **inside** the pipeline, metrics chosen before looking, a baseline, repeated/stratified CV, report spread, and cost (time/memory/latency). Differences smaller than the fold-to-fold standard deviation are ties.

## 2. Measured benchmark: OpenML "adult" (census income, mixed types)
10,000-row random subsample, 14 features (6 numeric, 8 categorical), 24.2% positive; 5-fold stratified CV; default-ish hyperparameters (not tuned); 2 CPU threads. Linear models/k-NN/SVM/MLP use imputation + scaling + one-hot; tree models use imputation + ordinal encoding.

| Model | ROC-AUC (±sd) | PR-AUC | Log loss | Fit s/fold |
|---|---|---|---|---|
| Majority baseline | 0.500 | 0.242 | — | 0.05 |
| Logistic regression | 0.903 ± 0.010 | 0.764 | 0.324 | 0.16 |
| k-NN (k=25) | 0.889 ± 0.011 | 0.712 | 0.422 | 0.05 |
| RBF SVM | 0.899 ± 0.010 | 0.764 | n/a (no probabilities) | 2.02 |
| MLP (64,32) | 0.909 ± 0.007 | 0.775 | 0.316 | 1.65 |
| Random forest (200) | 0.911 ± 0.009 | 0.792 | 0.312 | 1.19 |
| HistGradientBoosting | 0.919 ± 0.009 | 0.812 | 0.295 | 0.48 |
| **XGBoost** | **0.923 ± 0.008** | **0.819** | **0.287** | 0.27 |
| LightGBM | 0.921 ± 0.007 | 0.816 | 0.290 | 0.30 |

**How to read it**
- The three boosted models are statistically tied (differences ≤ 0.004 AUC vs sd ≈ 0.008); they beat the random forest by ~0.01 AUC and logistic regression by ~0.02 AUC.
- A well-regularized **logistic regression is already strong** (0.903) and 10× cheaper than an SVM; most of the "easy" lift is available from a linear baseline.
- k-NN is cheapest to fit but worst here (mixed categorical features and one-hot distance are a poor fit).
- This is one dataset. Rankings change with size, noise, feature types, and tuning (Grinsztajn et al. 2022 benchmark over many tabular datasets: tree ensembles tend to lead on medium-sized tabular data).

## 3. Master comparison table (rules of thumb)

| Algorithm | Task | Data size sweet spot | Nonlinear | Interpretability | Train cost | Predict cost | Key weakness | Chapter |
|---|---|---|---|---|---|---|---|---|
| Linear/Ridge/Lasso | Reg | Any | No | High | Very low | Very low | Linear only | 1.1–1.3 |
| Logistic | Clf | Any | No | High | Low | Very low | Linear boundary | 2.1 |
| Naive Bayes | Clf | Small / text | Limited | High | Very low | Very low | Independence, calibration | 2.3 |
| LDA/QDA | Clf | Small | Quadratic | Medium | Low | Low | Gaussian assumption | 2.4 |
| k-NN | Both | Small–medium | Yes | High (cases) | None | High | Curse of dimensionality | 2.5 |
| SVM (kernel) | Both | ≤ ~50k rows | Yes | Low | High | Medium | Scales poorly | 2.6 |
| Decision tree | Both | Any | Yes | High (small) | Low | Low | High variance | 3.1 |
| Random forest | Both | Medium–large | Yes | Medium | Medium | Medium | Memory, no extrapolation | 3.2 |
| Gradient boosting (XGB/LGBM/Cat) | Both | Medium–large | Yes | Medium | Medium | Low | Tuning, no extrapolation | 3.3–3.4 |
| GMM | Cluster/density | Small–medium | Gaussian mix | Medium | Medium | Low | Gaussian assumption | 3.5 |
| K-Means | Cluster | Any | No | High | Low | Low | Spherical clusters | 4.1 |
| DBSCAN/HDBSCAN | Cluster | Small–medium | Yes | Medium | Medium | — | Parameter/density sensitivity | 4.2 |
| PCA | Reduction | Any | No | Medium | Low | Low | Linear | 4.3 |
| t-SNE/UMAP | Visualization | ≤ 10⁶ | Yes | Low | High | — | Distortion | 4.4 |
| Isolation Forest/LOF | Anomaly | Any | Yes | Medium | Low | Low | Thresholding | 4.5 |
| MLP | Both | Large | Yes | Low | High | Medium | Tuning, data hunger | 5.1 |
| CNN | Images/signals | Large / pretrained | Yes | Low | High | Medium | Data, compute | 5.2 |
| RNN/LSTM | Sequences | Medium | Yes | Low | High | Low (streaming) | Long range, sequential | 5.3 |
| Transformer | Sequences/multimodal | Very large / pretrained | Yes | Low | Very high | High | O(n²) attention, cost | 5.4 |

## 4. Decision flow (tabular supervised)
```
Start → define metric, baseline, split strategy
  │
  ├─ Is the data tabular?  ─ yes → Logistic/Ridge baseline → LightGBM/XGBoost/HistGB → compare, tune lightly
  │                         no  → images/audio/text/graphs → pretrained deep model + fine-tune (Ch 5, 6.x)
  │
  ├─ Need explanations/regulation? → linear/GAM/shallow tree/monotone GBM; add SHAP/permutation for the rest
  ├─ Very few rows (<500)? → regularized linear, NB, SVM, k-NN; avoid big nets
  ├─ Temporal? → time-based CV, seasonal baseline, differenced/lag features
  ├─ Rare target? → PR-AUC, class weights, threshold from cost
  └─ Latency/memory tight? → linear, small GBM, quantized model
```

## 5. Model-selection pitfalls
Comparing default settings of one model vs tuned another · different preprocessing · leaking via encoders · comparing on a split reused for tuning · ignoring variance across folds/seeds · optimizing the wrong metric · neglecting calibration/latency/maintainability.

## 6. Evolution chains recap (what changed, what it cost)
| Chain | Improvement | Cost |
|---|---|---|
| OLS → Ridge → Lasso → ElasticNet | Stability → sparsity → grouping | Bias, tuning |
| Tree → RF → GBM → XGB/LGBM/Cat | Variance ↓ → bias ↓ → speed/regularization/categoricals | Interpretability, tuning |
| Perceptron → SVM → kernel SVM | Max-margin → nonlinear | Quadratic scaling |
| MLP → CNN → ResNet → ViT | Structure → depth → global attention | Data/compute |
| RNN → LSTM → Attention → Transformer | Memory → long range → parallelism | O(n²) |
| VAE/GAN → Diffusion | Fidelity & stability | Sampling cost |
| Prompt → RAG → LoRA → Agents | Behavior → knowledge → specialization → autonomy | Complexity, evaluation burden |
