# ML Algorithm Handbook

A structured, GitHub-friendly handbook for becoming an **ML Engineer**: every algorithm is documented with the same template, so you can compare them honestly and see *why* ML moved from simple methods to complex ones.

**Depth:** comprehensive · **Math:** intuition + equations + worked examples · **Code:** NumPy from scratch + scikit-learn

> **Status:** all planned chapters are written (see the roadmap in Section 4). Code in `code/` was executed; see the verification policy.

---

## 1. How to use this handbook

1. Read the **roadmap** (this file) once, to see the whole landscape.
2. Study chapters in order within a Phase; skip ahead only if the prerequisites column says you can.
3. For each chapter: read the concept notes → run the code in `code/` → do the exercises → fill your own row in the comparison sheet.
4. Priority tags (for time-constrained study): 🔴 MUST · 🟠 SHOULD · 🟡 GOOD · ⚪ OPTIONAL

Every chapter follows [`TEMPLATE.md`](TEMPLATE.md). Code in `code/` was run before being documented; numbers quoted in chapters come from those runs.

## 2. Repository layout

```
ml-algorithm-handbook/
├── README.md                  <- you are here (roadmap)
├── TEMPLATE.md                <- the 16-section chapter template
├── 00-foundations/            math, stats, optimization, evaluation
├── 01-regression/             linear, polynomial, regularized models
├── 02-classification/         logistic regression, SVM, Naive Bayes, k-NN
├── 03-trees-and-ensembles/    trees, forests, boosting
├── 04-unsupervised-learning/  clustering, PCA, anomaly detection
├── 05-deep-learning/          MLP, CNN, RNN, Transformers, generative
├── 06-specialized-ml/         time series, RL, causal, recommenders
├── 07-comparisons/            decision matrices, benchmarks
├── 08-projects/               end-to-end implementations
├── 09-nlp/                    dedicated NLP track (10 chapters) before GenAI
├── 10-genai-and-llms/         LLMs, prompting, RAG, adaptation, diffusion, multimodal, agents
└── code/                      runnable scripts referenced by chapters
```

## 3. The ML landscape (taxonomy)

```
Machine Learning
├── Supervised  (labels given)
│   ├── Regression ......... Linear, Ridge/Lasso, Quantile, SVR, trees, GBMs
│   └── Classification ..... Logistic, NB, LDA/QDA, SVM, k-NN, trees, GBMs
├── Unsupervised (no labels)
│   ├── Clustering ......... K-Means, GMM, Hierarchical, DBSCAN/HDBSCAN, Spectral
│   ├── Dim. reduction ..... PCA, Kernel PCA, ICA, NMF, t-SNE, UMAP
│   ├── Anomaly detection .. Isolation Forest, LOF, One-Class SVM
│   └── Association rules .. Apriori, FP-Growth, Eclat
├── Semi/Self-supervised ... pseudo-labeling, label propagation, contrastive, masked prediction
├── Reinforcement .......... bandits, DP, MC, TD, Q-learning, DQN, policy gradient, PPO, SAC
├── Deep learning .......... MLP, CNN, RNN/LSTM/GRU, Transformer, AE/VAE/GAN/Diffusion, GNN
├── Generative AI & LLMs ... LLM pretraining, prompting, RAG, PEFT/LoRA, alignment, diffusion,
│                            multimodal, tool calling, agents (see Phase 6)
└── Specialized ............ time series, recommenders, causal inference, survival, ranking
Modeling view: discriminative p(y|x) (logistic, SVM, trees, most NNs) vs. generative p(x) or p(x,y)
               (Naive Bayes, GMM, VAE, GAN, diffusion, autoregressive LLMs)
Cross-cutting: feature engineering · evaluation · optimization · ensembles · deployment (MLOps)
```

**Important:** these families are not one linear timeline. Some algorithms *evolve* from predecessors (Tree → Forest → Boosting); others *solve different problems* (k-Means vs. PCA). Each chapter's "Evolution" section states the actual relationship, including where a newer method does **not** beat an older one.

## 4. Chapter roadmap (all chapters written)

Legend: ✅ written, with executed code where noted · Priority 🔴 MUST · 🟠 SHOULD · 🟡 GOOD · ⚪ OPTIONAL

### Phase 0 — Foundations (study alongside Phase 1)
| # | Chapter | Priority |
|---|---|---|
| 0.1 | [Linear algebra & calculus you actually need](00-foundations/01-math-essentials.md) ✅ | 🔴 |
| 0.2 | [Probability, MLE/MAP, EM](00-foundations/02-probability-mle-map.md) ✅ | 🔴 |
| 0.3 | [Optimization: GD, SGD, momentum, Adam/AdamW](00-foundations/03-optimization.md) ✅ | 🔴 |
| 0.4 | [Bias–variance, regularization, generalization](00-foundations/04-bias-variance-regularization.md) ✅ | 🔴 |
| 0.5 | [Evaluation: metrics, CV, leakage, calibration, thresholds](00-foundations/05-evaluation.md) ✅ | 🔴 |
| 0.6 | [Feature engineering & preprocessing pipelines](00-foundations/06-feature-engineering.md) ✅ | 🔴 |

### Phase 1 — Classical supervised learning
| # | Chapter | Builds on | Priority |
|---|---|---|---|
| 1.1 | [Linear Regression (OLS)](01-regression/01-linear-regression.md) ✅ | Legendre/Gauss | 🔴 |
| 1.2 | [Polynomial regression & basis expansion](01-regression/02-polynomial-regression.md) ✅ | 1.1 | 🟠 |
| 1.3 | [Ridge, Lasso, Elastic Net](01-regression/03-ridge-lasso-elasticnet.md) ✅ | 1.1 | 🔴 |
| 1.4 | [Quantile, robust & Bayesian regression](01-regression/04-quantile-bayesian-regression.md) ✅ | 1.1 | 🟡 |
| 2.1 | [Logistic Regression](02-classification/01-logistic-regression.md) ✅ | 1.1 + GLM | 🔴 |
| 2.2 | [Perceptron](02-classification/02-perceptron.md) ✅ | neurons (precursor of nets) | 🟡 |
| 2.3 | [Naive Bayes](02-classification/03-naive-bayes.md) ✅ | Bayes' rule | 🟠 |
| 2.4 | [LDA / QDA](02-classification/04-lda-qda.md) ✅ | Fisher 1936 | 🟡 |
| 2.5 | [k-NN & nearest centroid](02-classification/05-knn.md) ✅ | — | 🟠 |
| 2.6 | [SVM: linear, kernel, SVR](02-classification/06-svm.md) ✅ | Perceptron / max-margin | 🟠 |
| 3.1 | [Decision trees (ID3 → C4.5 → CART)](03-trees-and-ensembles/01-decision-trees.md) ✅ | — | 🔴 |
| 3.2 | [Bagging, Random Forest, Extra Trees](03-trees-and-ensembles/02-bagging-random-forest.md) ✅ | 3.1 | 🔴 |
| 3.3 | [AdaBoost → Gradient Boosting](03-trees-and-ensembles/03-adaboost-gradient-boosting.md) ✅ | 3.1 | 🔴 |
| 3.4 | [XGBoost, LightGBM, CatBoost, HistGB](03-trees-and-ensembles/04-xgboost-lightgbm-catboost.md) ✅ | 3.3 | 🔴 |
| 3.5 | [GMM, HMM, Bayesian networks](03-trees-and-ensembles/05-gmm-hmm-bayesnet.md) ✅ | MLE, EM | 🟡 |

### Phase 2 — Unsupervised learning & representation
| # | Chapter | Priority |
|---|---|---|
| 4.1 | [K-Means / K-Medoids](04-unsupervised-learning/01-kmeans.md) ✅ | 🔴 |
| 4.2 | [Hierarchical, DBSCAN, HDBSCAN, OPTICS, Mean Shift, Spectral](04-unsupervised-learning/02-other-clustering.md) ✅ | 🟠 |
| 4.3 | [PCA, SVD, Kernel PCA, ICA, NMF](04-unsupervised-learning/03-pca-family.md) ✅ | 🔴 |
| 4.4 | [t-SNE, UMAP](04-unsupervised-learning/04-tsne-umap.md) ✅ | 🟠 |
| 4.5 | [Anomaly detection](04-unsupervised-learning/05-anomaly-detection.md) ✅ | 🟠 |
| 4.6 | [Association rules](04-unsupervised-learning/06-association-rules.md) ✅ | ⚪ |
| 4.7 | [Feature selection](04-unsupervised-learning/07-feature-selection.md) ✅ | 🟠 |

### Phase 3 — Model selection & tuning
Covered in [0.4](00-foundations/04-bias-variance-regularization.md), [0.5](00-foundations/05-evaluation.md), [3.4 (tuning order)](03-trees-and-ensembles/04-xgboost-lightgbm-catboost.md), the [comparison protocol](07-comparisons/README.md) and the [end-to-end project](08-projects/01-end-to-end-tabular-project.md) (randomized search, calibration, thresholds).

### Phase 4 — Deep learning
| # | Chapter | Priority |
|---|---|---|
| 5.1 | [MLP & backpropagation](05-deep-learning/01-mlp-backprop.md) ✅ | 🔴 |
| 5.2 | [CNN lineage → ViT](05-deep-learning/02-cnn-lineage.md) ✅ | 🔴 |
| 5.3 | [RNN / LSTM / GRU / TCN / Seq2Seq](05-deep-learning/03-sequence-models.md) ✅ | 🟠 |
| 5.4 | [Attention & Transformers](05-deep-learning/04-attention-transformers.md) ✅ | 🔴 |
| 5.5 | [Autoencoders, VAE, GAN](05-deep-learning/05-generative-ae-vae-gan.md) ✅ | 🟠 |
| 5.6 | [Graph neural networks](05-deep-learning/06-gnn.md) ✅ | 🟡 |
| 5.7 | [Self-supervised & representation learning](05-deep-learning/07-self-supervised.md) ✅ | 🔴 |

### Phase 5 — Specialized & advanced
| # | Chapter | Priority |
|---|---|---|
| 6.1 | [Time series](06-specialized-ml/01-time-series.md) ✅ | 🟠 |
| 6.2 | [Recommender systems](06-specialized-ml/02-recommender-systems.md) ✅ | 🟠 |
| 6.3 | [Reinforcement learning](06-specialized-ml/03-reinforcement-learning.md) ✅ | 🟠 |
| 6.4 | [Causal inference](06-specialized-ml/04-causal-inference.md) ✅ | 🟡 |
| 6.5 | [Survival analysis](06-specialized-ml/05-survival-analysis.md) ✅ | 🟡 |
| 6.6 | [Semi-supervised & weak supervision](06-specialized-ml/06-semi-weak-supervised.md) ✅ | 🟡 |

### Phase 5b — Natural Language Processing ([index](09-nlp/README.md)) — before GenAI/LLMs
| # | Chapter | Priority |
|---|---|---|
| N.1 | [Text preprocessing & tokenization](09-nlp/01-text-preprocessing-tokenization.md) ✅ | 🔴 |
| N.2 | [Text representation: BoW, TF-IDF, n-grams, hashing, LSA](09-nlp/02-text-representation.md) ✅ | 🔴 |
| N.3 | [Text classification & sentiment](09-nlp/03-text-classification-sentiment.md) ✅ | 🔴 |
| N.4 | [Language models: n-grams, smoothing, perplexity](09-nlp/04-language-models-ngram.md) ✅ | 🟠 |
| N.5 | [Word embeddings: word2vec, GloVe, fastText](09-nlp/05-word-embeddings.md) ✅ | 🔴 |
| N.6 | [Topic modeling: LSA, NMF, LDA](09-nlp/06-topic-modeling.md) ✅ | 🟡 |
| N.7 | [Sequence labeling: POS, NER (HMM, CRF, neural) + parsing/IE overview](09-nlp/07-sequence-labeling-ner-pos.md) ✅ | 🟠 |
| N.8 | [Contextual embeddings & pretrained transformers](09-nlp/08-pretrained-transformers-for-nlp.md) ✅ (📓 notebook) | 🔴 |
| N.9 | [Seq2seq: MT, summarization, QA, metrics](09-nlp/09-seq2seq-mt-summarization-qa.md) ✅ (📓 notebook) | 🟠 |
| N.10 | [NLP evaluation, multilingual & production practice](09-nlp/10-nlp-evaluation-practice.md) ✅ | 🔴 |

### Phase 6 — Generative AI, LLMs and agentic systems ([index](10-genai-and-llms/README.md))
| # | Chapter | Priority |
|---|---|---|
| G.1 | [Generative vs. discriminative](10-genai-and-llms/01-generative-vs-discriminative.md) ✅ | 🔴 |
| G.2 | [Language models → LLMs](10-genai-and-llms/02-language-models-llms.md) ✅ | 🔴 |
| G.3 | [Prompt engineering](10-genai-and-llms/03-prompt-engineering.md) ✅ | 🔴 |
| G.4 | [RAG](10-genai-and-llms/04-rag.md) ✅ | 🔴 |
| G.5 | [Efficient adaptation: LoRA/QLoRA, distillation, quantization](10-genai-and-llms/05-efficient-adaptation.md) ✅ | 🟠 |
| G.6 | [Alignment: SFT, RLHF, DPO](10-genai-and-llms/06-alignment-rlhf-dpo.md) ✅ | 🟠 |
| G.7 | [Diffusion models](10-genai-and-llms/07-diffusion-models.md) ✅ | 🟠 |
| G.8 | [Multimodal models](10-genai-and-llms/08-multimodal.md) ✅ | 🟠 |
| G.9 | [Tool calling](10-genai-and-llms/09-tool-calling.md) ✅ | 🔴 |
| G.10 | [Agents & agentic AI](10-genai-and-llms/10-agents-agentic-ai.md) ✅ | 🔴 |
| G.11 | [Evaluation, safety & LLMOps](10-genai-and-llms/11-evaluation-safety-llmops.md) ✅ | 🔴 |

### Cross-cutting
[Comparisons & benchmark](07-comparisons/README.md) ✅ · [End-to-end project](08-projects/01-end-to-end-tabular-project.md) ✅ · [Chapter template](TEMPLATE.md) · [Code index](code/README.md) · [Colab notebooks](notebooks/) (NLP classical on CPU, NLP transformers on GPU)

### Verification policy
Every chapter's numbers come from code in `code/` that was executed (scikit-learn 1.9.1, NumPy 2.x, XGBoost 3.4, LightGBM 4.7, CatBoost 1.2, statsmodels, lifelines). **Not executed** (marked in-chapter): the transformer parts of the NLP section (run `notebooks/nlp_2_transformers.ipynb` on Colab and record your results) and PyTorch / hosted-LLM-API snippets elsewhere (no API key; heavy runs were moved to notebooks at the user's request). Historical dates and citations are from memory — verify before publication. Fast-moving GenAI content is dated.

## 5. Evolution chains (preview)

For every chain we record: **improvement · problem solved · cost of the improvement.**

| Chain | Step | Improvement | Cost |
|---|---|---|---|
| Linear models | OLS → Ridge | Stable under collinearity, lower variance | Bias; needs λ tuning and scaling |
| | Ridge → Lasso | Sparse coefficients (feature selection) | Unstable among correlated features |
| | Lasso/Ridge → Elastic Net | Sparsity + grouping of correlated features | Two hyperparameters |
| Trees | Tree → Random Forest | Large variance reduction | Less interpretable, more memory |
| | Forest → Gradient Boosting | Optimizes arbitrary loss, often higher accuracy | Sequential, more tuning, overfits if unchecked |
| | GBM → XGBoost / LightGBM / CatBoost | Regularization, speed, missing values, categoricals | More hyperparameters; implementation-specific quirks |
| Margin | Perceptron → SVM → Kernel SVM | Max margin, nonlinear boundaries | O(n²–n³) training; kernel/C tuning |
| Sequence DL | RNN → LSTM/GRU → Attention → Transformer | Long-range dependencies, parallel training | Quadratic attention cost; data hungry |
| LLM use | Zero-shot prompt → few-shot / chain-of-thought → RAG → fine-tuning (LoRA) | Progressively more task-specific knowledge/behavior | More infrastructure, data, and evaluation burden |
| LLM systems | Single prompt → tool calling → agent loop → multi-agent | Can act, retrieve, verify | Latency, cost, compounding errors, security risk |
| Image generation | VAE / GAN → diffusion → latent diffusion | Stable training, high fidelity | Many sampling steps (mitigated by distillation/fast samplers) |

*Not every arrow means "better".* Random Forest, XGBoost and LightGBM trade off differently, and a regularized linear model often wins on small or noisy data.

## 6. Algorithm selection cheat-sheet (starting points, not winners)

| Situation | Start with | Why |
|---|---|---|
| Tabular regression / classification | Linear/Logistic baseline → HistGradientBoosting / XGBoost / LightGBM | Cheap baseline, then strongest general tabular family |
| Need explainability / regulatory | Linear/Logistic, shallow tree, GAM | Coefficients or rules are auditable |
| Few rows (<1k), many features | Regularized linear, SVM, Naive Bayes | Low variance |
| Customer segmentation | K-Means, GMM, DBSCAN | Different cluster-shape assumptions |
| Fraud / anomalies | Boosted trees if labels, Isolation Forest if not | Depends on labels and imbalance |
| Text | TF-IDF + Logistic → fine-tuned Transformer | Cheap baseline before heavy models |
| Images / audio | CNN / ViT with transfer learning | Spatial structure |
| Forecasting | Seasonal naive → ETS/ARIMA → boosted trees | Always beat the naive baseline first |
| Recommendations | Popularity → matrix factorization → two-tower + ranker | Depends on interaction volume |

Data quality, leakage, evaluation design and serving constraints often matter more than the algorithm.

## 7. Suggested schedule (1–2 h/day)

| Weeks | Focus |
|---|---|
| 1–2 | Python/NumPy/pandas, statistics, calculus basics, train/test split |
| 3–4 | Linear & logistic regression, regularization, metrics |
| 5–6 | Trees, forests, boosting |
| 7–8 | Clustering, PCA, anomaly detection |
| 9–10 | Cross-validation, tuning, error analysis, tabular project |
| 11–14 | Neural nets, backprop, CNNs, sequence models, attention |
| 15–18 | GenAI track: LLMs, prompting, RAG, tool calling, agents (6.1–6.4, 6.9–6.11 first) |
| 19+ | Specialize: time series, NLP, recsys, RL, causal, diffusion/multimodal |

## 8. Core references (used throughout)

- **scikit-learn User Guide** — https://scikit-learn.org/stable/user_guide.html
- **Google ML Crash Course** — https://developers.google.com/machine-learning/crash-course/
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning* (free PDF from the authors)
- James, Witten, Hastie, Tibshirani — *An Introduction to Statistical Learning* (free; Python edition)
- Bishop — *Pattern Recognition and Machine Learning*
- Murphy — *Probabilistic Machine Learning: An Introduction* (free online)
- Jurafsky & Martin — *Speech and Language Processing* (free draft online; LLM chapters)
- Chip Huyen — *AI Engineering* (LLM application engineering)
- Vendor docs for prompting, tool use and agents (e.g. Anthropic, OpenAI) — fast-moving, always check the current version
- Goodfellow, Bengio, Courville — *Deep Learning* (free online)
- Géron — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*

Per-algorithm original papers are cited in each chapter's Section 16. Verify any citation before relying on it in published work.

## 9. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scikit-learn matplotlib statsmodels   # chapters tested with scikit-learn 1.9.1, NumPy 2.x
```
