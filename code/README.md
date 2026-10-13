# Code index

Run with `python -W ignore <file>` inside a virtualenv with: `numpy pandas scikit-learn scipy statsmodels lightgbm xgboost catboost lifelines umap-learn` (tested with scikit-learn 1.9.1). Some scripts download datasets on first run (OpenML `adult`, 20 newsgroups, California housing).

| File | Chapters | What it shows |
|---|---|---|
| linear_regression_scratch.py / _sklearn.py | 1.1 | OLS three ways; California housing baseline |
| foundations_demos.py | 0.2–0.5 | MLE/MAP, optimizers, bias–variance, leakage, imbalance metrics, calibration |
| regression_family.py | 1.2–1.4 | Polynomial, Ridge/Lasso/ENet, Huber/quantile, Bayesian ridge |
| classification_scratch.py / classification_family.py | 2.1–2.6 | From-scratch logistic/perceptron/NB/kNN; model comparison; SVM scaling; text |
| trees_scratch.py / trees_family.py | 3.1–3.4 | CART + GBM from scratch; forests and boosting libraries (see memory note below) |
| probabilistic_models.py | 3.5 | EM for GMM, HMM forward/Viterbi, Bayesian-network inference |
| unsupervised_family.py | 4.1–4.7 | Clustering, PCA, t-SNE/UMAP, anomaly detection, feature selection, Apriori |
| deep_learning_numpy.py | 5.1–5.7 | Backprop with gradient check, conv, LSTM, attention, autoencoder, KL, InfoNCE, GCN |
| timeseries_demo.py | 6.1 | ETS/SARIMA/boosted lags with rolling-origin evaluation |
| recsys_demo.py | 6.2 | Popularity/kNN/ALS on synthetic data |
| rl_demo.py | 6.3 | Bandits, value iteration, Q-learning vs SARSA, REINFORCE |
| causal_survival_semisup.py | 6.4–6.6 | Confounding, IPW, DiD, KM/Cox, label spreading |
| tabular_benchmark.py | 7 | Fair multi-model benchmark on OpenML adult |
| project_adult_income.py | 8 | End-to-end tabular project |
| nlp_text_basics.py | N.1–N.3 | Tokenizers, stem/lemma, Zipf/Heaps, TF-IDF, sentiment ablations |
| nlp_topics_lm.py | N.4, N.6 | NMF/LDA topics; n-gram LMs with smoothing and perplexity |
| nlp_embeddings.py | N.5 | PPMI+SVD and skip-gram negative sampling from scratch |
| nlp_sequence_labeling.py | N.7 | HMM/CRF POS tagging; CRF NER on CoNLL-2002 |
| genai_demos.py | G.2–G.8 | BPE, sampling, KV cache, retrieval, LoRA, quantization, DPO, diffusion toy |
| agent_loop_mock.py | G.9–G.10 | Tool-calling loop with a scripted model |

Memory note: `trees_family.py` fits many large ensembles; on a machine with <8 GB RAM reduce `n_estimators` or `n_jobs`.

Notebooks (Google Colab): `../notebooks/nlp_1_classical.ipynb` (CPU; runs the four nlp_*.py scripts) and `../notebooks/nlp_2_transformers.ipynb` (GPU; BERT/DistilBERT/T5/MiniLM/NER/QA/BLEU — not executed by the author). Rebuild notebooks with `python notebooks/_build.py`.
