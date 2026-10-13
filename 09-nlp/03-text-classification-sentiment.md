# Chapter N.3 — Text Classification and Sentiment Analysis

[← NLP index](README.md) · Prev [N.2](02-text-representation.md) · Next [N.4 Language models](04-language-models-ngram.md) · Priority 🔴 · Code: [`nlp_text_basics.py`](../code/nlp_text_basics.py) (executed), [`classification_family.py`](../code/classification_family.py) · Notebook: [`nlp_2_transformers.ipynb`](../notebooks/nlp_2_transformers.ipynb) §6

## 1. Identity & problem
Assign one or more labels to a document: spam vs ham, topic, intent, language, toxicity, sentiment (positive/negative/neutral or aspect-level). The most common production NLP task, and the one where a **simple baseline is usually surprisingly strong**.

## 2. Historical progression (and trade-offs)
| Era | Approach | Strength | Limit |
|---|---|---|---|
| 1960s–90s | Hand rules, keyword lists, sentiment lexicons (General Inquirer, SentiWordNet) | Interpretable, no data | Brittle, domain-specific |
| 1990s | Naive Bayes (Ch 2.3), Rocchio, decision trees | Fast, small data | Independence, weak features |
| 2002 | Pang, Lee & Vaithyanathan: movie-review sentiment with NB/ME/SVM; unigrams+SVM strong; binary features helped | Established ML baseline | Ignores order |
| 2000s | **TF-IDF + linear SVM/LR** (Joachims 1998) — long-time champion; NB-SVM (Wang & Manning 2012) | Excellent small/medium data | No semantics |
| 2013–16 | Averaged embeddings, CNNs (Kim 2014), LSTMs, fastText (Joulin 2016) | Dense semantics, speed | Need data/tuning |
| 2018+ | Fine-tuned **BERT-family** encoders | State of the art on most benchmarks | Compute, latency |
| 2020+ | Prompted/zero-shot LLMs, embeddings + LR | No/low labeled data | Cost, consistency, calibration |

## 3. Choosing a method
```
labels available?
 ├ few (<100) ......... zero/few-shot LLM prompt or sentence-embedding + LR; NB as baseline
 ├ moderate (1e2-1e4) . TF-IDF + LR/SVM baseline  → embeddings + LR → fine-tune small encoder
 └ many (>1e4) ........ fine-tuned transformer usually best; distill to small model for latency
latency/cost tight? → TF-IDF linear or small distilled encoder
need explanations? ..→ linear model coefficients, LIME/SHAP, rationale extraction
```

## 4. Mathematics (recap)
Linear classifier on TF-IDF: `p(y=1|x)=σ(wᵀx+b)` (Ch 2.1); Naive Bayes: `argmax_c log P(c)+Σ_j x_j log θ_cj` with Laplace smoothing (Ch 2.3); SVM maximizes margin (Ch 2.6). Multi-label: one-vs-rest sigmoid outputs; multi-class: softmax. Fine-tuned encoders: `p(y|x)=softmax(W h_[CLS] + b)` trained with cross-entropy, all weights updated with a small learning rate (N.8).
**Worked example (NB log-odds).** Word "great" appears in 8% of positive and 2% of negative reviews: log-likelihood ratio `ln(0.08/0.02)=1.386`; "awful": 1% vs 6% → `ln(1/6)=−1.792`. A review containing both has evidence `1.386−1.792=−0.406` (+ prior log-odds) → leans negative; each word contributes additively, which is why NB coefficients are interpretable but over-confident when words are correlated.

## 5. Executed results — sentiment on 2,000 movie reviews (5-fold CV, accuracy)
Full table in N.2. Key rows: majority 0.500 · NB (binary) 0.822 · LR on counts 0.836 · **TF-IDF + LR 0.859** · **TF-IDF + bigrams + LinearSVC 0.879**. Preprocessing ablations (stopwords, stemming) did not help (0.855, 0.854). A 4-class topic task (20 newsgroups; Ch 2.3/2.6): MultinomialNB 0.842, TF-IDF+LR 0.844, TF-IDF+LinearSVC 0.858.
Takeaway: the classical stack reaches ~0.86–0.88 on this long-review task (document-level sentiment is easier than sentence-level; published numbers for fine-tuned transformers on full IMDB reach the low 90s, which you can reproduce in the notebook with truncation limits in mind). Always compare against it before building something heavy.
**Notebook (not executed by the author):** SST-2 sentence-level sentiment — TF-IDF(1,2)+LR vs frozen MiniLM embeddings + LR vs fine-tuned DistilBERT. Record your validation accuracies here:

| Model (SST-2 validation, 872 sentences) | Your accuracy |
|---|---|
| TF-IDF(1,2) + LR | _run notebook §6_ |
| MiniLM embeddings (frozen) + LR | _run notebook §6_ |
| DistilBERT fine-tuned (2 epochs, 20k train) | _run notebook §6_ |

## 6. Hyperparameters and design
Representation (N.2), regularization `C` (tune on log grid), class weights for imbalance, thresholds per class (multi-label), calibration (Ch 0.5), text length truncation (transformers: typically ≤512 tokens; long documents → chunk and aggregate), learning rate 2e-5–5e-5 / epochs 2–4 / batch 16–32 for fine-tuning.

## 7. Sentiment-specific issues
Negation scope ("not good", "not bad at all"), sarcasm, domain shift (a "predictable" plot is bad in a movie review, good in a utility's reliability report), aspect-based sentiment ("great food, slow service" → per-aspect labels), neutral class ambiguity, label noise (inter-annotator agreement often 70–85%), emoji and slang. Lexicon tools (VADER, TextBlob) are quick zero-training baselines for social media but domain-limited.

## 8. Evaluation
Accuracy only for balanced data; otherwise **macro-F1**, per-class precision/recall, PR-AUC, confusion matrix; stratified CV; hold out **by source/time/author** to avoid leakage (same-author reviews inflate scores); measure calibration; error analysis by slice (length, domain, negation). Report confidence intervals: with 2,000 examples and accuracy 0.86, standard error ≈ 0.8 points, so differences under ~2 points need paired tests.

## 9. Implementation
```python
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
clf = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=3000, C=10))
scores = cross_val_score(clf, texts, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
top = sorted(zip(clf[-1].coef_[0], clf[0].get_feature_names_out()))[-15:]      # most positive n-grams
```
Fine-tuning with Hugging Face: see the notebook (plain PyTorch loop, version-robust).

## 10. Case study (template)
Support-ticket routing, 12 intents, 6k labeled: TF-IDF+LR baseline → add char n-grams → sentence embeddings + LR → fine-tune a small encoder. Compare macro-F1, p95 latency, cost; examine confusions between near-synonymous intents; set a reject threshold so low-confidence tickets go to humans.

## Interview Q&A
- *Best baseline for text classification?* TF-IDF (word+char n-grams) with logistic regression or linear SVM.
- *Why does NB work for text?* Needs correct ranking of class scores, not calibrated probabilities; high-dimensional sparse data suit its low variance.
- *When fine-tune vs prompt an LLM?* Enough labels, stable label set, volume/latency/cost constraints → fine-tune a small model; few labels or fast-changing label definitions → prompting.
- *How handle long documents?* Chunk + aggregate, hierarchical models, long-context models, or summarize first.
- *Class imbalance?* Class weights, resampling, PR-AUC, threshold tuning.

## Exercises
Build the TF-IDF baseline on a dataset of your choice and report top n-grams; add a negation-aware feature (prefix tokens after "not" until punctuation) and measure its effect; fine-tune DistilBERT (notebook) and compare errors with the linear model; calibrate and choose a rejection threshold.

## Further reading
Pang & Lee (2008) *Opinion Mining and Sentiment Analysis*; Joachims (1998); Wang & Manning (2012) NBSVM; Kim (2014) CNN for sentences; Joulin et al. (2016) fastText; Devlin et al. (2018) BERT; Liu (2012) *Sentiment Analysis and Opinion Mining*.
