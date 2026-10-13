# Chapter N.10 — NLP Evaluation, Multilingual Issues and Production Practice

[← NLP index](README.md) · Prev [N.9](09-seq2seq-mt-summarization-qa.md) · Next section: [10 — GenAI & LLMs](../10-genai-and-llms/README.md) · Priority 🔴

## 1. Purpose
Tie the NLP chapters together: how to choose a model, evaluate it honestly, handle languages and domains, and run NLP systems in production. This chapter is a checklist-oriented synthesis built on the executed experiments in N.1–N.7.

## 2. The model-selection ladder (use the simplest that meets the bar)
| Rung | Method | When it wins | Evidence in this handbook |
|---|---|---|---|
| 0 | Rules / regex / gazetteers / lexicons | Structured patterns, no data, audits | N.7 (hybrid approach) |
| 1 | TF-IDF + linear / NB | Small–medium labels, speed, interpretability | Sentiment 0.859–0.879 vs majority 0.500 (N.2/N.3); 4-class topic 0.84–0.86 |
| 2 | CRF / feature-rich linear taggers | Token labeling with limited data | POS 0.9705 vs baseline 0.9297; NER F1 0.787 (N.7) |
| 3 | Static embeddings (word2vec/fastText) + classifier | Small labels + domain text, OOV morphology | N.5 |
| 4 | Frozen sentence embeddings + LR / k-NN | Little labeled data, quick semantic features | Notebook §5–6 |
| 5 | Fine-tuned small/base encoder | Production classification/NER/reranking | Notebook §6, N.8 |
| 6 | Seq2seq / LLM prompting / RAG | Generation, flexible tasks, few labels | N.9, Section 10 |
Move up a rung only if a **measured** improvement justifies the added cost, latency and maintenance. Keep the rung-1 model as a regression test and fallback.

## 3. Evaluation protocol for NLP
1. **Define the metric from the decision:** accuracy for balanced single-label; macro-F1 for imbalance; entity-level F1 (not token accuracy) for NER; EM/F1 for QA; BLEU/chrF/COMET for MT; ROUGE + faithfulness for summarization; MRR/NDCG/Recall@k for retrieval.
2. **Split correctly:** by document/author/thread/time/source, never random sentences (leakage via near-duplicates; e.g. newsgroup quoted replies, product-review templates). Deduplicate before splitting.
3. **Baselines:** majority, keyword/lexicon, TF-IDF+LR; report them in every table.
4. **Uncertainty:** CIs by bootstrap; multiple seeds for neural models; paired tests for model comparisons. **Worked example:** accuracy 0.859 on 2,000 documents ⇒ standard error `√(0.859·0.141/2000)=0.0078`, 95% CI ≈ ±1.5 points — so 0.859 vs 0.855 (Table N.2) is not a real difference.
5. **Error analysis:** sample 100 errors, tag causes (negation, sarcasm, OOV/rare entities, label noise, truncation, domain shift), fix categories not instances. Slice metrics by length, language, dialect, topic, demographic mentions.
6. **Annotation quality:** measure inter-annotator agreement (Cohen's κ / Krippendorff's α; entity F1 between annotators); write guidelines with examples; adjudicate disagreements; the human ceiling bounds achievable scores.
7. **Robustness & behavior tests:** CheckList-style tests (negation, typos, name swaps, paraphrase), adversarial/perturbation suites, out-of-domain evaluation.
8. **Calibration & abstention:** reliability curves, confidence thresholds to route uncertain cases to humans (Ch 0.5).

## 4. Multilingual NLP
- **Challenges:** no whitespace (zh, ja, th), rich morphology (fi, tr, ar), scripts and normalization (Unicode NFKC, Arabic diacritics), code-switching, dialects, transliteration, low-resource languages with little unlabeled/labeled data, tokenizer fertility (more tokens per word in some languages ⇒ higher cost and shorter effective context).
- **Approaches:** language-specific models where available; multilingual encoders (mBERT, **XLM-R**) with zero-shot cross-lingual transfer (fine-tune on English, test on others — works best for related languages and large-resource languages); translate-train / translate-test; multilingual LLMs; language ID first (fastText LID or character n-gram LMs, N.4); evaluate **per language**, never only on average.
- **Fairness note:** performance gaps across languages and dialects are common; report them.

## 5. Bias, safety and privacy in NLP
Embeddings and models inherit societal biases (N.5); sentiment/toxicity classifiers can mislabel dialects (e.g. African-American English flagged toxic more often — Sap et al. 2019); NER may miss non-Western names; generation can leak training data. Mitigations: audit slices, counterfactual augmentation (swap names/pronouns), de-identification (PII NER + redaction), data governance, documentation (datasheets/model cards), human oversight for high-stakes use.

## 6. Production architecture
```
raw text → language ID → normalization/cleaning → tokenization (model-matched) → model(s) → post-processing/validation → business rules → output
              │                                                                       │
          monitoring: input length, language mix, OOV rate, score distributions, latency, drift, human review samples
```
- **Reproducibility:** pin tokenizer + model versions; one shared preprocessing function for training and serving (train/serve skew is the #1 NLP bug).
- **Latency/cost:** batch requests, dynamic padding, ONNX/TensorRT, quantization, distillation, caching of embeddings, ANN indices (Ch 2.5), truncation policies, asynchronous processing.
- **Drift:** vocabulary and topic drift (new slang, products, events), label drift; monitor out-of-vocabulary rate, embedding-space distribution shift, performance on a rolling labeled sample; schedule retraining.
- **Human-in-the-loop:** active learning (label the most uncertain/diverse examples), review queues, feedback capture, relabeling workflows.
- **Testing:** unit tests for tokenization edge cases (emoji, Unicode, long inputs, empty strings), regression suite of historical failures, shadow deployment and canary releases.

## 7. Common mistakes (seen repeatedly)
Random split of near-duplicate texts · evaluating NER with token accuracy · comparing models trained with different preprocessing · tuning on the test set · ignoring class imbalance · assuming a BERT beats a linear model without testing · forgetting `max_length` truncation hides the decisive sentence · lowercasing for case-sensitive models · mismatched tokenizer · reporting average over languages hiding failures · no monitoring after launch.

## 8. Project template: end-to-end text classification (reuse for any label set)
1. Collect and deduplicate data; define labels and guidelines; label 500–2,000 examples (stratified, with a gold subset labeled twice for agreement).
2. Baseline: TF-IDF (word + char n-grams) + LR; macro-F1 and confusion matrix; inspect top features and errors.
3. Candidates: sentence embeddings + LR; fine-tuned small encoder (several seeds); optionally prompted LLM on a sample.
4. Compare under the same split and CIs; add cost/latency; choose the simplest model within tolerance of the best.
5. Calibrate, set thresholds, define abstention; fairness/slice analysis.
6. Package as one pipeline; serving tests; monitoring; retraining plan.
Everything needed for steps 2–4 exists in `code/nlp_text_basics.py` and `notebooks/nlp_2_transformers.ipynb`; the tabular analogue is in [Project 1](../08-projects/01-end-to-end-tabular-project.md).

## 9. Datasets and tools (starting points)
Text classification: 20 Newsgroups, IMDB, SST-2, AG News; NER: CoNLL-2002/2003, OntoNotes, WNUT; POS/parsing: Penn Treebank, Universal Dependencies; QA: SQuAD, Natural Questions, HotpotQA; MT: WMT, FLORES; summarization: CNN/DailyMail, XSum; embeddings/retrieval: MTEB, BEIR; benchmarks: GLUE/SuperGLUE (saturated), XTREME/XNLI (multilingual). Tools: NLTK, spaCy, scikit-learn, gensim, Hugging Face (transformers, datasets, evaluate, tokenizers), sentence-transformers, seqeval, sacreBLEU. Check each dataset's license.

## 10. From NLP to LLMs (bridge to Section 10)
LLMs subsume many pipeline stages (tokenization → subwords; representation → learned embeddings; task heads → prompts) but the same engineering questions remain: representative evaluation sets, leakage, uncertainty, error analysis, bias, drift, cost. The tokenization, language-model, embedding and transformer concepts in N.1, N.4, N.5 and N.8 are exactly what [Ch G.2](../10-genai-and-llms/02-language-models-llms.md) builds on; retrieval metrics from N.2/N.8 feed [RAG](../10-genai-and-llms/04-rag.md); evaluation discipline continues in [Ch G.11](../10-genai-and-llms/11-evaluation-safety-llmops.md).

## Interview Q&A
- *You have 500 labeled examples and need a text classifier. Plan?* TF-IDF+LR baseline, sentence-embedding+LR, maybe fine-tune a small encoder with several seeds; choose by CV with CIs, then calibrate and monitor.
- *Why did your offline score not match production?* Distribution shift, leakage, train/serve skew, label noise, truncation, or metric mismatch.
- *How evaluate NER properly?* Strict entity-level F1 with seqeval, per-type breakdown, out-of-domain test.
- *How handle many languages?* Language ID, multilingual encoder or per-language models, per-language evaluation.
- *How decide BERT vs linear?* Measure on your data with CIs; consider cost/latency and maintenance.

## Exercises
Draft annotation guidelines for a 5-class problem and compute κ on 100 double-labeled items; run a CheckList-style test (negation, typos, name swap) on your baseline and a fine-tuned model; build a drift monitor that tracks OOV rate and score distribution; evaluate a multilingual model per language on XNLI.

## Further reading
Ribeiro et al. (2020) "Beyond accuracy: behavioral testing of NLP models with CheckList"; Sap et al. (2019) "The risk of racial bias in hate speech detection"; Bender & Friedman (2018) data statements; Mitchell et al. (2019) Model Cards; Hovy & Prabhumoye (2021) "Five sources of bias in NLP"; Ruder (2019–) NLP-progress and multilingual NLP resources; Chip Huyen, *Designing Machine Learning Systems*.
