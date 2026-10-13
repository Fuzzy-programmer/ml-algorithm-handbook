# Section 9 — Natural Language Processing (NLP)

[← Roadmap](../README.md) · Next section: [10 — GenAI & LLMs](../10-genai-and-llms/README.md)

A dedicated, detailed NLP track that sits **before** GenAI/LLMs: it teaches how text becomes numbers, how classical and neural NLP models work, and how the field moved from counting words to pretrained transformers — the foundation that LLM chapters build on.

**How the evidence is organized**
- ✅ **Executed locally (numbers quoted in chapters):** [`code/nlp_text_basics.py`](../code/nlp_text_basics.py), [`nlp_topics_lm.py`](../code/nlp_topics_lm.py), [`nlp_embeddings.py`](../code/nlp_embeddings.py), [`nlp_sequence_labeling.py`](../code/nlp_sequence_labeling.py) (scikit-learn 1.9.1, NLTK 3.10, sklearn-crfsuite, seqeval).
- 📓 **Notebooks for Google Colab:** [`notebooks/nlp_1_classical.ipynb`](../notebooks/nlp_1_classical.ipynb) (CPU; re-runs everything above) and [`notebooks/nlp_2_transformers.ipynb`](../notebooks/nlp_2_transformers.ipynb) (GPU; BERT/DistilBERT/T5/MiniLM/NER/QA/BLEU). The transformer notebook was **not run by the author** — chapters N.8–N.10 mark which numbers are measured and which are for you to fill in. Run the notebook and paste your results into the "Your results" tables.

## The NLP timeline (what each era fixed)
```
1950s-80s  rules, grammars, WordNet-style lexicons ......... brittle, labor intensive
1990s      statistical NLP: n-grams, HMMs, naive Bayes ..... needs counts, sparse
2000s      feature engineering + linear models, CRFs, LDA .. engineered features
2013-2016  word2vec/GloVe, RNN/LSTM, seq2seq + attention ... dense semantics, end-to-end
2017-2019  Transformer, ELMo/BERT pretraining ............... contextual transfer learning
2020-now   LLMs: prompting, RAG, tools, agents .............. see Section 10
```

## Chapters
| # | Chapter | Core question | Priority |
|---|---|---|---|
| N.1 | [Text preprocessing & tokenization](01-text-preprocessing-tokenization.md) | How do I turn raw text into clean tokens? | 🔴 |
| N.2 | [Text representation: BoW, TF-IDF, n-grams, hashing, LSA](02-text-representation.md) | How do I turn tokens into vectors? | 🔴 |
| N.3 | [Text classification & sentiment analysis](03-text-classification-sentiment.md) | What is the best baseline and what moves the needle? | 🔴 |
| N.4 | [Language models: n-grams, smoothing, perplexity](04-language-models-ngram.md) | How do we model word sequences probabilistically? | 🟠 |
| N.5 | [Word embeddings: word2vec, GloVe, fastText](05-word-embeddings.md) | How do we represent meaning densely? | 🔴 |
| N.6 | [Topic modeling: LSA, NMF, LDA](06-topic-modeling.md) | What are the themes in a corpus? | 🟡 |
| N.7 | [Sequence labeling: POS, NER, chunking (HMM, CRF, neural); parsing & IE overview](07-sequence-labeling-ner-pos.md) | How do I label every token? | 🟠 |
| N.8 | [Contextual embeddings & pretrained transformers for NLP](08-pretrained-transformers-for-nlp.md) | How did ELMo/BERT change NLP? | 🔴 |
| N.9 | [Seq2seq tasks: translation, summarization, QA, generation metrics](09-seq2seq-mt-summarization-qa.md) | How do we generate text and score it? | 🟠 |
| N.10 | [NLP evaluation, multilingual & production practice](10-nlp-evaluation-practice.md) | How do I ship a reliable NLP system? | 🔴 |

## Task → method map
| Task | Classical | Neural / pretrained |
|---|---|---|
| Text classification | TF-IDF + LR/SVM/NB | Fine-tuned encoder (BERT-style), embeddings + LR, prompted LLM |
| Sentiment | Lexicon, TF-IDF + LR | Fine-tuned transformer |
| NER / POS | HMM, CRF | BiLSTM-CRF, transformer token classification |
| Topic discovery | LDA, NMF | Embedding clustering (e.g. BERTopic-style pipelines) |
| Semantic search | BM25 | Bi-encoder embeddings + rerank (RAG, Section 10) |
| Machine translation | Phrase-based SMT | Transformer seq2seq / LLM |
| Summarization | Extractive (TextRank) | Seq2seq (BART/T5), LLM |
| QA | IR + rules | Extractive reader (SQuAD), generative + retrieval |
| Language modeling | n-gram + smoothing | Transformer LM (Section 10) |

## Study order
N.1 → N.2 → N.3 (build a baseline classifier) → N.5 → N.8 (feel the jump to contextual models) → N.7 → N.9 → N.10; N.4 and N.6 provide theory and interpretability; then Section 10.
