# Chapter N.2 — Text Representation: Bag-of-Words, TF-IDF, n-grams, Hashing, LSA

[← NLP index](README.md) · Prev [N.1](01-text-preprocessing-tokenization.md) · Next [N.3 Classification](03-text-classification-sentiment.md) · Priority 🔴 · Code: [`nlp_text_basics.py`](../code/nlp_text_basics.py) (executed)

## 1. Identity & problem
Turn a document into a fixed-length numeric vector so ML models (Ch 1–3) can use it. The classical answer ignores word order and counts words; modern answers learn dense contextual vectors (N.5, N.8).

## 2. Lineage
Boolean/vector-space model for retrieval (Salton, 1960s–70s) → **TF-IDF** (Spärck Jones 1972: IDF; Salton & Buckley 1988 weighting) → probabilistic retrieval/BM25 (Robertson et al., 1990s) → **n-grams** to capture local order → **hashing trick** (Weinberger et al. 2009) for memory-bounded vocabularies → **LSA/LSI** (Deerwester et al. 1990; SVD of the term–document matrix) to capture synonymy → probabilistic topic models (N.6) → dense neural embeddings (N.5, N.8). Each step addresses a sparsity/semantics limitation of the previous; none dominates everywhere (TF-IDF remains a very strong baseline).

## 3. Representations
| Representation | Definition | Pros | Cons |
|---|---|---|---|
| One-hot / **BoW counts** | `x_j = count(word j in doc)` | Simple, interpretable | High-dim, sparse, no order/semantics |
| **Binary BoW** | presence/absence | Robust for short docs (sentiment) | Loses frequency |
| **TF-IDF** | `tf × idf`, `idf = log(N/df)` (+ smoothing) | Down-weights common words, strong baseline | Still no semantics |
| **n-grams** | counts of word sequences (n=2,3) or char n-grams | Captures negation/phrases ("not good"), robust to typos (char) | Explodes vocabulary |
| **Hashing** | index = hash(token) mod 2^b | Fixed memory, streaming, no vocabulary | Collisions, not invertible |
| **LSA** | truncated SVD of TF-IDF → k dims | Dense, synonym-aware | Linear, hard to interpret |
| Embedding averages / contextual | N.5/N.8 | Semantic | Need pretrained models |

## 4. Mathematics
Term frequency `tf(t,d)` (raw count, or sublinear `1+log tf`); inverse document frequency `idf(t) = log((1+N)/(1+df(t))) + 1` (scikit-learn smooth variant); `tfidf = tf·idf`, then L2-normalize rows so `cosine(d₁,d₂) = d₁·d₂`.
**Worked example (executed).** Documents: `the cat sat on the mat`, `the dog sat on the log`, `cats and dogs` (N=3). Vocabulary (alphabetical): `and, cat, cats, dog, dogs, log, mat, on, sat, the`.
Count matrix:
```
         and cat cats dog dogs log mat on sat the
doc1      0   1   0    0   0    0   1   1   1   2
doc2      0   0   0    1   0    1   0   1   1   2
doc3      1   0   1    0   1    0   0   0   0   0
```
Document frequencies: `cat`=1, `the`=2, `on`=2, `sat`=2, … So `idf(cat) = ln((1+3)/(1+1)) + 1 = ln 2 + 1 =` **1.693**; `idf(the) = ln(4/3) + 1 =` **1.288** (matches scikit-learn's `idf_`). Notice "cat" and "cats", "dog" and "dogs" are unrelated columns — the lexical-gap problem that stemming/lemmatization (N.1) or embeddings address.
BM25 (retrieval variant) adds term-frequency saturation and length normalization (details in [RAG, Ch G.4](../10-genai-and-llms/04-rag.md)).
**LSA:** `X ≈ U_k Σ_k V_kᵀ`; document coordinates `U_kΣ_k`; query folding `q_k = q V_k Σ_k⁻¹`. Cosine in the latent space can match documents sharing no words but sharing co-occurring words.

## 5. Costs
BoW/TF-IDF matrices are sparse (store only non-zeros): memory ∝ total tokens, not N×V. Typical vocabulary 10⁴–10⁶; linear models train in seconds on sparse data. LSA: randomized SVD `O(nnz·k)`; dense output `N×k`. Hashing: constant memory.

## 6. Executed comparison (sentiment, 2,000 movie reviews, 5-fold CV accuracy)
| Representation / classifier | Accuracy |
|---|---|
| Majority baseline | 0.500 |
| Binary BoW + Naive Bayes | 0.822 |
| Count BoW + Naive Bayes | 0.816 |
| Count BoW + LR | 0.836 |
| **TF-IDF (sublinear) + LR** | **0.859** |
| TF-IDF + bigrams (min_df=2) + LR | 0.855 |
| TF-IDF + bigrams + LinearSVC | **0.879** |
| TF-IDF, stopwords removed + LR | 0.855 |
| TF-IDF, Porter stemming + LR | 0.854 |
| Hashing 2¹⁶, bigrams, L2-normalized counts, no idf + LR (C=1) | 0.752 |
| Hashing 2¹⁶ + TF-IDF weighting + LR | 0.856 |
Reading: (1) TF-IDF > raw counts (+2.3 points for LR); (2) bigrams helped the SVM but not LR here — gains depend on the classifier and regularization; (3) stopword removal and stemming did not help (differences of 0.4–0.5 points are within the ~±1.5-point noise of a 2,000-document CV, so treat as "no benefit"); (4) hashing works once weights are reasonable — the 0.752 result is an **under-fit caused by tiny L2-normalized count features with default C**, not a flaw of hashing itself (adding TF-IDF weighting recovered 0.856).
**LSA** on the 6-newsgroup corpus (executed): the first 6 components explain only 0.3–0.9% of variance each — text variance is spread across thousands of directions, so LSA needs 100–300 dims in practice, and tuning k matters.

## 7. When to use / not
Use TF-IDF+linear for: baselines, small/medium labeled sets, interpretability (coefficients = words), speed, high-dimensional sparse regimes. Use hashing for: streaming, huge/unbounded vocabularies. Use n-grams when negation/phrases matter. Use LSA for quick semantic smoothing/visualization. Avoid bag-of-words for: tasks needing word order/syntax/coreference, paraphrase matching, or long-range context — use contextual models (N.8).

## 8. Hyperparameters
`ngram_range`, `min_df`/`max_df` (prune rare/ubiquitous), `max_features`, `sublinear_tf`, `norm`, `binary`, `stop_words`, `lowercase`, `analyzer` (word/char/char_wb), hashing `n_features`, LSA `n_components`. Tune with CV on the **training folds only** (pipeline) to avoid vocabulary leakage.

## 9. Implementation
```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)
lsa = make_pipeline(vec, TruncatedSVD(200, random_state=0))
hv = make_pipeline(HashingVectorizer(n_features=2**18, alternate_sign=False, norm=None), TfidfTransformer(sublinear_tf=True))
```
From scratch: build a vocabulary dict, a `scipy.sparse` CSR matrix of counts, multiply columns by `idf`, row-normalize.

## 10. Evaluation
Judge representations by the downstream metric under cross-validation with a fixed classifier; for similarity, use retrieval metrics (precision@k) on labeled pairs. On topic retrieval (8 newsgroups, 300 queries, precision@10; executed in the RAG chapter): BM25 0.432, TF-IDF cosine 0.400, LSA-150 0.383, random 0.122.

## Interview Q&A
- *Why IDF?* Words appearing in many documents are less discriminative; IDF down-weights them.
- *Why L2-normalize TF-IDF?* So document length doesn't dominate cosine/dot products.
- *Hashing trick trade-off?* Fixed memory and no vocabulary vs collisions and no feature names.
- *What does LSA capture?* Co-occurrence-based latent dimensions: synonyms land close.
- *When does BoW fail?* Negation scope, word sense, long-range dependencies, paraphrase.

## Exercises
Implement TF-IDF from scratch and verify against scikit-learn; plot accuracy vs `min_df`; compare word and character n-grams on typo-heavy text; inspect LR coefficients for top positive/negative n-grams.

## Further reading
Spärck Jones (1972); Salton & Buckley (1988); Deerwester et al. (1990); Weinberger et al. (2009) "Feature hashing for large scale multitask learning"; Manning et al., *IR* Ch 6; scikit-learn "Working with text data".
