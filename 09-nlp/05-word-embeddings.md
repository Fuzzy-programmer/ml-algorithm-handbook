# Chapter N.5 — Word Embeddings: Co-occurrence, word2vec, GloVe, fastText

[← NLP index](README.md) · Prev [N.4](04-language-models-ngram.md) · Next [N.6 Topic modeling](06-topic-modeling.md) · Priority 🔴 · Code: [`nlp_embeddings.py`](../code/nlp_embeddings.py) (executed)

## 1. Identity & problem
Represent each word as a dense vector (50–300 dims) so that **similar words have similar vectors** and arithmetic/similarity captures meaning. Replaces sparse one-hot/BoW features (where "cat" and "dog" are orthogonal) with a learned geometry; pretrained vectors provide transfer learning from unlabeled text.

## 2. Lineage — distributional semantics
"You shall know a word by the company it keeps" (Firth 1957; Harris 1954) → count-based vectors: co-occurrence matrices + PPMI + SVD (LSA 1990; HAL; Bullinaria & Levy 2007) → neural LMs learn embeddings as parameters (Bengio et al. 2003; Collobert & Weston 2008) → **word2vec** (Mikolov et al. 2013: CBOW and skip-gram with negative sampling, fast and scalable) → **GloVe** (Pennington et al. 2014: weighted least squares on log co-occurrences) → **fastText** (Bojanowski et al. 2017: word = sum of character n-gram vectors; handles OOV and morphology) → theory linking SGNS to factorizing a shifted PMI matrix (**Levy & Goldberg 2014**) → **contextual embeddings** (ELMo 2018, BERT 2018; N.8) because a single vector per word cannot represent polysemy ("bank").

## 3. Intuition
Words occurring in similar contexts get similar vectors. Skip-gram: given a center word, predict its neighbors; the vectors are the by-product that makes predictions work.

## 4. Mathematics
**Skip-gram with negative sampling (SGNS):** for each (center `w`, context `c`) pair from a window, maximize `log σ(v_w·u_c) + Σ_{k=1..K} E_{n~P_n}[log σ(−v_w·u_n)]`, with noise distribution `P_n ∝ f^{3/4}` and frequent-word subsampling `P(keep) ≈ √(t/f)`. Gradients for one pair: with `g_pos = σ(v·u_c) − 1`, `g_neg,k = σ(v·u_nk)`: `∂/∂v = g_pos u_c + Σ g_neg,k u_nk`.
**Worked example.** `v·u_c = 0.5`: positive term `log σ(0.5) = −0.474`; negative samples with `v·u_n = 0.3, −0.2`: `log σ(−0.3)=−0.854`, `log σ(0.2)=−0.598`. Objective contribution `−0.474−0.854−0.598 = −1.926`; gradients push `v` toward `u_c` (weight `σ(0.5)−1 = −0.378`) and away from negatives (weights 0.574 and 0.450).
**PPMI + SVD:** `PMI(w,c) = log [P(w,c) / (P(w)P(c))]`, `PPMI = max(PMI,0)`; context distribution smoothing `P(c)^{0.75}`; truncated SVD `P ≈ UΣVᵀ`; embeddings `U Σ^{1/2}`. **GloVe:** minimize `Σ f(X_ij)(w_iᵀw̃_j + b_i + b̃_j − log X_ij)²`. **Similarity:** cosine. **Analogies:** `king − man + woman ≈ queen` works when the corpus is large (billions of tokens) and is partly an artifact of evaluation conventions (the query words are excluded from the answers).
**Properties and caveats:** embeddings encode biases of the training text (gender/occupation stereotypes; Bolukbasi et al. 2016); frequency effects distort cosine; word vectors from different training runs are not aligned (rotation ambiguity).

## 5. Executed experiment (Brown corpus, ~0.88M tokens kept, vocabulary = 6,056 words with count ≥15)
Two embedding methods built **from scratch**: PPMI+SVD (window 4, 100 dims) and skip-gram negative sampling (60 dims, 3 epochs, 5 negatives, batch gradient updates in NumPy; ~142 s on CPU).
Nearest neighbors by cosine:
| Query | PPMI+SVD | SGNS |
|---|---|---|
| president | kennedy, vice, richard, chairman, former, nixon | kennedy, vice, nixon, eisenhower, senator, elected |
| car | parked, truck, cars, driver, up, shayne | driver, cars, parked, drive, garage, truck |
| church | fellowship, catholic, congregation, churches, christian, roman | fellowship, congregation, churches, worship, christian, catholic |
| money | pay, farming, taxes, amount, bills, income | chances, worries, checks, giffen, bills, trouble |
| war | world, communism, wars, russia, civil, communist | russian, lao, katanga, communists, civil, battle |
| university | harvard, college, graduate, portland, dartmouth, school | harvard, graduate, carleton, attended, college, dartmouth |
Neighbors are sensible and reflect the 1960s news corpus (Kennedy, Nixon, Katanga). Neighbors mix **similarity and relatedness** (car–driver, church–worship).
**Proxy quantitative check:** Spearman correlation between cosine similarity and WordNet path similarity over 3,997 random noun pairs: random vectors **0.013**, SGNS **0.073**, PPMI+SVD **0.127**. Both beat random, but absolute values are low — WordNet path similarity is a crude proxy, and a 1M-token corpus is tiny (published embeddings use billions of tokens). The count-based PPMI+SVD beat my short SGNS run here, consistent with Levy & Goldberg's finding that well-configured count methods are competitive on small corpora; I did not tune SGNS (epochs, dimensions, learning rate).

## 6. Using pretrained embeddings
Load GloVe/fastText/word2vec vectors, average (optionally TF-IDF-weighted) for document features, or initialize embedding layers. Typical gains over TF-IDF are modest for large labeled sets and larger for small ones; contextual encoders (N.8) usually dominate. fastText is a good default for morphologically rich languages and noisy text (OOV via subwords). Evaluate with word-similarity benchmarks (WordSim-353, SimLex-999), analogy sets, and — most importantly — the downstream task.

## 7. When to use / not
Use: small labeled data with unlabeled domain text, features for classical models, fast lightweight semantic search, initializing models, domain-specific embeddings trained on your corpus (legal, medical). Avoid: polysemy-heavy tasks (use contextual models), when a pretrained sentence encoder is available and compute is fine, or when bias would cause harm without auditing.

## 8. Hyperparameters
Dimension (50–300), window (small ⇒ syntactic similarity, large ⇒ topical), negative samples (5–20 small data), subsampling threshold (1e-5–1e-3), min count, epochs, learning rate schedule, `P_n` exponent 0.75; fastText: n-gram range 3–6. For PPMI+SVD: window weighting, context smoothing α=0.75, eigenvalue weighting exponent (0.5 here).

## 9. Implementation
```python
# From scratch core of one SGNS minibatch (see code/nlp_embeddings.py)
gp = 1/(1+np.exp(-(vc*uo).sum(1))) - 1; gn = 1/(1+np.exp(-np.einsum('bd,bkd->bk', vc, un)))
dvc = gp[:,None]*uo + np.einsum('bk,bkd->bd', gn, un)
np.add.at(Win, c, -lr*dvc); np.add.at(Wout, o, -lr*gp[:,None]*vc); np.add.at(Wout, n, -lr*gn[:,:,None]*vc[:,None,:])
# Library (gensim, on a machine where it installs):
# from gensim.models import Word2Vec, FastText
# w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=5, sg=1, negative=10, epochs=5)
```

## 10. Comparison
| | PPMI+SVD | word2vec | GloVe | fastText | Contextual (BERT) |
|---|---|---|---|---|---|
| Training | SVD of counts | SGD, streaming | Weighted LS on counts | SGD + subwords | Pretraining, expensive |
| OOV | No | No | No | **Yes** | Yes (subwords) |
| Polysemy | No | No | No | No | **Yes** |
| Cost | Cheap | Cheap | Cheap | Cheap | High |

## Interview Q&A
- *Why negative sampling?* Replaces an expensive softmax over the vocabulary with a few binary classifications.
- *word2vec vs GloVe?* Predictive online vs count-based regression on log co-occurrences; similar quality in practice.
- *How does fastText handle OOV?* Sums character n-gram vectors.
- *Limitations?* One vector per word (no polysemy), embeds corpus bias, no sentence composition.
- *Relation to SVD/LSA?* SGNS implicitly factorizes a shifted PMI matrix.

## Exercises
Train both embeddings on a domain corpus and compare neighbors; replace Brown with a larger corpus and recompute the WordNet-proxy correlation; implement analogy evaluation with exclusion of query words; measure gender-bias directions.

## Further reading
Mikolov et al. (2013a,b); Pennington, Socher & Manning (2014); Bojanowski et al. (2017); Levy & Goldberg (2014) "Neural word embedding as implicit matrix factorization"; Levy, Goldberg & Dagan (2015) "Improving distributional similarity with lessons learned from word embeddings"; Bolukbasi et al. (2016); Jurafsky & Martin Ch 6.
