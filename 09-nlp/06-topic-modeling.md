# Chapter N.6 — Topic Modeling: LSA, NMF, LDA

[← NLP index](README.md) · Prev [N.5](05-word-embeddings.md) · Next [N.7 Sequence labeling](07-sequence-labeling-ner-pos.md) · Priority 🟡 · Code: [`nlp_topics_lm.py`](../code/nlp_topics_lm.py) (executed)

## 1. Identity & problem
Discover latent **themes** in an unlabeled collection and represent each document as a mixture of topics. Used for corpus exploration, search/browse, trend tracking, feature generation, and summarizing large archives (news, support tickets, papers, reviews).

## 2. Lineage
LSA/LSI (Deerwester et al. 1990; SVD) → **pLSA** (Hofmann 1999; probabilistic, but overfits, no generative model for new docs) → **LDA** (Blei, Ng & Jordan 2003; Dirichlet priors give a proper generative model) → NMF for topics (Lee & Seung 1999; Arora et al. 2012 theory), hierarchical/dynamic/correlated topic models (HDP 2006; Dynamic TM 2006; CTM 2007) → neural/embedding-based topic modeling (ETM 2019; BERTopic 2022 — embed documents, reduce dimension, cluster with HDBSCAN, label with class-based TF-IDF). Not a strict ladder: LDA's probabilistic semantics vs NMF's speed/stability vs embedding-clustering's semantic quality.

## 3. Intuition
A document is a mixture of topics; a topic is a distribution over words. LDA generative story: for each document draw topic proportions θ ~ Dirichlet(α); for each word draw a topic z ~ θ then a word w ~ φ_z (φ_k ~ Dirichlet(β)). Inference runs this backwards.

## 4. Mathematics
- **LSA:** `X ≈ U_kΣ_kV_kᵀ` — dimensions are signed, hard to read as topics.
- **NMF:** `X ≈ WH`, `W,H ≥ 0` (docs × topics, topics × words) — additive parts, readable topics; minimize Frobenius or KL divergence.
- **LDA:** joint `p(θ,z,w|α,β) = p(θ|α) Π_n p(z_n|θ) p(w_n|z_n,β)`. Inference: variational Bayes (batch/online, Hoffman et al. 2010) or collapsed Gibbs sampling. α small ⇒ sparse topic mixtures per document; β small ⇒ sparse word distributions per topic.
**Worked example.** Two topics: A = {game 0.5, team 0.4, car 0.1}, B = {car 0.6, engine 0.3, game 0.1}. A document with θ = (0.7 A, 0.3 B) has word probabilities `P(game) = 0.7·0.5+0.3·0.1 = 0.38`, `P(car)=0.7·0.1+0.3·0.6 = 0.25`, `P(team)=0.28`, `P(engine)=0.09`. Given observed counts the posterior over θ shifts toward A if "game/team" dominate.
**Model selection:** choose K by held-out perplexity or likelihood, coherence (NPMI/UMass/C_v correlate better with human judgments than perplexity — Chang et al. 2009 "Reading tea leaves"), stability across seeds, and interpretability for the use case.

## 5. Executed results (6 newsgroups: space, autos, politics, graphics, hockey, medicine; ~3.4k posts)
**NMF topics (TF-IDF):** (0) `people don just think like know make time` — generic chatter; (1) `geb dsl cadre shameful n3jxp chastity skepticism intellect` — a single prolific poster's signature quotes (garbage topic); (2) `game team hockey players play games season nhl` ✔ hockey; (3) `thanks graphics files does know image mail file` ✔ graphics; (4) `space nasa shuttle launch orbit earth lunar station` ✔ space; (5) `car cars dealer engine new price good like` ✔ autos.
**LDA topics (counts):** (0) `president think people government time mr work going` ✔ politics; (1) `image use graphics edu software file like does` ✔ graphics; (2) `10 25 11 55 12 16 14 18` — numbers/stats tables (junk); (3) `space nasa data launch center 1993 information university` ✔ space; (4) `don just people think like know game did` — mixed chatter/hockey; (5) `car edu com like just cars don new` ✔ autos.
Assigning each document to its top LDA topic and comparing with the true category: **purity 0.541** (chance ≈ 0.17) — much better than random, far from clean. **Held-in perplexity** for K = 2/6/12: 3,320 / 2,799 / 2,646 (always decreases with K on training data, so it cannot choose K; use held-out data). LSA's first components each explain <1% variance.
Honest reading: 3–4 of 6 topics per model align with real categories; others are **boilerplate or chatter topics**, typical when preprocessing is minimal (no thread/header cleaning, no bigram phrases, limited tuning). Improving results = better stopword/min_df/max_df filters, bigrams, more iterations, tuned α/β, K sweep with coherence, removing signatures/quotes.

## 6. When to use / not
Use: exploring unfamiliar corpora, generating candidate categories to label, dimensionality reduction for interpretable features, monitoring theme prevalence over time. Avoid: short texts (tweets) with classical LDA (sparse; use biterm/embedding methods), expecting topics to equal human categories, final classification (supervised models win), tasks needing precise semantics.

## 7. Strengths / weaknesses
+ Unsupervised, interpretable, scalable (online LDA), probabilistic (LDA). − Choosing K, instability across runs, junk topics, bag-of-words (no order), sensitive to preprocessing, evaluation is subjective.

## 8. Hyperparameters
K, α (doc-topic prior), β/η (topic-word prior), learning method/iterations, vocabulary filters (`min_df`, `max_df`, stopwords, n-grams), random seed; NMF: init (`nndsvd`), regularization, loss.

## 9. Implementation
```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
cv = CountVectorizer(stop_words='english', max_df=.5, min_df=5); C = cv.fit_transform(docs)
lda = LatentDirichletAllocation(n_components=6, learning_method='batch', max_iter=25, random_state=0).fit(C)
def top_words(model, names, n=8): return [' '.join(names[i] for i in comp.argsort()[::-1][:n]) for comp in model.components_]
nmf = NMF(6, init='nndsvd', random_state=0, max_iter=400).fit(TfidfVectorizer(stop_words='english', max_df=.5, min_df=5).fit_transform(docs))
```
Libraries: gensim (LDA + coherence), BERTopic, Top2Vec, tomotopy.

## 10. Evaluation
Held-out likelihood/perplexity, topic coherence (NPMI), topic diversity, human evaluation (word intrusion, topic intrusion tasks), downstream usefulness (does topic-augmented retrieval/classification improve?), stability (Jaccard overlap of top words across seeds). With labels: purity/NMI/ARI as a sanity check only.

## Interview Q&A
- *LDA vs LSA?* Probabilistic, non-negative, Dirichlet-regularized mixtures vs linear algebra decomposition with signed factors.
- *How choose K?* Coherence and held-out likelihood sweeps plus human inspection; no single right answer.
- *What do α and β control?* Sparsity of document–topic and topic–word distributions.
- *Why do junk topics appear?* Boilerplate/frequent patterns form coherent but uninteresting clusters; preprocess or add as stopwords.
- *Modern alternative?* Embed documents, cluster (HDBSCAN), extract topic words with class-based TF-IDF (BERTopic).

## Exercises
Re-run with bigrams and a custom stopword list and re-measure purity; sweep K with coherence; compare NMF vs LDA stability across 5 seeds; try the embedding-clustering approach using the notebook's MiniLM model.

## Further reading
Blei, Ng & Jordan (2003) *JMLR*; Blei (2012) "Probabilistic topic models" *CACM*; Chang et al. (2009); Hoffman, Blei & Bach (2010) online LDA; Röder et al. (2015) coherence measures; Grootendorst (2022) BERTopic.
