# Chapter N.4 — Statistical Language Models: n-grams, Smoothing, Perplexity

[← NLP index](README.md) · Prev [N.3](03-text-classification-sentiment.md) · Next [N.5 Word embeddings](05-word-embeddings.md) · Priority 🟠 · Code: [`nlp_topics_lm.py`](../code/nlp_topics_lm.py) (executed)

## 1. Identity & problem
Assign probabilities to word sequences: `P(w₁…w_n)`. Used for speech recognition, machine translation (fluency), spelling/autocomplete, OCR, and as the conceptual ancestor of neural LMs and LLMs (Section 10).

## 2. History
Shannon (1948, 1951: entropy of English via n-gram approximations) → Markov assumption; Jelinek & IBM speech group (1970s–80s) → smoothing: Good–Turing (1953), Katz back-off (1987), Jelinek–Mercer interpolation, **Kneser–Ney** (1995; Chen & Goodman 1998 showed modified KN is best) → large-scale n-gram models (Brants et al. 2007, "stupid backoff") → neural LM (Bengio et al. 2003) → RNN LMs (Mikolov 2010) → Transformers (2017) → LLMs. Why neural models won: n-grams cannot share statistical strength between similar words/contexts and the table of contexts grows exponentially with n.

## 3. Mathematics
Chain rule: `P(w₁…w_n) = Π P(w_i | w_{<i})`. **Markov assumption** of order n−1: `P(w_i|w_{<i}) ≈ P(w_i | w_{i−n+1…i−1})`. MLE for a bigram: `P(w_i|w_{i−1}) = c(w_{i−1}w_i)/c(w_{i−1})`.
**Worked example.** Corpus: `<s> I am Sam </s>`, `<s> Sam I am </s>`, `<s> I do not like green eggs </s>`. `c(I)=3`, `c(<s> I)=2`, `c(I am)=2` ⇒ `P(I|<s>)=2/3`, `P(am|I)=2/3`, `P(Sam|am)=1/2`. So `P(<s> I am Sam </s>) = P(I|<s>)P(am|I)P(Sam|am)P(</s>|Sam) = 2/3·2/3·1/2·1/2 = 0.111`. Any unseen bigram (e.g. "I like") gets probability 0, so a single novel pair zeroes a whole sentence's probability → **smoothing is mandatory**.
**Smoothing methods**
- Add-k (Laplace): `(c(w₁w₂)+k)/(c(w₁)+kV)` — simple but gives too much mass to unseen events.
- Interpolation: `P = λP_bigram + (1−λ)P_unigram` (λ tuned on held-out data).
- Absolute discounting + **Kneser–Ney**: `P_KN(w|v) = max(c(vw)−D,0)/c(v) + D·N₁₊(v•)/c(v) · P_cont(w)`, where `P_cont(w) = N₁₊(•w)/N₁₊(••)` counts how many *distinct contexts* w follows. Intuition: "Francisco" is frequent but only after "San" — its continuation probability is low, so it shouldn't fill probability mass in other contexts.
- Back-off (Katz), stupid backoff (web scale).
**Perplexity:** `PP = exp(−(1/N) Σ log P(w_i|context))` — the weighted average branching factor; lower is better; compare only on identical tokenization/vocabulary and test data. Uniform over V gives PP = V.

## 4. Executed results (Brown "news", 90/10 sentence split, vocabulary = words with count ≥2 plus `<unk>`: **5,843** types; 10,496 test tokens)
| Model | Test perplexity |
|---|---|
| Unigram, add-1 | 303 |
| Bigram, MLE | ∞ (3,396 of 10,496 = **32.4%** of test bigrams never seen in training) |
| Bigram, add-1 | 747 |
| Bigram, interpolated with unigram (λ=0.7) | 184 |
| **Bigram, Kneser–Ney-style (D=0.75)** | **150** |
| Trigram, interpolated (0.5/0.3/0.2, untuned) | 246 |
Lessons: (1) a third of test bigrams are unseen even in a 100k-sentence-scale corpus — sparsity is the central problem; (2) **add-1 smoothing badly hurts bigrams** (747, worse than a unigram model at 303) because it moves too much probability mass to the 34M possible unseen pairs; (3) good smoothing (interpolation, KN) cuts perplexity 4–5× vs add-1; (4) the trigram did not beat the bigram **here** because the interpolation weights were fixed rather than tuned and the corpus is small (~1M tokens) — trigrams need far more data and proper KN/back-off. (Brown is small; numbers are for illustration of relative behavior, not benchmark-comparable.)

## 5. Using n-gram LMs
Sentence scoring/ranking (spell correction, ASR rescoring), autocomplete (argmax next word), language identification (character n-grams), text generation (sampling; produces locally fluent, globally incoherent text), feature extraction (perplexity as a fluency/domain-shift signal), data filtering (KenLM is widely used to filter web corpora). Tools: KenLM, NLTK `nltk.lm`, SRILM.

## 6. When to use / not
Use: fast, tiny, interpretable LM for rescoring, low-resource/embedded settings, language ID, data filtering, baselines and teaching. Avoid: long-range dependencies, semantics, generation quality — use neural LMs (Section 10).

## 7. Strengths / weaknesses
+ Trivial to train (counting), fast, no GPU, exact probabilities, well understood. − Sparse, exponential growth with n, no generalization across similar words/contexts, fixed short context, memory-heavy at scale.

## 8. Hyperparameters
n (order), vocabulary cutoff / `<unk>` policy, smoothing method and discount D, interpolation weights (fit by EM on held-out data), pruning thresholds.

## 9. Implementation (core of the executed code)
```python
import math, collections
u = collections.Counter(w for s in train for w in s); b = collections.Counter((s[i], s[i+1]) for s in train for i in range(len(s)-1))
bc = collections.Counter(s[i] for s in train for i in range(len(s)-1))
cont = collections.Counter(w2 for (w1, w2) in b); n_types = len(b); fol = collections.Counter(w1 for (w1, w2) in b)
def p_kn(w1, w2, D=0.75):
    pc = cont[w2] / n_types
    return pc if bc[w1] == 0 else max(b[(w1, w2)] - D, 0) / bc[w1] + D * fol[w1] / bc[w1] * pc
ppl = math.exp(-sum(math.log(p_kn(s[i-1], s[i])) for s in test for i in range(1, len(s))) / n_test_tokens)
```

## 10. Evaluation
Intrinsic: perplexity (held-out, same vocabulary). Extrinsic: downstream WER (speech), BLEU (MT), accuracy of spelling correction. Beware: perplexity is not comparable across different tokenizations; `<unk>` handling changes values dramatically.

## Interview Q&A
- *What is perplexity?* Exponentiated average negative log-likelihood per token; effective number of equally likely choices.
- *Why smoothing?* Unseen n-grams otherwise get probability 0.
- *Why Kneser–Ney works better than add-1?* It redistributes mass according to how many contexts a word can follow, not its raw frequency.
- *Limitations vs neural LMs?* No generalization across similar words, short context, sparse counts.

## Exercises
Implement stupid backoff and compare with interpolation; tune λ on a validation set; compute character-level n-gram perplexity for two languages and use it for language ID; generate text from bigram vs trigram and compare coherence.

## Further reading
Jurafsky & Martin Ch 3 (N-gram language models); Chen & Goodman (1998) "An empirical study of smoothing techniques for language modeling"; Kneser & Ney (1995); Heafield (2011) KenLM; Bengio et al. (2003).
