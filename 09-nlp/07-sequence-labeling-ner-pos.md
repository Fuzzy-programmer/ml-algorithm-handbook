# Chapter N.7 — Sequence Labeling: POS Tagging, NER, Chunking (HMM → CRF → Neural), plus Parsing & Information Extraction

[← NLP index](README.md) · Prev [N.6](06-topic-modeling.md) · Next [N.8 Pretrained transformers](08-pretrained-transformers-for-nlp.md) · Priority 🟠 · Code: [`nlp_sequence_labeling.py`](../code/nlp_sequence_labeling.py) (executed), [`probabilistic_models.py`](../code/probabilistic_models.py) (HMM forward/Viterbi)

## 1. Identity & problem
Assign a label to **every token** in a sequence, where neighboring labels depend on each other. Tasks: **POS tagging** (NOUN, VERB…), **named-entity recognition** (PER/ORG/LOC), chunking (NP/VP), slot filling in dialogue, keyphrase extraction, medical/legal entity extraction, token-level toxicity/PII detection.

## 2. Lineage (and what each step fixed)
| Era | Model | Fixes | Limit |
|---|---|---|---|
| 1960s–90s | Rule/lexicon taggers (Brill 1992 transformation-based) | Interpretable | Manual rules |
| 1990s | **HMM** (Church 1988; Brants TnT 2000) | Probabilistic, Viterbi decoding | Generative, local features only, poor OOV |
| 1996–2000 | Maximum-entropy / MEMM (Ratnaparkhi 1996) | Rich overlapping features | **Label bias** |
| 2001 | **CRF** (Lafferty, McCallum & Pereira) | Globally normalized, no label bias, arbitrary features | Feature engineering |
| 2008–2015 | Neural taggers: Collobert & Weston (2011), **BiLSTM-CRF** (Huang et al. 2015; Lample et al. 2016) with char-CNN/LSTM | Learn features, OOV via characters | Needs data; slow vs linear |
| 2018+ | **Pretrained transformers** with token-classification heads (BERT, RoBERTa, XLM-R) | Context, transfer, multilingual | Compute; subword↔word alignment |
| 2020+ | LLM prompting/few-shot extraction, span/generative IE (UIE-style) | No labels, flexible schemas | Cost, consistency |

## 3. Mathematics
**HMM tagger:** `argmax_t Π P(t_i|t_{i−1}) P(w_i|t_i)`, decoded by **Viterbi** DP in `O(nT²)`; parameters by counting with smoothing; OOV words need a suffix/shape model.
**CRF (linear-chain):** `P(y|x) = (1/Z(x)) exp Σ_i [Σ_k λ_k f_k(y_{i−1},y_i,x,i)]`, `Z(x)` computed by the forward algorithm; trained by maximizing regularized conditional log-likelihood (L-BFGS); features: word identity, suffixes/prefixes, capitalization, neighbors, gazetteers, POS tags. Global normalization lets the model learn that `I-PER` cannot follow `B-ORG`.
**BIO tagging:** `B-X` begin, `I-X` inside, `O` outside: `Angela(B-PER) Merkel(I-PER) visited(O) Berlin(B-LOC)`. **Entity-level evaluation** counts an entity correct only if its span *and* type match exactly.
**Worked Viterbi example** (weather HMM from Ch 3.5): observations walk, shop, clean → best hidden path Sunny → Rainy → Rainy, probability 0.01344 (executed and verified by brute force over all 8 paths). POS tagging is the same algorithm with tags as hidden states and words as observations.
**Neural taggers:** contextual encoder → per-token softmax (or CRF layer over BiLSTM/transformer outputs). With wordpiece tokenization, label the first subword of each word and ignore the rest in the loss.

## 4. Executed results
### POS tagging — NLTK Penn Treebank sample, universal tagset (12 tags); 3,131 training / 783 test sentences, 20,039 test tokens; 8.5% of test tokens are unseen in training
| Model | Overall accuracy | Accuracy on OOV tokens (1,706) |
|---|---|---|
| Most-frequent-tag baseline (unknown → most common tag) | 0.9297 | 0.564 |
| HMM (bigram tag transitions) + Viterbi + suffix model for unknowns | 0.9434 | 0.725 |
| **CRF** (word/suffix/prefix/capitalization/digit/neighbor features) | **0.9705** | **0.883** |
Reading: the simple baseline is already 93% (ambiguity is limited with a coarse 12-tag set) — so always report it. Gains concentrate on **OOV words** (0.56 → 0.88) where features such as suffixes and capitalization matter; the CRF removes ~57% of the baseline's errors. A pretrained transformer reaches ~97–98% on the full Penn Treebank with the fine tagset; I did not run one here.
### NER — CoNLL-2002 Spanish (8,323 train / 1,517 test sentences), CRF with word, suffix, shape, POS and neighbor features
Entity-level micro **precision 0.793 · recall 0.781 · F1 0.787**; token accuracy 0.9709 — misleading, because 88.0% of tokens are `O` (always predicting `O` gets 0.880 accuracy and entity F1 = 0).
| Entity | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| PER | 0.842 | 0.878 | 0.859 | 735 |
| ORG | 0.780 | 0.806 | 0.793 | 1,400 |
| LOC | 0.805 | 0.772 | 0.788 | 1,084 |
| MISC | 0.664 | 0.494 | 0.567 | 340 |
Reading: MISC (heterogeneous category) is hardest; PER easiest (capitalized names). A BiLSTM-CRF or fine-tuned multilingual transformer typically improves these numbers substantially — run notebook `nlp_2_transformers.ipynb` §4 for a pretrained English NER demo; add your measured numbers below.

| Your results (notebook) | Value |
|---|---|
| Pretrained `dslim/bert-base-NER` qualitative output on your sentences | _run notebook §4_ |

## 5. Beyond tagging: parsing and information extraction (overview)
- **Dependency parsing:** head-dependent arcs (transition-based: Nivre 2003; graph-based: McDonald et al. 2005; neural biaffine: Dozat & Manning 2017). Evaluation: UAS/LAS. Used in relation extraction, grammar tools, linguistics; much less necessary for end-to-end neural pipelines.
- **Constituency parsing:** CKY/PCFG → neural span parsers. **Coreference resolution:** link mentions to entities ("Marie … she"); neural end-to-end models (Lee et al. 2017).
- **Relation extraction & event extraction:** classify relations between entity pairs ("Curie — born_in — Warsaw"), distant supervision (Mintz 2009), pretrained encoders with entity markers; **entity linking** maps mentions to knowledge-base IDs.
- **Open IE** extracts (subject, relation, object) triples without a fixed schema. Pipeline: tokenization → NER → entity linking → relation extraction → knowledge graph.

## 6. When to use / not
CRF with good features: small labeled sets, fast CPU inference, interpretable, strong baseline — still used in production for well-defined entities. Transformers: when you have moderate labeled data or a pretrained domain model (BioBERT, LegalBERT). LLM prompting: few labels or evolving schemas, with validation. Regex/gazetteers/rules: high-precision patterns (IDs, emails, dates). Hybrid is common: rules for structured entities + model for the rest.

## 7. Hyperparameters
CRF: `c1` (L1), `c2` (L2), features, `max_iterations`, algorithm. Neural: encoder size, char features, CRF layer, dropout, LR (2e-5–5e-5 for transformers), max length, label scheme (BIO vs BIOES), subword-label alignment policy.

## 8. Implementation (executed code pattern)
```python
import sklearn_crfsuite
def feats(s, i):
    w = s[i][0]
    f = {'bias': 1, 'w.lower': w.lower(), 'suf3': w[-3:], 'suf2': w[-2:], 'istitle': w.istitle(), 'isupper': w.isupper(), 'isdigit': w.isdigit()}
    if i: f.update({'-1:w': s[i-1][0].lower(), '-1:title': s[i-1][0].istitle()})
    else: f['BOS'] = True
    if i < len(s)-1: f['+1:w'] = s[i+1][0].lower()
    else: f['EOS'] = True
    return f
crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
crf.fit([[feats(s, i) for i in range(len(s))] for s in train], train_labels)
from seqeval.metrics import classification_report; print(classification_report(gold, crf.predict(X_test)))
```

## 9. Evaluation
Token accuracy (POS), **entity-level precision/recall/F1** (strict span+type; also report partial-match/relaxed variants), per-type metrics, confusion between types, error taxonomy (boundary errors, type errors, missed, spurious), out-of-domain test (news-trained NER collapses on tweets/clinical text), annotation consistency (inter-annotator F1 sets a ceiling). Use `seqeval`.

## 10. Pitfalls
Evaluating NER with token accuracy · leaking test sentences via document-level duplicates · inconsistent tag schemes (BIO vs IOB1) between data and evaluator · subword–word alignment mistakes · domain shift · nested/overlapping entities (flat taggers can't represent them) · annotation guideline drift.

## Interview Q&A
- *HMM vs CRF?* Generative with local independence vs discriminative, globally normalized, arbitrary overlapping features.
- *Why Viterbi?* Dynamic programming finds the best tag sequence in `O(nT²)` instead of `T^n`.
- *What is label bias?* Locally normalized models (MEMM) prefer states with few outgoing transitions; CRFs avoid it with global normalization.
- *Why not token accuracy for NER?* `O` dominates; use entity-level F1.
- *BIO vs BIOES?* BIOES adds explicit end/single tags, sometimes small gains.

## Exercises
Add gazetteer and cluster features to the CRF and measure F1; implement Viterbi for a tagger from scratch and verify against brute force; compute strict vs relaxed entity F1; fine-tune a transformer for NER on the same data (notebook extension).

## Further reading
Lafferty, McCallum & Pereira (2001); Jurafsky & Martin Ch 8 (Sequence labeling); Huang, Xu & Yu (2015); Lample et al. (2016); Devlin et al. (2018) NER results; Tjong Kim Sang (2002) CoNLL-2002; Sutton & McCallum (2012) *An Introduction to Conditional Random Fields*.
