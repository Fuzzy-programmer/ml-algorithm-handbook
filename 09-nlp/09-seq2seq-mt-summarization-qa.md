# Chapter N.9 — Sequence-to-Sequence NLP: Machine Translation, Summarization, Question Answering and Generation Metrics

[← NLP index](README.md) · Prev [N.8](08-pretrained-transformers-for-nlp.md) · Next [N.10 Evaluation & practice](10-nlp-evaluation-practice.md) · Priority 🟠 · Notebook: [`nlp_2_transformers.ipynb`](../notebooks/nlp_2_transformers.ipynb) §3, §7 (**not run by the author**; the hand-worked metric examples below were computed and verified)

## 1. Identity & problem
Map an input text sequence to an output text sequence of different length: translation, summarization, paraphrase/simplification, data-to-text, dialogue, question answering (extractive or generative), code generation. Defining difficulty: many valid outputs, so evaluation is hard.

## 2. Lineage
| Era | Machine translation | Others |
|---|---|---|
| 1950s–80s | Rule-based (direct, transfer, interlingua) | Rule/template generation |
| 1990–2013 | **Statistical MT**: IBM Models 1–5 (Brown et al. 1993) word alignment; **phrase-based** (Koehn et al. 2003; Moses) | Extractive summarization (TextRank 2004), IR-based QA |
| 2014 | **Seq2seq** encoder–decoder RNNs (Sutskever; Cho) → **attention** (Bahdanau 2014; Luong 2015) fixes the fixed-vector bottleneck | Pointer-generator summarizers (See et al. 2017) |
| 2016 | GNMT (Wu et al.) production NMT; subword units (BPE) solve rare words | SQuAD (2016) extractive QA with BiDAF |
| 2017 | **Transformer** (Vaswani) — parallel, SOTA | |
| 2018–20 | BERT readers for QA; pretrained seq2seq **BART, T5, mBART, PEGASUS** for summarization/translation/generation | Multilingual NMT (M2M-100, NLLB) |
| 2020–now | LLMs for translation/summarization/QA, RAG for open-domain QA (Ch G.4) | Evaluation by LLM judges |

## 3. Core techniques
- **Encoder–decoder with attention:** decoder step t computes attention over encoder states, `c_t = Σ α_{ti} h_i`, `p(y_t|y_{<t},x) = softmax(W[s_t; c_t])`; trained with cross-entropy and **teacher forcing**; **exposure bias** arises because at test time the model conditions on its own outputs.
- **Decoding:** greedy; **beam search** (keep top-k partial hypotheses; length normalization; beam 4–5 typical for MT); sampling (temperature, top-p) for open-ended generation; constrained decoding. Beam search can favor generic or truncated outputs, and very large beams can *reduce* quality.
- **Subword vocabularies** (BPE/SentencePiece) shared across languages.
- **Summarization:** extractive (select sentences) vs abstractive (generate); faithfulness (hallucination) is the main risk; long inputs need chunking or long-context models.
- **QA:** *extractive* (predict start/end span in a passage; SQuAD v1.1/v2.0 with unanswerable questions), *generative* (T5/LLM writes the answer), *open-domain* (retrieve then read; RAG), *multi-hop* (HotpotQA), *conversational*.
- **Pretraining objectives for seq2seq:** denoising autoencoding (BART), span corruption (T5), gap-sentence generation (PEGASUS).

## 4. Evaluation metrics — with worked examples
**BLEU** (Papineni et al. 2002): geometric mean of modified n-gram precisions (n=1..4) times a brevity penalty `BP = min(1, exp(1 − r/c))`.
**Worked example (verified).** Hypothesis: *the cat is on the mat* (6 tokens). Reference: *there is a cat on the mat* (7 tokens).
- 1-grams: clipped matches = {the:1 (ref has one "the"), cat, is, on, mat} = 5 of 6 ⇒ `p₁ = 0.833`
- 2-grams: matches "on the", "the mat" = 2 of 5 ⇒ `p₂ = 0.400`
- 3-grams: "on the mat" = 1 of 4 ⇒ `p₃ = 0.250`
- 4-grams: 0 of 3 ⇒ `p₄ = 0`, so unsmoothed BLEU-4 = **0** even though the sentence is a decent paraphrase — why sentence-level BLEU needs smoothing and BLEU is meant for corpus-level evaluation.
- Brevity penalty: `exp(1 − 7/6) = 0.846`. BLEU-2 `= 0.846 × √(0.833×0.400) = 0.489`.
**ROUGE** (Lin 2004; summarization): ROUGE-1 recall = unigram overlap / reference unigrams; same pair: overlap 5, so recall `5/7 = 0.714`, precision `5/6 = 0.833`, F1 `= 0.769`. ROUGE-L uses the longest common subsequence; ROUGE-2 bigram overlap.
**chrF** (character n-gram F-score): robust for morphologically rich languages; **METEOR** (stems/synonyms); **BERTScore** (cosine of contextual embeddings; correlates better with humans for paraphrase); **COMET/BLEURT** (learned metrics trained on human judgments; best correlation for MT); **QA:** Exact Match and token-level F1 (SQuAD); **faithfulness** metrics (NLI-based consistency, QAGS/SummaC, LLM judges); human evaluation (fluency, adequacy, MQM error annotation).
**Pitfalls:** BLEU values depend on tokenization/casing — report **sacreBLEU** signatures; n-gram metrics under-reward valid paraphrases and can't detect hallucinated facts; improvements of <1 BLEU are often noise (use paired bootstrap significance, Koehn 2004).

## 5. What to run (notebook §7) and your results
T5-small English→French on six sentences with references, beam search 4:
| Metric | Your result |
|---|---|
| Corpus BLEU (sacreBLEU) | _run notebook §7_ |
| chrF | _run notebook §7_ |
The notebook also reproduces the hand-computed example with sacreBLEU (unsmoothed sentence BLEU-4 = 0) and ROUGE via `rouge_score`. Six sentences is far too few for conclusions — use ≥500 sentences (e.g. WMT test sets) for real evaluation. Extractive QA (notebook §3): answerable questions get correct spans; the unanswerable question still returns a span — add a no-answer threshold or use a SQuAD-v2 model.

## 6. When to use / not
Use dedicated seq2seq/NMT systems for: high-volume translation with latency/cost limits, domain-adapted translation (fine-tune on parallel data), summarization of known formats with fine-tuned models. Use LLMs when: flexible instructions/styles, low-resource tasks via prompting, multi-step transformations — with verification. Avoid: abstractive summarization of high-stakes content (legal/medical) without faithfulness checks and human review; BLEU-only evaluation; translating without handling named entities/terminology (use glossaries, constrained decoding).

## 7. Hyperparameters
Beam size, length penalty, max length, repetition/no-repeat-ngram constraints, temperature/top-p (sampling), label smoothing, learning rate/warmup, batch size in tokens, vocabulary size, dropout; for summarization: input truncation strategy, minimum/maximum summary length.

## 8. Implementation skeleton (version-robust)
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tok = AutoTokenizer.from_pretrained('t5-small'); m = AutoModelForSeq2SeqLM.from_pretrained('t5-small').eval()
enc = tok(['translate English to French: The weather is nice today.'], return_tensors='pt', padding=True)
out = m.generate(**enc, max_new_tokens=64, num_beams=4)
print(tok.batch_decode(out, skip_special_tokens=True))
import sacrebleu; print(sacrebleu.corpus_bleu(hyps, [refs]).score)
```
(Newer `transformers` releases removed some high-level task pipelines; calling `generate` directly is stable across versions.)

## 9. Failure modes
Hallucinated content in summaries (faithfulness), omissions, repetition/loops, over-literal translation, mistranslated names/numbers, gender bias in translation (pronoun defaults), low-resource language quality, exposure bias, length bias (beam search favors short outputs without length normalization).

## 10. Case study (template)
Support-ticket summarization: baseline = extractive lead-3/TextRank; model = fine-tuned BART/T5 or prompted LLM; evaluate ROUGE + faithfulness checks (entity/number consistency) + human rating of 100 samples; ship with citation of source sentences and fallbacks to extractive summaries when confidence is low.

## Interview Q&A
- *What is attention in seq2seq?* A learned soft alignment letting the decoder look at all encoder states, removing the single-vector bottleneck.
- *What is exposure bias?* Training on gold prefixes vs inference on own predictions.
- *Why does BLEU have a brevity penalty?* Precision alone rewards very short outputs.
- *BLEU vs BERTScore vs COMET?* n-gram overlap vs embedding similarity vs trained quality estimator.
- *Extractive vs abstractive summarization?* Select vs generate; abstractive can hallucinate.
- *Beam search vs sampling?* Maximization (translation) vs diversity (creative generation).

## Exercises
Recompute the BLEU example with smoothing (add-1) and compare with sacreBLEU's `exp` smoothing; implement beam search for a toy LM; evaluate T5-small vs a larger model on 200 WMT sentences with paired bootstrap; measure ROUGE vs human faithfulness ratings on 30 summaries.

## Further reading
Brown et al. (1993); Koehn, Och & Marcu (2003); Bahdanau, Cho & Bengio (2014); Vaswani et al. (2017); Papineni et al. (2002) BLEU; Lin (2004) ROUGE; Post (2018) "A call for clarity in reporting BLEU scores" (sacreBLEU); Rei et al. (2020) COMET; Rajpurkar et al. (2016, 2018) SQuAD; Lewis et al. (2020) BART; Raffel et al. (2020) T5.
