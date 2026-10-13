# Chapter N.1 — Text Preprocessing and Tokenization

[← NLP index](README.md) · Next [N.2 Text representation](02-text-representation.md) · Priority 🔴 · Code: [`nlp_text_basics.py`](../code/nlp_text_basics.py) (executed) · Notebook: [`nlp_1_classical.ipynb`](../notebooks/nlp_1_classical.ipynb)

## 1. Identity & problem
Text is a variable-length string; models need discrete units. **Preprocessing** cleans and normalizes text; **tokenization** splits it into units (words, subwords, characters, bytes). Choices here silently determine vocabulary size, out-of-vocabulary (OOV) rate, sparsity and downstream accuracy.

## 2. Historical background
Rule-based tokenizers and sentence splitters (Penn Treebank conventions) → statistical segmenters (Punkt, Kiss & Strunk 2006) → stemmers (Lovins 1968; **Porter 1980**) and lexicon-based lemmatization (WordNet) → subword units to kill OOV: **BPE** (Gage 1994; applied to NMT by Sennrich et al. 2016), **WordPiece** (Schuster & Nakajima 2012; BERT), **Unigram LM / SentencePiece** (Kudo 2018) → **byte-level BPE** (GPT-2) — no unknown characters. Modern LLM pipelines mostly skip stemming/stopword removal and rely on subwords.

## 3. Pipeline stages and decisions
| Stage | Options | Notes |
|---|---|---|
| Cleaning | Unicode normalization (NFC/NFKC), strip HTML/markup, fix encodings, remove boilerplate | Do not over-clean: punctuation, case and emoji often carry sentiment/entity signal |
| Case | Lowercase vs keep | Lowercasing merges "Apple/apple" (entity loss); keep case for NER/cased models |
| Sentence splitting | Rules / Punkt / learned | Abbreviations break naive splitting (see below) |
| Tokenization | Whitespace, regex, rule-based (Treebank), subword (BPE/WordPiece/Unigram), char, byte | Language-dependent: Chinese/Japanese/Thai have no spaces; German compounds; Arabic clitics |
| Normalization | Stemming (crude suffix stripping), lemmatization (dictionary form with POS) | Stemming faster; lemmatization linguistically cleaner |
| Stopwords | Remove very frequent function words | Helps bag-of-words noise; **hurts** negation/phrases and is unnecessary for neural models |
| Filtering | min_df/max_df, length limits, numbers/URLs → placeholders | Control vocabulary size |

## 4. Executed examples
**Tokenizer contrast** on `Dr. Smith didn't pay $3.50 for U.S. e-mail; it's well-known... #NLP @user https://x.co`:
- whitespace: `['Dr.', 'Smith', "didn't", 'pay', '$3.50', 'for', 'U.S.', 'e-mail;', "it's", 'well-known...', '#NLP', '@user', 'https://x.co']` — punctuation glued to words
- regex `\w+`: `['Dr','Smith','didn','t','pay','3','50',…,'e','mail',…,'https','x','co']` — splits `didn't`, `3.50`, URL (destroys structure)
- NLTK `word_tokenize`: `['Dr.','Smith','did',"n't",'pay','$','3.50','for','U.S.','e-mail',';','it',"'s",'well-known','...','#','NLP','@','user','https',':','//x.co']` — handles clitics and abbreviations but still splits the URL/hashtag.
No tokenizer is "correct": social-media text needs a tweet-aware tokenizer; URLs may be normalized to a placeholder.
**Sentence splitting:** `"Dr. Smith arrived at 5 p.m. He left. Mr. Lee stayed."` → Punkt gives `['Dr. Smith arrived at 5 p.m.', 'He left.', 'Mr. Lee stayed.']` — handled abbreviations correctly here; splitting on `". "` would have produced 5 fragments.
**Stemming vs lemmatization:**
| word | Porter stem | lemma (verb) | lemma (noun) |
|---|---|---|---|
| running | run | run | running |
| studies | studi | study | study |
| better | better | better | better |
| happily | happili | happily | happily |
| universities | univers | universities | university |
| was | wa | be | wa |
Stems need not be words (`studi`, `happili`, `wa`); lemmatization depends on POS (`was` → `be` only as a verb); neither converts `better` → `good` (needs the adjective POS tag and WordNet exceptions).

## 5. Statistics of text: Zipf's and Heaps' laws (executed on the Brown corpus)
Brown corpus: 981,716 alphabetic tokens, **40,234 types**.
- **Zipf's law:** frequency ∝ 1/rank, i.e. rank × frequency ≈ constant.
| Rank | Word | Frequency | rank × freq |
|---|---|---|---|
| 1 | the | 69,971 | 69,971 |
| 10 | he | 9,548 | 95,480 |
| 100 | well | 897 | 89,700 |
| 1,000 | built | 103 | 103,000 |
| 10,000 | generalizations | 7 | 70,000 |
- **Hapax legomena (words seen once): 15,367 = 38.2% of all types** — a huge long tail, the reason OOV words are unavoidable with word-level vocabularies.
- **Heaps' law:** vocabulary grows sublinearly with corpus size: first 10,000 tokens → 2,470 types; 100,000 → 12,235; 500,000 → 30,049; all 981,716 → 40,234. Vocabulary never saturates.
Consequences: (1) a few hundred words cover most tokens, (2) most types are rare, (3) count-based estimates for rare events are unreliable → smoothing (N.4), subwords, embeddings (N.5).

## 6. Subword tokenization
**BPE:** start from characters, repeatedly merge the most frequent adjacent pair until reaching the target vocabulary size (worked example with merges in [Section 10, Ch G.2](../10-genai-and-llms/02-language-models-llms.md)). **WordPiece** chooses merges maximizing corpus likelihood; **Unigram LM** starts with a large vocabulary and prunes by likelihood loss and can sample multiple segmentations (regularization). Executed with `bert-base-uncased`:
- `unbelievably tokenization is fun` → `un ##bel ##ie ##va ##bly token ##ization is fun`
- `COVID-19 vaccines` → `co ##vid - 19 vaccines`
- `ChatGPT hallucinates pseudoscience` → `chat ##gp ##t hall ##uc ##inates pseudo ##sc ##ience`
- `hello world` → ids `[101, 7592, 2088, 102]` = `[CLS] hello world [SEP]`
Observations: new words split into pieces (no OOV), but pieces need not be morphemes; vocabulary of a model trained before 2020 splits "COVID" awkwardly. **Always use the tokenizer that matches the pretrained model.**

## 7. When to use / not
| Situation | Recommendation |
|---|---|
| Classical bag-of-words | Light cleaning, lowercase, keep negations; stopword removal optional (test it) |
| Neural/pretrained models | Use the model's own tokenizer; minimal cleaning |
| Morphologically rich or low-resource languages | Subwords / character models; lemmatizers with language-specific tools |
| Social media | Tweet-aware tokenizer, normalize URLs/@mentions, keep emoji/hashtags |
| Languages without spaces | Segmenters (jieba, MeCab) or character n-grams/subwords |
Avoid: aggressive stemming for tasks needing precision (search for names), lowercasing for NER, removing stopwords for tasks where function words matter (authorship, negation).

## 8. Pitfalls
Train/serve tokenizer mismatch · fitting vocabulary on test data · dropping negation ("not good" → "good") through stopword removal · unicode confusables · truncating long texts silently (transformer max length 512 tokens) · leaking PII into vocabularies.

## 9. Implementation
```python
import nltk, re
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.word_tokenize(text); nltk.sent_tokenize(text)
PorterStemmer().stem("studies")              # 'studi'
WordNetLemmatizer().lemmatize("was", "v")    # 'be'
from transformers import AutoTokenizer       # subword (see notebook nlp_2)
AutoTokenizer.from_pretrained("bert-base-uncased").tokenize("tokenization")
```
spaCy (`spacy.load("en_core_web_sm")`) provides a fast rule-based tokenizer, lemmatizer, POS tagger and NER pipeline.

## 10. Evaluation of preprocessing
Treat each choice as a hyperparameter: ablate with cross-validation on the downstream task (N.3 shows stopword removal and stemming did **not** help sentiment accuracy with TF-IDF+LR: 0.859 → 0.855 / 0.854). Track vocabulary size, OOV rate on validation, average sequence length (cost), and error examples.

## Interview Q&A
- *Stemming vs lemmatization?* Rule-based suffix chopping (fast, may give non-words) vs dictionary/POS-aware base forms (slower, cleaner).
- *Why subword tokenization?* Fixed-size vocabulary with no OOV; shares statistics across related words.
- *State Zipf's law and its implication.* Frequency ∝ 1/rank; heavy tail means most types are rare → sparsity.
- *Should you remove stopwords for BERT?* No.
- *Why can sentence splitting fail?* Abbreviations, decimals, quotations.

## Exercises
Train BPE on your own corpus and inspect merges; plot the Zipf log-log curve for Brown vs a tweet corpus; measure OOV rate vs vocabulary size cutoffs; compare lowercasing vs cased features for NER (N.7).

## Further reading
Jurafsky & Martin, *Speech and Language Processing*, Ch 2 (Regular expressions, text normalization); Sennrich et al. (2016); Kudo & Richardson (2018) SentencePiece; Porter (1980); Manning, Raghavan & Schütze, *Introduction to Information Retrieval* Ch 2.
