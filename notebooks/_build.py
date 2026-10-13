"""Builds the .ipynb files. Run: python _build.py   (notebooks are for Google Colab / any Jupyter)."""
import json, pathlib
CODE = pathlib.Path(__file__).resolve().parent.parent / "code"
def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True)}
def save(name, cells):
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
    (pathlib.Path(__file__).resolve().parent / name).write_text(json.dumps(nb, indent=1)); print("wrote", name)
def script(fn): return (CODE / fn).read_text()

# ---------------- Notebook 1: classical NLP (CPU is fine) ----------------
save("nlp_1_classical.ipynb", [
 md("# NLP handbook — Part 1: classical NLP (tokenization, TF-IDF, classification, topics, n-gram LMs, embeddings, POS/NER)\nRuns on CPU in Colab (~10-20 min total). Each section corresponds to a chapter in `09-nlp/`. Numbers in the chapters came from running these exact cells."),
 code("!pip -q install nltk scikit-learn scipy sklearn-crfsuite seqeval\nimport nltk\nfor p in ['punkt','punkt_tab','treebank','movie_reviews','conll2002','stopwords','wordnet','averaged_perceptron_tagger_eng','universal_tagset','brown']: nltk.download(p, quiet=True)"),
 md("## 1. Tokenization, normalization, Zipf/Heaps, TF-IDF, sentiment classification ablations (chapters N.1-N.3)"), code(script("nlp_text_basics.py")),
 md("## 2. Topic models and n-gram language models (chapters N.4, N.6)"), code(script("nlp_topics_lm.py")),
 md("## 3. Word embeddings: PPMI+SVD and skip-gram negative sampling from scratch (chapter N.5)\nSGNS takes ~2-3 min on CPU."), code(script("nlp_embeddings.py")),
 md("## 4. Sequence labeling: HMM / CRF POS tagging and CRF NER (chapter N.7)"), code(script("nlp_sequence_labeling.py")),
])

# ---------------- Notebook 2: transformers (use a GPU runtime) ----------------
save("nlp_2_transformers.ipynb", [
 md("# NLP handbook — Part 2: pretrained transformers for NLP\n**Runtime -> Change runtime type -> GPU (T4)** in Colab. These cells were written for chapters N.8-N.10 and have **not** been executed by the author; fill in your own numbers in the chapters' result tables. Pinned-free installs: if an API changed, check the library docs."),
 code("!pip -q install transformers datasets sentence-transformers sacrebleu rouge-score evaluate seqeval accelerate scikit-learn nltk\nimport torch, numpy as np, warnings; warnings.filterwarnings('ignore')\ndev = 'cuda' if torch.cuda.is_available() else 'cpu'; print('device:', dev)"),
 md("## 1. WordPiece tokenization and special tokens"),
 code("""from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('bert-base-uncased')
for s in ['unbelievably tokenization is fun', 'COVID-19 vaccines', 'ChatGPT hallucinates pseudoscience']:
    print(s, '->', tok.tokenize(s))
ids = tok('hello world')['input_ids']; print(ids, tok.convert_ids_to_tokens(ids))"""),
 md("## 2. Static vs contextual embeddings: the word *bank*\nWe compare the same word in finance vs river contexts at layer 0 (static), 8 and 12. Expect within-sense cosine > between-sense cosine in upper layers."),
 code("""from transformers import AutoModel
import itertools
m = AutoModel.from_pretrained('bert-base-uncased', output_hidden_states=True).eval().to(dev)
fin = ['she deposited cash at the bank', 'the bank approved my loan', 'he works at a bank downtown', 'the bank raised interest rates']
riv = ['they sat on the river bank', 'the muddy bank of the stream', 'fish swam near the bank of the lake', 'she walked along the grassy bank']
def vecs(sent, word):
    enc = tok(sent, return_tensors='pt').to(dev)
    with torch.no_grad(): hs = m(**enc).hidden_states
    i = tok.convert_ids_to_tokens(enc['input_ids'][0]).index(word); return {L: hs[L][0, i].cpu().numpy() for L in (0, 8, 12)}
V = {'fin': [vecs(x, 'bank') for x in fin], 'riv': [vecs(x, 'bank') for x in riv]}
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
for L in (0, 8, 12):
    within = np.mean([cos(x[L], y[L]) for g in V.values() for x, y in itertools.combinations(g, 2)])
    between = np.mean([cos(x[L], y[L]) for x in V['fin'] for y in V['riv']])
    print(f'layer {L:2d}: within-sense {within:.3f} | between-sense {between:.3f} | gap {within-between:+.3f}')"""),
 md("## 3. Masked language modeling (fill-mask) and extractive QA (manual span decoding; works across transformers versions)"),
 code("""from transformers import AutoModelForMaskedLM, AutoModelForQuestionAnswering
mlm = AutoModelForMaskedLM.from_pretrained('bert-base-uncased').eval().to(dev)
def fill(s, k=3):
    enc = tok(s, return_tensors='pt').to(dev); pos = (enc['input_ids'][0] == tok.mask_token_id).nonzero()[0].item()
    with torch.no_grad(): p = mlm(**enc).logits[0, pos].softmax(-1)
    t = p.topk(k); return [(tok.decode([i]), round(v.item(), 3)) for v, i in zip(t.values, t.indices)]
for s in ['The capital of France is [MASK].', 'She was [MASK] to hear the good news.']: print(s, fill(s))

qtok = AutoTokenizer.from_pretrained('distilbert-base-cased-distilled-squad'); qm = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad').eval().to(dev)
ctx = 'Marie Curie won the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911. She was born in Warsaw and worked in Paris.'
def answer(q, max_len=30):
    enc = qtok(q, ctx, return_tensors='pt').to(dev)
    with torch.no_grad(): o = qm(**enc)
    seq = enc.sequence_ids(0); mask = torch.tensor([x == 1 for x in seq], device=dev)
    sl = o.start_logits[0].masked_fill(~mask, -1e9); el = o.end_logits[0].masked_fill(~mask, -1e9)
    sc = sl[:, None] + el[None, :]; n = sc.shape[0]; valid = torch.triu(torch.ones(n, n, device=dev)) * (1 - torch.triu(torch.ones(n, n, device=dev), max_len))
    sc = sc.masked_fill(valid == 0, -1e9); i, j = divmod(int(sc.argmax()), n); return qtok.decode(enc['input_ids'][0][i:j + 1])
for q in ['When did Marie Curie win the Chemistry prize?', 'Where was she born?', 'Who discovered penicillin?']: print(q, '->', repr(answer(q)))
print('Note the unanswerable question still gets a span: SQuAD-v1 models cannot abstain.')"""),
 md("## 4. Pretrained NER (token classification) vs the CRF baseline"),
 code("""from transformers import AutoModelForTokenClassification
ntok = AutoTokenizer.from_pretrained('dslim/bert-base-NER'); nm = AutoModelForTokenClassification.from_pretrained('dslim/bert-base-NER').eval().to(dev)
text = 'Angela Merkel met Satya Nadella of Microsoft in Berlin on Monday.'
enc = ntok(text, return_tensors='pt').to(dev)
with torch.no_grad(): pred = nm(**enc).logits[0].argmax(-1).cpu().tolist()
for t, p in zip(ntok.convert_ids_to_tokens(enc['input_ids'][0]), pred): print(f'{t:12s} {nm.config.id2label[p]}')"""),
 md("## 5. Sentence embeddings: semantic search vs BM25 (same protocol as the RAG chapter: 8 newsgroup topics, 300 short queries, precision@10 of same-topic retrieval)\nReference numbers from the RAG chapter: BM25 0.432, TF-IDF 0.400, LSA 0.383, hybrid 0.425."),
 code("""from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
st = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=dev)
cats = ['sci.space','rec.autos','talk.politics.misc','comp.graphics','rec.sport.hockey','sci.med','misc.forsale','soc.religion.christian']
tr = fetch_20newsgroups(subset='train', categories=cats, remove=('headers','footers','quotes')); te = fetch_20newsgroups(subset='test', categories=cats, remove=('headers','footers','quotes'))
qidx = np.random.default_rng(1).choice([i for i, d in enumerate(te.data) if len(d.split()) > 30], 300, replace=False)
queries = [' '.join(te.data[i].split()[:12]) for i in qidx]; yq = te.target[qidx]
D = st.encode([' '.join(d.split()[:200]) for d in tr.data], batch_size=64, normalize_embeddings=True, show_progress_bar=False); Q = st.encode(queries, normalize_embeddings=True)
S = Q @ D.T; print('MiniLM P@10 =', round(np.mean([(tr.target[np.argsort(-S[i])[:10]] == yq[i]).mean() for i in range(len(queries))]), 3))"""),
 md("## 6. Sentiment: TF-IDF vs frozen sentence embeddings vs fine-tuned DistilBERT (SST-2)\nSST-2 (GLUE) validation set: 872 sentences. Report accuracy for each model and compare."),
 code("""from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
ds = load_dataset('glue', 'sst2'); trn = ds['train'].shuffle(seed=0).select(range(20000)); val = ds['validation']
tf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2); Xtr = tf.fit_transform(trn['sentence'])
print('TF-IDF(1,2) + LR  val acc:', round(LogisticRegression(max_iter=3000).fit(Xtr, trn['label']).score(tf.transform(val['sentence']), val['label']), 4))
Etr = st.encode(trn['sentence'], batch_size=128, normalize_embeddings=True, show_progress_bar=False); Eva = st.encode(val['sentence'], normalize_embeddings=True)
print('MiniLM frozen + LR val acc:', round(LogisticRegression(max_iter=3000).fit(Etr, trn['label']).score(Eva, val['label']), 4))"""),
 code("""from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
name = 'distilbert-base-uncased'; dt = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(dev)
def batches(d, bs, shuffle):
    idx = np.random.permutation(len(d)) if shuffle else np.arange(len(d))
    for s in range(0, len(d), bs):
        b = d.select(idx[s:s + bs].tolist()); enc = dt(b['sentence'], padding=True, truncation=True, max_length=64, return_tensors='pt')
        yield {k: v.to(dev) for k, v in enc.items()}, torch.tensor(b['label']).to(dev)
epochs, bs = 2, 32; opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
steps = epochs * (len(trn) // bs + 1); sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
def evaluate():
    model.eval(); c = 0
    with torch.no_grad():
        for x, y in batches(val, 128, False): c += (model(**x).logits.argmax(-1) == y).sum().item()
    model.train(); return c / len(val)
model.train()
for ep in range(epochs):
    for i, (x, y) in enumerate(batches(trn, bs, True)):
        loss = model(**x, labels=y).loss; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad()
        if i % 200 == 0: print(f'epoch {ep} step {i} loss {loss.item():.3f}')
    print(f'epoch {ep} val acc: {evaluate():.4f}')"""),
 md("## 7. Machine translation (T5-small) and metrics: BLEU, chrF, ROUGE\nWe also reproduce the hand-worked BLEU example from the chapter."),
 code("""from transformers import AutoModelForSeq2SeqLM
import sacrebleu
mt = AutoTokenizer.from_pretrained('t5-small'); mm = AutoModelForSeq2SeqLM.from_pretrained('t5-small').eval().to(dev)
src = ['The weather is nice today.', 'I would like a cup of coffee, please.', 'The meeting has been postponed until next week.', 'Where is the nearest train station?', 'She reads a book every evening.', 'Machine learning models need good data.']
ref = ["Il fait beau aujourd'hui.", "Je voudrais une tasse de café, s'il vous plaît.", 'La réunion a été reportée à la semaine prochaine.', 'Où est la gare la plus proche ?', 'Elle lit un livre tous les soirs.', "Les modèles d'apprentissage automatique ont besoin de bonnes données."]
enc = mt(['translate English to French: ' + x for x in src], return_tensors='pt', padding=True).to(dev)
with torch.no_grad(): out = mm.generate(**enc, max_new_tokens=64, num_beams=4)
hyp = mt.batch_decode(out, skip_special_tokens=True)
for s, h in zip(src, hyp): print(s, '->', h)
print('BLEU', round(sacrebleu.corpus_bleu(hyp, [ref]).score, 1), '| chrF', round(sacrebleu.corpus_chrf(hyp, [ref]).score, 1))
# hand example: hyp 'the cat is on the mat' vs ref 'there is a cat on the mat'
b = sacrebleu.sentence_bleu('the cat is on the mat', ['there is a cat on the mat'], smooth_method='none'); print('sentence BLEU (no smoothing, BLEU-4):', b.score, '(0 because there are no matching 4-grams)')
from rouge_score import rouge_scorer
print(rouge_scorer.RougeScorer(['rouge1', 'rougeL']).score('there is a cat on the mat', 'the cat is on the mat'))"""),
])
