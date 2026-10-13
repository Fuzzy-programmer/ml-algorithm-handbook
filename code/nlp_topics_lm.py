"""Topic models (LSA/NMF/LDA) and n-gram language models with smoothing, evaluated by perplexity (executed)."""
import math, collections, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from nltk.corpus import brown

cats = ["sci.space", "rec.autos", "talk.politics.misc", "comp.graphics", "rec.sport.hockey", "sci.med"]
d = fetch_20newsgroups(subset="train", categories=cats, remove=("headers", "footers", "quotes")); texts = d.data
def top(components, names, n=8): return [" ".join(names[i] for i in c.argsort()[::-1][:n]) for c in components]
tf = TfidfVectorizer(stop_words="english", max_df=.5, min_df=5); X = tf.fit_transform(texts)
print("== NMF topics (TF-IDF) =="); [print(f" {i}: {t}") for i, t in enumerate(top(NMF(6, init="nndsvd", random_state=0, max_iter=400).fit(X).components_, tf.get_feature_names_out()))]
print("== LDA topics (counts) ==")
cv = CountVectorizer(stop_words="english", max_df=.5, min_df=5); C = cv.fit_transform(texts)
lda = LatentDirichletAllocation(6, learning_method="batch", max_iter=25, random_state=0).fit(C)
[print(f" {i}: {t}") for i, t in enumerate(top(lda.components_, cv.get_feature_names_out()))]
purity = []
th = lda.transform(C).argmax(1)
tab = np.zeros((6, 6), int)
for t, y in zip(th, d.target): tab[t, y] += 1
print("LDA topic->true category purity:", round(tab.max(1).sum() / tab.sum(), 3), "(random ~", round(1 / 6, 2), ")")
print("LDA perplexity (held-in) for K=2,6,12:", [round(LatentDirichletAllocation(k, max_iter=15, random_state=0).fit(C).perplexity(C), 0) for k in (2, 6, 12)], "(lower is better but held-in perplexity favors big K)")
svd = TruncatedSVD(6, random_state=0).fit(X); print("LSA explained variance ratio:", svd.explained_variance_ratio_.round(3))

print("\n== n-gram language models on Brown (news) ==")
sents = [[w.lower() for w in s] for s in brown.sents(categories="news")]; split = int(len(sents) * .9); tr, te = sents[:split], sents[split:]
cnt1 = collections.Counter(w for s in tr for w in s); thr = 2
vocab = {w for w, c in cnt1.items() if c >= thr} | {"<s>", "</s>", "<unk>"}
prep = lambda ss: [["<s>"] + [w if w in vocab else "<unk>" for w in s] + ["</s>"] for s in ss]
trp, tep = prep(tr), prep(te); V = len(vocab)
u = collections.Counter(w for s in trp for w in s); b = collections.Counter((s[i], s[i + 1]) for s in trp for i in range(len(s) - 1)); t3 = collections.Counter((s[i], s[i + 1], s[i + 2]) for s in trp for i in range(len(s) - 2))
bc = collections.Counter(s[i] for s in trp for i in range(len(s) - 1)); N = sum(u.values())
# KN-style continuation counts for interpolated absolute discounting (bigram)
cont = collections.Counter(w2 for (w1, w2) in b); n_types = len(b); fol = collections.Counter(w1 for (w1, w2) in b)
def p_uni(w): return (u[w] + 1) / (N + V)
def p_bi_add1(w1, w2): return (b[(w1, w2)] + 1) / (bc[w1] + V)
def p_bi_int(w1, w2, lam=0.7): return lam * (b[(w1, w2)] / bc[w1] if bc[w1] else 0) + (1 - lam) * p_uni(w2)
D = 0.75
def p_bi_kn(w1, w2):
    pc = cont[w2] / n_types
    if bc[w1] == 0: return pc
    return max(b[(w1, w2)] - D, 0) / bc[w1] + D * fol[w1] / bc[w1] * pc
bigram_ctx = collections.Counter((s[i], s[i + 1]) for s in trp for i in range(len(s) - 2))
def p_tri_int(w1, w2, w3): 
    c12 = b[(w1, w2)]; pt = t3[(w1, w2, w3)] / c12 if c12 else 0; return 0.5 * pt + 0.3 * (b[(w2, w3)] / bc[w2] if bc[w2] else 0) + 0.2 * p_uni(w3)
def ppl(fn, order):
    ll = n = 0
    for s in tep:
        for i in range(max(order - 1, 1), len(s)):
            p = fn(*s[i - order + 1:i + 1]); ll += math.log(p) if p > 0 else -1e9; n += 1
    return math.exp(-ll / n)
print("vocab", V, "test tokens", sum(len(s) - 1 for s in tep))
print("unigram add-1 perplexity:        %.0f" % ppl(p_uni, 1))
print("bigram MLE perplexity:            inf (unseen bigrams get probability 0)  -> zeros:", sum(1 for s in tep for i in range(len(s) - 1) if b[(s[i], s[i + 1])] == 0), "of", sum(len(s) - 1 for s in tep), "test bigrams unseen")
print("bigram add-1 perplexity:         %.0f" % ppl(p_bi_add1, 2))
print("bigram interpolated (0.7/0.3):   %.0f" % ppl(p_bi_int, 2))
print("bigram Kneser-Ney (D=0.75):      %.0f" % ppl(p_bi_kn, 2))
print("trigram interpolated:            %.0f" % ppl(p_tri_int, 3))
