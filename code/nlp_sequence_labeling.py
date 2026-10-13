"""POS tagging (baseline / HMM-Viterbi / CRF) and NER (CRF) -- executed on NLTK treebank and CoNLL-2002 Spanish."""
import collections, numpy as np, math, warnings; warnings.filterwarnings("ignore")
from nltk.corpus import treebank, conll2002
import sklearn_crfsuite
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

sents = [[(w, t) for w, t in s] for s in treebank.tagged_sents(tagset="universal")]
cut = int(len(sents) * .8); tr, te = sents[:cut], sents[cut:]; print("POS: train sents", len(tr), "test sents", len(te), "tokens", sum(len(s) for s in te))
tags = sorted({t for s in tr for _, t in s}); T = len(tags); ti = {t: i for i, t in enumerate(tags)}
wt = collections.defaultdict(collections.Counter)
for s in tr:
    for w, t in s: wt[w.lower()][t] += 1
common = collections.Counter(t for s in tr for _, t in s).most_common(1)[0][0]
def acc(pred): 
    c = n = 0
    for s, p in zip(te, pred): c += sum(a == b for (_, a), b in zip(s, p)); n += len(s)
    return c / n
known = lambda w: w.lower() in wt
base = [[wt[w.lower()].most_common(1)[0][0] if known(w) else common for w, _ in s] for s in te]
print(f"most-frequent-tag baseline:                   {acc(base):.4f}")
oov = sum(not known(w) for s in te for w, _ in s) / sum(len(s) for s in te); print(f"OOV rate in test: {oov:.3f}")
# HMM with add-k smoothing and a crude suffix/shape-based unknown model
trans = np.ones((T + 1, T)) * .01; emit = collections.defaultdict(lambda: np.zeros(T)); start = T
for s in tr:
    prev = start
    for w, t in s: trans[prev, ti[t]] += 1; emit[w.lower()][ti[t]] += 1; prev = ti[t]
trans /= trans.sum(1, keepdims=True); tagc = np.array([sum(emit[w][i] for w in emit) for i in range(T)])
def suffix_model():
    d = collections.defaultdict(lambda: np.zeros(T))
    for s in tr:
        for w, t in s:
            if wt[w.lower()].total() <= 1: d[w[-3:].lower()][ti[t]] += 1
    return d
suf = suffix_model()
def e_prob(w):
    if known(w): v = emit[w.lower()] + 1e-3; return v / tagc  # P(w|t) ~ count(w,t)/count(t)
    v = suf.get(w[-3:].lower(), np.ones(T)) + 1; v = v / v.sum()
    if w[0].isdigit(): v = np.ones(T) * 1e-3; v[ti["NUM"]] = 1
    elif w[0].isupper(): v = v * np.where(np.isin(tags, ["NOUN"]), 3, 1)
    return v / tagc * 1e-3
def viterbi(words):
    n = len(words); dp = np.full((n, T), -1e18); bp = np.zeros((n, T), int)
    dp[0] = np.log(trans[start]) + np.log(e_prob(words[0]))
    for i in range(1, n):
        sc = dp[i - 1][:, None] + np.log(trans[:T]); bp[i] = sc.argmax(0); dp[i] = sc.max(0) + np.log(e_prob(words[i]))
    path = [int(dp[-1].argmax())]
    for i in range(n - 1, 0, -1): path.append(int(bp[i][path[-1]]))
    return [tags[i] for i in reversed(path)]
hmm = [viterbi([w for w, _ in s]) for s in te]; print(f"HMM (bigram tags) + Viterbi:                  {acc(hmm):.4f}")
def feats(s, i):
    w = s[i][0]; f = {"bias": 1, "w.lower": w.lower(), "suf3": w[-3:], "suf2": w[-2:], "pre2": w[:2], "isupper": w.isupper(), "istitle": w.istitle(), "isdigit": w.isdigit(), "hyphen": "-" in w}
    f.update({"-1:w": s[i - 1][0].lower(), "-1:title": s[i - 1][0].istitle()} if i else {"BOS": True}); f.update({"+1:w": s[i + 1][0].lower()} if i < len(s) - 1 else {"EOS": True}); return f
crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.05, c2=0.05, max_iterations=100, all_possible_transitions=True)
crf.fit([[feats(s, i) for i in range(len(s))] for s in tr], [[t for _, t in s] for s in tr]); cp = crf.predict([[feats(s, i) for i in range(len(s))] for s in te])
print(f"CRF with word-shape/context features:         {acc(cp):.4f}")
unk = [(i, j) for i, s in enumerate(te) for j, (w, _) in enumerate(s) if not known(w)]
for name, P in [("baseline", base), ("HMM", hmm), ("CRF", cp)]: print(f"  accuracy on OOV tokens ({len(unk)}) -- {name}: {np.mean([P[i][j] == te[i][j][1] for i, j in unk]):.3f}")

print("\n== NER: CoNLL-2002 Spanish, CRF, entity-level F1 ==")
trn = conll2002.iob_sents("esp.train"); tst = conll2002.iob_sents("esp.testb"); print("train sents", len(trn), "test sents", len(tst))
def nf(s, i):
    w, p, _ = s[i]; f = {"bias": 1, "w.lower": w.lower(), "w[-3:]": w[-3:], "w[-2:]": w[-2:], "isupper": w.isupper(), "istitle": w.istitle(), "isdigit": w.isdigit(), "postag": p, "postag2": p[:2]}
    if i: w1, p1, _ = s[i - 1]; f.update({"-1:w.lower": w1.lower(), "-1:istitle": w1.istitle(), "-1:postag": p1})
    else: f["BOS"] = True
    if i < len(s) - 1: w1, p1, _ = s[i + 1]; f.update({"+1:w.lower": w1.lower(), "+1:istitle": w1.istitle(), "+1:postag": p1})
    else: f["EOS"] = True
    return f
X = [[nf(s, i) for i in range(len(s))] for s in trn]; Y = [[l for _, _, l in s] for s in trn]
c2 = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True).fit(X, Y)
Xt = [[nf(s, i) for i in range(len(s))] for s in tst]; Yt = [[l for _, _, l in s] for s in tst]; Pt = c2.predict(Xt)
print("token accuracy (inflated by 'O'):", round(np.mean([a == b for y, p in zip(Yt, Pt) for a, b in zip(y, p)]), 4))
print(f"entity-level micro  P {precision_score(Yt, Pt):.3f}  R {recall_score(Yt, Pt):.3f}  F1 {f1_score(Yt, Pt):.3f}")
print(classification_report(Yt, Pt, digits=3))
print("majority-'O' token accuracy:", round(np.mean([l == "O" for y in Yt for l in y]), 4), "-> entity F1 0.0")
