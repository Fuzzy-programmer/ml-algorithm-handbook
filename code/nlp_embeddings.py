"""Word embeddings (executed): co-occurrence/PPMI+SVD, skip-gram negative sampling from scratch, neighbor checks, WordNet-proxy evaluation."""
import collections, numpy as np, time, warnings; warnings.filterwarnings("ignore")
from nltk.corpus import brown, wordnet as wn
from scipy.stats import spearmanr
from scipy.sparse import coo_matrix
from sklearn.utils.extmath import randomized_svd
rng = np.random.default_rng(0)
toks = [w.lower() for w in brown.words() if w.isalpha()]
cnt = collections.Counter(toks); vocab = [w for w, c in cnt.most_common() if c >= 15]; idx = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
ids = np.array([idx[w] for w in toks if w in idx]); print("tokens", len(toks), "kept", len(ids), "vocab", V)
def neighbors(E, w, k=6):
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9); s = En @ En[idx[w]]; return [vocab[i] for i in np.argsort(-s)[1:k + 1]]

# ---- PPMI + SVD (Levy & Goldberg 2014 view of SGNS) ----
W = 4; rows, cols, vals = [], [], []
for off in range(1, W + 1):
    a, b = ids[:-off], ids[off:]; w = 1.0 / off
    rows += [a, b]; cols += [b, a]; vals += [np.full(len(a), w)] * 2
M = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(V, V)).tocsr(); M.sum_duplicates()
tot = M.sum(); rs = np.asarray(M.sum(1)).ravel(); cs = np.asarray(M.sum(0)).ravel() ** 0.75; cs = cs / cs.sum() * tot   # context distribution smoothing alpha=0.75
Mc = M.tocoo(); pmi = np.log((Mc.data * tot) / (rs[Mc.row] * cs[Mc.col])); keep = pmi > 0
P = coo_matrix((pmi[keep], (Mc.row[keep], Mc.col[keep])), shape=(V, V)).tocsr()
U, S, Vt = randomized_svd(P, 100, random_state=0); E_ppmi = U * np.sqrt(S)
print("\nPPMI+SVD(100) neighbors:")
for w in ["president", "car", "church", "money", "war", "university"]: print(f"  {w:10s} -> {neighbors(E_ppmi, w)}")

# ---- skip-gram negative sampling from scratch ----
def sgns(dim=60, epochs=3, neg=5, lr0=0.05, win=4, batch=4096, seed=0):
    r = np.random.default_rng(seed); Win = (r.random((V, dim)) - .5) / dim; Wout = np.zeros((V, dim))
    f = np.array([cnt[w] for w in vocab], float); keep_p = np.minimum(1, np.sqrt(1e-4 / (f / f.sum())) + 1e-4 / (f / f.sum()))   # frequent-word subsampling
    noise = f ** .75; noise /= noise.sum()
    for ep in range(epochs):
        sub = ids[r.random(len(ids)) < keep_p[ids]]; pairs = []
        for off in range(1, win + 1): pairs.append(np.c_[sub[:-off], sub[off:]]); pairs.append(np.c_[sub[off:], sub[:-off]])
        pairs = np.concatenate(pairs); pairs = pairs[r.permutation(len(pairs))]; lr = lr0 * (1 - ep / epochs)
        for s in range(0, len(pairs), batch):
            c, o = pairs[s:s + batch, 0], pairs[s:s + batch, 1]; n = r.choice(V, size=(len(c), neg), p=noise)
            vc = Win[c]; uo = Wout[o]; un = Wout[n]
            gp = 1 / (1 + np.exp(-(vc * uo).sum(1))) - 1; gn = 1 / (1 + np.exp(-np.einsum("bd,bkd->bk", vc, un)))   # gradients of -log sigma(.)
            dvc = gp[:, None] * uo + np.einsum("bk,bkd->bd", gn, un)
            np.add.at(Win, c, -lr * dvc); np.add.at(Wout, o, -lr * gp[:, None] * vc); np.add.at(Wout, n, -lr * gn[:, :, None] * vc[:, None, :])
    return Win
t = time.time(); E_sg = sgns(); print(f"\nSGNS (dim 60, 3 epochs) trained in {time.time()-t:.0f}s; neighbors:")
for w in ["president", "car", "church", "money", "war", "university"]: print(f"  {w:10s} -> {neighbors(E_sg, w)}")

# ---- proxy evaluation: Spearman between cosine and WordNet path similarity over noun pairs ----
nouns = [w for w in vocab[:3000] if wn.synsets(w, "n")][:600]
pairs = [(nouns[i], nouns[j]) for i, j in rng.integers(0, len(nouns), (4000, 2)) if i != j]
def wn_sim(a, b): return max((x.path_similarity(y) or 0) for x in wn.synsets(a, "n")[:3] for y in wn.synsets(b, "n")[:3])
ref = np.array([wn_sim(a, b) for a, b in pairs])
def cos(E, a, b): ea, eb = E[idx[a]], E[idx[b]]; return ea @ eb / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-9)
for name, E in [("random vectors", rng.normal(size=(V, 60))), ("PPMI+SVD(100)", E_ppmi), ("SGNS (scratch)", E_sg)]:
    print(f"{name:16s} Spearman vs WordNet path-sim over {len(pairs)} noun pairs: {spearmanr(ref, [cos(E, a, b) for a, b in pairs])[0]:.3f}")
