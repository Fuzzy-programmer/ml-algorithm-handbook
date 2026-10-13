"""GenAI building blocks, executable without an LLM: BPE, sampling, KV-cache math, BM25/TF-IDF/LSA retrieval + RRF, LoRA, quantization, DPO/RLHF math, diffusion toy, ViT patches, agent loop with a scripted model."""
import re, json, math, collections, numpy as np, warnings; warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

print("== BPE (Sennrich et al. 2016) on a tiny corpus ==")
corpus = "low low low low low lower lower newest newest newest newest newest newest widest widest widest".split()
vocab = collections.Counter(" ".join(w) + " </w>" for w in corpus)
def pairs(v):
    p = collections.Counter()
    for w, f in v.items():
        s = w.split(); [p.update({(s[i], s[i + 1]): f}) for i in range(len(s) - 1)]
    return p
merges = []
for _ in range(8):
    p = pairs(vocab); best = max(p, key=p.get); merges.append((best, p[best]))
    pat = re.compile(r"(?<!\S)" + re.escape(" ".join(best)) + r"(?!\S)"); vocab = {pat.sub("".join(best), w): f for w, f in vocab.items()}
print("merges:", [(a + b, c) for (a, b), c in merges]); print("segmented:", list(vocab))

print("\n== sampling: temperature / top-k / top-p on logits [4,3,2,1,0] ==")
def probs(l, T=1.0): z = np.array(l) / T; e = np.exp(z - z.max()); return e / e.sum()
l = [4, 3, 2, 1, 0]
for T in [0.5, 1.0, 2.0]: print(f"T={T}: p={probs(l, T).round(3)}  entropy={-(probs(l,T)*np.log2(probs(l,T))).sum():.2f} bits")
p = probs(l); srt = np.argsort(-p); cum = np.cumsum(p[srt]); keep = srt[:np.searchsorted(cum, 0.9) + 1]; print("top-p=0.9 keeps tokens", keep.tolist(), "mass", round(p[keep].sum(), 3))

print("\n== KV cache memory (fp16) ==")
def kv(layers, kv_heads, hd, seq, batch=1, bytes_=2): return 2 * layers * kv_heads * hd * seq * batch * bytes_
print("32 layers, 32 KV heads, head_dim 128, 4096 ctx, batch 1: %.2f GB" % (kv(32, 32, 128, 4096) / 1e9))
print("same with GQA (8 KV heads): %.2f GB" % (kv(32, 8, 128, 4096) / 1e9), "| batch 16:", "%.1f GB" % (kv(32, 8, 128, 4096, 16) / 1e9))

print("\n== retrieval: BM25 vs TF-IDF vs LSA vs hybrid (RRF) on 20 newsgroups, P@10 / Recall of same-topic ==")
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
cats = ["sci.space", "rec.autos", "talk.politics.misc", "comp.graphics", "rec.sport.hockey", "sci.med", "misc.forsale", "soc.religion.christian"]
tr = fetch_20newsgroups(subset="train", categories=cats, remove=("headers", "footers", "quotes")); te = fetch_20newsgroups(subset="test", categories=cats, remove=("headers", "footers", "quotes"))
docs = [d for d in tr.data]; ydocs = tr.target
qidx = np.random.default_rng(1).choice([i for i, d in enumerate(te.data) if len(d.split()) > 30], 300, replace=False)
queries = [" ".join(te.data[i].split()[:12]) for i in qidx]; yq = te.target[qidx]       # short query = first 12 words of a held-out post; relevant = same topic
cv = CountVectorizer(stop_words="english"); C = cv.fit_transform(docs).tocsc(); N = C.shape[0]; df_ = np.diff(C.indptr); idf = np.log((N - df_ + .5) / (df_ + .5) + 1)
dl = np.asarray(C.sum(1)).ravel(); avg = dl.mean(); k1, b = 1.5, .75
def bm25(q):
    ids = [i for i in cv.transform([q]).nonzero()[1]]; s = np.zeros(N)
    for t in ids:
        col = C[:, t].tocoo(); tf = col.data; s[col.row] += idf[t] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[col.row] / avg))
    return s
tf = TfidfVectorizer(stop_words="english", sublinear_tf=True); D = tf.fit_transform(docs)
svd = TruncatedSVD(150, random_state=0).fit(D); Dl = normalize(svd.transform(D))
def scores_all(q):
    s_b = bm25(q); s_t = (D @ tf.transform([q]).T).toarray().ravel(); s_l = Dl @ normalize(svd.transform(tf.transform([q])))[0]; return s_b, s_t, s_l
def rrf(*ss, k=60):
    out = np.zeros(N)
    for s in ss: r = np.empty(N); r[np.argsort(-s)] = np.arange(1, N + 1); out += 1 / (k + r)
    return out
res = collections.defaultdict(list)
for q, y in zip(queries, yq):
    s_b, s_t, s_l = scores_all(q)
    for name, s in [("BM25", s_b), ("TF-IDF cosine", s_t), ("LSA-150 (dense)", s_l), ("Hybrid RRF(BM25+LSA)", rrf(s_b, s_l)), ("random", rng.random(N))]:
        top = np.argsort(-s)[:10]; res[name].append((ydocs[top] == y).mean())
for n, v in res.items(): print(f"{n:22s} precision@10 = {np.mean(v):.3f}")

print("\n== LoRA ==")
d, r = 4096, 8; print(f"one {d}x{d} matrix: full {d*d/1e6:.1f}M params; LoRA r={r}: {2*d*r/1e3:.1f}K params ({100*2*d*r/(d*d):.2f}%)")
n, dd, rr = 200, 32, 4; W0 = rng.normal(size=(dd, dd)) / np.sqrt(dd); dW = rng.normal(size=(dd, rr)) @ rng.normal(size=(rr, dd)) / np.sqrt(dd * rr) * 1.5
X = rng.normal(size=(n, dd)); Y = X @ (W0 + dW).T
for rank in [1, 4, 8]:
    A = rng.normal(size=(rank, dd)) * .1; B = np.zeros((dd, rank)); lr = .02
    for it in range(3000):
        pred = X @ (W0 + B @ A).T; E = pred - Y; gW = E.T @ X / n; gB = gW @ A.T; gA = B.T @ gW; B -= lr * gB; A -= lr * gA
    print(f"rank {rank}: trainable {A.size+B.size:5d} (full {dd*dd}), final MSE {np.mean((X@(W0+B@A).T-Y)**2):.5f}   [true update has rank {rr}]")

print("\n== quantization error (Gaussian weights, 4096x256) ==")
W = rng.normal(0, 1, (4096, 256)) * rng.uniform(.2, 2, (4096, 1))        # rows with different scales
def quant(W, bits, axis):
    qmax = 2 ** (bits - 1) - 1; s = np.abs(W).max(axis=axis, keepdims=True) / qmax; return np.clip(np.round(W / s), -qmax - 1, qmax) * s
for bits in [8, 4]:
    pt = quant(W, bits, None); pc = quant(W, bits, 1)
    print(f"int{bits}: per-tensor RMSE {np.sqrt(np.mean((W-pt)**2)):.4f} | per-row RMSE {np.sqrt(np.mean((W-pc)**2)):.4f}  | memory vs fp16: {bits/16:.2f}x")

print("\n== preference optimization math ==")
sig = lambda z: 1 / (1 + np.exp(-z)); r_w, r_l = 1.2, 0.4
print("Bradley-Terry P(chosen > rejected) =", round(sig(r_w - r_l), 3), "| RM loss", round(-np.log(sig(r_w - r_l)), 3))
beta = .1; lp_w, lp_l, ref_w, ref_l = -10., -12., -11., -11.5
margin = beta * ((lp_w - ref_w) - (lp_l - ref_l)); print("DPO margin", round(margin, 3), "loss", round(-np.log(sig(margin)), 4), "(margin 0 -> loss 0.6931)")

print("\n== diffusion toy: exact-score DDPM sampler on a 1-D two-mode mixture ==")
T = 200; betas = np.linspace(1e-4, 0.05, T); alphas = 1 - betas; abar = np.cumprod(alphas); mus, sd, w = np.array([-2., 2.]), .3, np.array([.5, .5])
print("alpha_bar at t=1, 50, 100, 200:", abar[[0, 49, 99, 199]].round(4), "| SNR at T:", round(abar[-1] / (1 - abar[-1]), 5))
def eps_hat(x, t):
    m = np.sqrt(abar[t]) * mus; v = abar[t] * sd ** 2 + (1 - abar[t]); lik = w * np.exp(-(x[:, None] - m) ** 2 / (2 * v)) / np.sqrt(v); post = lik / lik.sum(1, keepdims=True)
    score = (post * (-(x[:, None] - m) / v)).sum(1); return -np.sqrt(1 - abar[t]) * score
x = rng.normal(size=20000)
for t in reversed(range(T)):
    z = rng.normal(size=x.shape) if t > 0 else 0
    x = (x - betas[t] / np.sqrt(1 - abar[t]) * eps_hat(x, t)) / np.sqrt(alphas[t]) + np.sqrt(betas[t]) * z
print(f"samples: frac near -2: {np.mean(np.abs(x+2)<1):.3f}, near +2: {np.mean(np.abs(x-2)<1):.3f}, mean of left mode {x[x<0].mean():.2f}, right {x[x>0].mean():.2f}, std right {x[x>0].std():.2f}  (target 0.5/0.5, -2/+2, 0.30)")

print("\n== ViT / multimodal arithmetic ==")
print("224x224 image, 16x16 patches ->", (224 // 16) ** 2, "tokens (+1 CLS); 336px/14 ->", (336 // 14) ** 2, "tokens")
