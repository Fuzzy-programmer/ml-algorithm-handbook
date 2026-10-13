"""Recommenders on synthetic implicit-explicit data: popularity, user-kNN, item-kNN, matrix factorization (ALS/SGD), evaluated on held-out interactions."""
import numpy as np
rng = np.random.default_rng(0)
nu, ni, k = 400, 300, 5
U = rng.normal(size=(nu, k)); V = rng.normal(size=(ni, k)); pop = rng.normal(0, 1, ni)
S = U @ V.T / np.sqrt(k) + 1.0 * pop                       # affinity = personal taste + global popularity
P = 1 / (1 + np.exp(-(S - 2.2))); Y = (rng.random((nu, ni)) < P).astype(float)
# hold out 1 positive per user for test
test = {}; train = Y.copy()
for u in range(nu):
    pos = np.where(Y[u] > 0)[0]
    if len(pos) >= 5: i = rng.choice(pos); test[u] = i; train[u, i] = 0
print("users with test item:", len(test), "| density", round(Y.mean(), 3))
def eval_scores(score, K=10):
    hit = ndcg = 0
    for u, i in test.items():
        s = score[u].copy(); s[train[u] > 0] = -1e9
        rank = (s > s[i]).sum()
        if rank < K: hit += 1; ndcg += 1 / np.log2(rank + 2)
    return hit / len(test), ndcg / len(test)
res = {}
res["random"] = eval_scores(rng.random((nu, ni)))
res["popularity"] = eval_scores(np.tile(train.sum(0), (nu, 1)))
def cos(A): n = np.linalg.norm(A, axis=1, keepdims=True) + 1e-9; return (A / n) @ (A / n).T
res["user-kNN (k=30)"] = None
Su = cos(train); np.fill_diagonal(Su, 0); idx = np.argsort(-Su, 1)[:, 30:]; Sk = Su.copy(); np.put_along_axis(Sk, idx, 0, 1); res["user-kNN (k=30)"] = eval_scores(Sk @ train)
Si = cos(train.T); np.fill_diagonal(Si, 0); res["item-kNN"] = eval_scores(train @ Si)
# Implicit-feedback ALS (Hu, Koren, Volinsky 2008), confidence c = 1 + alpha*r
def als(R, f=10, lam=0.1, alpha=20, iters=15, seed=0):
    r = np.random.default_rng(seed); X = r.normal(0, .1, (R.shape[0], f)); Yf = r.normal(0, .1, (R.shape[1], f)); C = 1 + alpha * R
    for _ in range(iters):
        for A, B, Rm, Cm in [(X, Yf, R, C), (Yf, X, R.T, C.T)]:
            for u in range(A.shape[0]):
                Cu = Cm[u]; A[u] = np.linalg.solve((B.T * Cu) @ B + lam * np.eye(f), (B.T * Cu) @ Rm[u])
    return X, Yf
for f in [2, 5, 10]:
    X, Yf = als(train, f=f); res[f"ALS implicit MF (f={f})"] = eval_scores(X @ Yf.T)
print(f"{'method':24s} HitRate@10  NDCG@10")
for n, (h, g) in res.items(): print(f"{n:24s} {h:9.3f} {g:8.3f}")
# SGD explicit MF worked check
print("\nexplicit MF tiny example: r=4, p=[1,0.5], q=[2,1] -> pred", 1*2+.5*1, "err", 4-2.5)
