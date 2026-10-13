"""Deep-learning building blocks in pure NumPy (verified): MLP+backprop with gradient check, conv, LSTM cell, attention,
transformer block shapes, autoencoder vs PCA, VAE KL, InfoNCE, GCN layer."""
import numpy as np, warnings; warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

# ---------- MLP with backprop ----------
class MLP:
    def __init__(self, sizes, seed=0, act="relu"):
        r = np.random.default_rng(seed); self.act = act
        self.W = [r.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]; self.b = [np.zeros(b) for b in sizes[1:]]
    def _f(self, z): return np.maximum(z, 0) if self.act == "relu" else np.tanh(z)
    def _df(self, z): return (z > 0).astype(float) if self.act == "relu" else 1 - np.tanh(z) ** 2
    def forward(self, X):
        self.zs, self.as_ = [], [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = self.as_[-1] @ W + b; self.zs.append(z)
            self.as_.append(z if i == len(self.W) - 1 else self._f(z))
        z = self.zs[-1] - self.zs[-1].max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
    def loss(self, X, Y): P = self.forward(X); return -np.log(P[np.arange(len(X)), Y] + 1e-12).mean()
    def grads(self, X, Y):
        P = self.forward(X); n = len(X); d = P.copy(); d[np.arange(n), Y] -= 1; d /= n; gW, gb = [], []
        for i in reversed(range(len(self.W))):
            gW.insert(0, self.as_[i].T @ d); gb.insert(0, d.sum(0))
            if i > 0: d = (d @ self.W[i].T) * self._df(self.zs[i - 1])
        return gW, gb
    def fit(self, X, Y, epochs=200, lr=0.1, bs=64, seed=0):
        r = np.random.default_rng(seed)
        for e in range(epochs):
            idx = r.permutation(len(X))
            for s in range(0, len(X), bs):
                j = idx[s:s + bs]; gW, gb = self.grads(X[j], Y[j])
                for k in range(len(self.W)): self.W[k] -= lr * gW[k]; self.b[k] -= lr * gb[k]
        return self
    def predict(self, X): return self.forward(X).argmax(1)

print("== gradient check (numeric vs backprop) ==")
Xg = rng.normal(size=(5, 3)); Yg = np.array([0, 1, 2, 1, 0]); m = MLP([3, 4, 3], act="tanh")
gW, _ = m.grads(Xg, Yg); eps = 1e-6; num = np.zeros_like(m.W[0])
for i in range(num.shape[0]):
    for j in range(num.shape[1]):
        m.W[0][i, j] += eps; lp = m.loss(Xg, Yg); m.W[0][i, j] -= 2 * eps; lm = m.loss(Xg, Yg); m.W[0][i, j] += eps; num[i, j] = (lp - lm) / (2 * eps)
print("max |analytic - numeric| =", float(np.abs(num - gW[0]).max()))

print("\n== XOR ==")
Xx = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float); Yx = np.array([0, 1, 1, 0])
for hidden in [0, 4]:
    sizes = [2, 2] if hidden == 0 else [2, hidden, 2]
    mm = MLP(sizes, act="tanh", seed=1).fit(np.tile(Xx, (50, 1)), np.tile(Yx, 50), epochs=300, lr=0.2, bs=16)
    print("hidden" if hidden else "no hidden", "-> acc", (mm.predict(Xx) == Yx).mean())

print("\n== digits: scratch MLP vs sklearn ==")
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
X, y = load_digits(return_X_y=True); X = X / 16.0; Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=0, stratify=y)
sc = MLP([64, 64, 10], seed=0).fit(Xtr, ytr, epochs=60, lr=0.1)
print("scratch MLP test acc", round((sc.predict(Xte) == yte).mean(), 3), "| param count", sum(w.size for w in sc.W) + sum(b.size for b in sc.b))
print("sklearn MLP", round(MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, random_state=0).fit(Xtr, ytr).score(Xte, yte), 3), "| logistic", round(LogisticRegression(max_iter=3000).fit(Xtr, ytr).score(Xte, yte), 3))

print("\n== vanishing gradient arithmetic: sigmoid' <= 0.25 ==")
print("0.25**10 =", 0.25 ** 10, "| 0.25**30 =", 0.25 ** 30)

print("\n== conv worked example ==")
img = np.array([[1, 2, 0, 1], [3, 1, 2, 0], [0, 1, 3, 2], [1, 0, 2, 1]], float); k = np.array([[1, 0], [0, -1]], float)
out = np.array([[(img[i:i + 2, j:j + 2] * k).sum() for j in range(3)] for i in range(3)]); print(out)
print("params: conv 3x3, 64->64 ch:", 3 * 3 * 64 * 64 + 64, "| dense on 32x32x64 -> same out:", (32 * 32 * 64) ** 2)

print("\n== LSTM cell one step ==")
sig = lambda z: 1 / (1 + np.exp(-z)); x, hprev, cprev = 0.5, 0.1, 0.2
wf, wi, wo, wc = 1.0, 1.0, 1.0, 1.0; uf = ui = uo = uc = 1.0
f = sig(wf * x + uf * hprev); i = sig(wi * x + ui * hprev); o = sig(wo * x + uo * hprev); g = np.tanh(wc * x + uc * hprev)
c = f * cprev + i * g; h = o * np.tanh(c); print({k: round(float(v), 4) for k, v in dict(f=f, i=i, o=o, g=g, c=c, h=h).items()})

print("\n== scaled dot-product attention ==")
Q = np.array([[1, 0], [0, 1.]]); K = np.array([[1, 0], [0, 1.], [1, 1.]]); V = np.array([[1, 0], [0, 2.], [3, 3.]])
S = Q @ K.T / np.sqrt(2); A = np.exp(S - S.max(1, keepdims=True)); A /= A.sum(1, keepdims=True); print("weights\n", A.round(3), "\noutput\n", (A @ V).round(3))
def mha(X, h, rng):
    n, d = X.shape; dk = d // h; Wq, Wk, Wv, Wo = [rng.normal(0, d ** -.5, (d, d)) for _ in range(4)]
    q, k, v = [(X @ W).reshape(n, h, dk).transpose(1, 0, 2) for W in (Wq, Wk, Wv)]
    s = q @ k.transpose(0, 2, 1) / np.sqrt(dk); s += np.triu(np.full((n, n), -1e9), 1)    # causal mask
    a = np.exp(s - s.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True); return (a @ v).transpose(1, 0, 2).reshape(n, d) @ Wo, a
o, a = mha(rng.normal(size=(5, 16)), 4, rng); print("MHA output", o.shape, "attn", a.shape, "row0 attends only to itself:", a[0, 0].round(2).tolist(), "| upper triangle max:", float(np.triu(a[0], 1).max()))
n, d = 4096, 64; print("attention matrix entries at n=4096 (per head):", n * n, "-> float16 MB:", n * n * 2 / 1e6)

print("\n== autoencoder vs PCA (digits, 8-d code) ==")
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
p = PCA(8).fit(Xtr); rec = p.inverse_transform(p.transform(Xte)); print("PCA(8) MSE", round(float(np.mean((Xte - rec) ** 2)), 4))
ae = MLPRegressor(hidden_layer_sizes=(32, 8, 32), activation="tanh", max_iter=800, random_state=0).fit(Xtr, Xtr); print("MLP autoencoder (32-8-32) MSE", round(float(np.mean((Xte - ae.predict(Xte)) ** 2)), 4))

print("\n== VAE KL term, reparameterization ==")
mu, logvar = np.array([1.0, 0.0]), np.array([0.0, np.log(4)]); kl = 0.5 * (np.exp(logvar) + mu ** 2 - 1 - logvar); print("KL per dim", kl.round(4), "total", round(kl.sum(), 4))

print("\n== InfoNCE (contrastive) ==")
z = np.array([[1, 0], [0.9, 0.1], [0, 1], [-1, 0.]]); z = z / np.linalg.norm(z, axis=1, keepdims=True); tau = 0.5
sim = z @ z.T / tau; anchor, pos = 0, 1; neg = [2, 3]; num = np.exp(sim[anchor, pos]); den = num + sum(np.exp(sim[anchor, j]) for j in neg); print("loss", round(-np.log(num / den), 4))

print("\n== GCN layer: H' = ReLU(D^-1/2 (A+I) D^-1/2 H W) ==")
A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0.]]); Ah = A + np.eye(3); Dm = np.diag(Ah.sum(1) ** -.5); An = Dm @ Ah @ Dm
H = np.array([[1, 0], [0, 1], [1, 1.]]); W = np.eye(2); print(An.round(3)); print(np.maximum(An @ H @ W, 0).round(3))
