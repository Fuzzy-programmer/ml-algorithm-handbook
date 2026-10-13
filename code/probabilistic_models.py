"""GMM (EM from scratch + sklearn BIC), HMM (forward/Viterbi from scratch), tiny Bayesian network by enumeration."""
import numpy as np, itertools, warnings; warnings.filterwarnings("ignore")
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

rng = np.random.default_rng(0)
def em_gmm(X, K, iters=100, seed=0):
    n, d = X.shape; r = np.random.default_rng(seed)
    mu = X[r.choice(n, K, replace=False)]; cov = np.array([np.cov(X.T) for _ in range(K)]); pi = np.ones(K) / K
    for it in range(iters):
        ll = np.zeros((n, K))
        for k in range(K):
            diff = X - mu[k]; inv = np.linalg.inv(cov[k])
            ll[:, k] = np.log(pi[k]) - .5 * (np.einsum("ij,jk,ik->i", diff, inv, diff) + np.linalg.slogdet(cov[k])[1] + d * np.log(2 * np.pi))
        m = ll.max(1, keepdims=True); lse = m[:, 0] + np.log(np.exp(ll - m).sum(1)); R = np.exp(ll - lse[:, None])     # E-step
        Nk = R.sum(0); pi = Nk / n; mu = (R.T @ X) / Nk[:, None]                                                      # M-step
        for k in range(K):
            diff = X - mu[k]; cov[k] = (R[:, k, None] * diff).T @ diff / Nk[k] + 1e-6 * np.eye(d)
    return pi, mu, cov, R, lse.sum()

print("== GMM vs KMeans on elongated overlapping clusters ==")
A = rng.multivariate_normal([0, 0], [[4, 3.4], [3.4, 4]], 300); B = rng.multivariate_normal([4, -2], [[4, 3.4], [3.4, 4]], 300)
X = np.r_[A, B]; y = np.r_[np.zeros(300), np.ones(300)]
print("KMeans ARI", round(adjusted_rand_score(y, KMeans(2, n_init=10, random_state=0).fit_predict(X)), 3))
pi, mu, cov, R, LL = em_gmm(X, 2)
print("scratch EM ARI", round(adjusted_rand_score(y, R.argmax(1)), 3), "weights", pi.round(2), "means", mu.round(1).tolist())
g = GaussianMixture(2, random_state=0).fit(X); print("sklearn GMM ARI", round(adjusted_rand_score(y, g.predict(X)), 3), "loglik/n scratch vs sk:", round(LL / len(X), 3), round(g.score(X), 3))

print("\n== Choosing K by BIC (3 true blobs) ==")
Xb, _ = make_blobs(600, centers=3, cluster_std=1.0, random_state=1)
for k in range(1, 7): print(k, round(GaussianMixture(k, n_init=3, random_state=0).fit(Xb).bic(Xb), 1))

print("\n== HMM: weather (hidden) -> activity (observed) ==")
S = ["Rainy", "Sunny"]; O = ["walk", "shop", "clean"]
pi0 = np.array([.6, .4]); T = np.array([[.7, .3], [.4, .6]]); E = np.array([[.1, .4, .5], [.6, .3, .1]])
obs = [0, 1, 2]  # walk, shop, clean
alpha = pi0 * E[:, obs[0]]; print("alpha1", alpha.round(4))
for o in obs[1:]: alpha = (alpha @ T) * E[:, o]; print("alpha", alpha.round(5))
print("P(obs) =", round(alpha.sum(), 6))
d = pi0 * E[:, obs[0]]; bp = []
for o in obs[1:]:
    sc = d[:, None] * T; bp.append(sc.argmax(0)); d = sc.max(0) * E[:, o]
path = [int(d.argmax())]
for b in reversed(bp): path.append(int(b[path[-1]]))
print("Viterbi path:", [S[i] for i in reversed(path)], "prob", round(d.max(), 6))
# brute-force check
best = max(itertools.product(range(2), repeat=3), key=lambda s: pi0[s[0]] * E[s[0], 0] * T[s[0], s[1]] * E[s[1], 1] * T[s[1], s[2]] * E[s[2], 2])
print("brute-force best:", [S[i] for i in best])

print("\n== Bayesian network: Rain->Sprinkler, Rain->Wet, Sprinkler->Wet ==")
pR = .2; pS = {1: .01, 0: .4}; pW = {(1, 1): .99, (1, 0): .8, (0, 1): .9, (0, 0): 0.0}   # (R,S)->P(W=1)
def joint(r, s, w):
    p = (pR if r else 1 - pR) * (pS[r] if s else 1 - pS[r]); pw = pW[(r, s)]; return p * (pw if w else 1 - pw)
num = sum(joint(1, s, 1) for s in (0, 1)); den = sum(joint(r, s, 1) for r in (0, 1) for s in (0, 1))
print("P(W=1) =", round(den, 4), " P(R=1|W=1) =", round(num / den, 4))
num2 = sum(joint(1, 1, 1) for _ in [0]); den2 = sum(joint(r, 1, 1) for r in (0, 1))
print("P(R=1|W=1,S=1) =", round(num2 / den2, 4), "(explaining away: sprinkler explains the wet grass)")
