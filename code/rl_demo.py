"""RL from scratch: bandits (eps-greedy, UCB, Thompson), value iteration, Q-learning vs SARSA on Cliff Walking, REINFORCE on a bandit-ish task."""
import numpy as np
rng = np.random.default_rng(0)

print("== 10-armed bandit: avg reward & % optimal over 1000 steps, 300 runs ==")
def bandit_run(strategy, steps=1000, runs=300, K=10):
    tot = np.zeros(steps); opt = np.zeros(steps)
    for r in range(runs):
        g = np.random.default_rng(r); mu = g.normal(0, 1, K); best = mu.argmax(); Q = np.zeros(K); N = np.zeros(K); S_ = np.zeros(K)
        for t in range(steps):
            if strategy == "greedy": a = Q.argmax() if t >= K else t
            elif strategy == "eps=0.1": a = g.integers(K) if g.random() < .1 else Q.argmax()
            elif strategy == "UCB(c=2)": a = t if t < K else int(np.argmax(Q + 2 * np.sqrt(np.log(t + 1) / N)))
            elif strategy == "Thompson": a = int(np.argmax(g.normal(S_ / (N + 1), 1 / np.sqrt(N + 1))))   # posterior N(sum/(n+1), 1/(n+1)) for prior N(0,1), noise var 1
            rew = g.normal(mu[a], 1); N[a] += 1; S_[a] += rew; Q[a] += (rew - Q[a]) / N[a]
            tot[t] += mu[a]; opt[t] += (a == best)
    return tot / runs, opt / runs
for s in ["greedy", "eps=0.1", "UCB(c=2)", "Thompson"]:
    r, o = bandit_run(s); print(f"{s:10s} mean true-value of chosen arm (last 500): {r[500:].mean():.3f}   % optimal (last 500): {100*o[500:].mean():.1f}")

print("\n== Value iteration on a 1-D corridor (5 states, goal at right, gamma=0.9) ==")
V = np.zeros(5); gamma = .9
for it in range(50):
    new = V.copy()
    for s in range(4):
        new[s] = max(((1 if s + 1 == 4 else 0) + gamma * V[min(s + 1, 4)]), (0 + gamma * V[max(s - 1, 0)]))
    V = new
print("V*:", V.round(3), "(expected 0.9^k for k steps from goal: [0.729 0.81 0.9 1. 0.])")

print("\n== Cliff walking 4x12: Q-learning vs SARSA ==")
R_, C_ = 4, 12; start, goal = (3, 0), (3, 11)
def step(s, a):
    r, c = s; dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][a]; r = min(max(r + dr, 0), R_ - 1); c = min(max(c + dc, 0), C_ - 1)
    if r == 3 and 0 < c < 11: return start, -100, False
    return (r, c), -1, (r, c) == goal
def run(alg, episodes=500, eps=.1, alpha=.5, gamma=1.0, seed=0):
    g = np.random.default_rng(seed); Q = np.zeros((R_, C_, 4)); rets = []
    pol = lambda s: g.integers(4) if g.random() < eps else int(np.argmax(Q[s]))
    for ep in range(episodes):
        s = start; a = pol(s); G = 0; done = False
        while not done:
            s2, r, done = step(s, a); a2 = pol(s2)
            target = r + (0 if done else gamma * (Q[s2][a2] if alg == "sarsa" else Q[s2].max()))
            Q[s][a] += alpha * (target - Q[s][a]); s, a = s2, a2; G += r
        rets.append(G)
    return Q, np.mean(rets[-100:])
def greedy_path(Q):
    s = start; path = [s]
    for _ in range(40):
        s, _, d = step(s, int(np.argmax(Q[s]))); path.append(s)
        if d: break
    return path
def greedy_path(Q, cap=100):
    s = start; path = [s]
    for _ in range(cap):
        s, _, d = step(s, int(np.argmax(Q[s]))); path.append(s)
        if d: return path, True
    return path, False
for alg in ["q", "sarsa"]:
    perfs, lens, reached = [], [], 0
    for sd in range(30):
        Q, perf = run(alg, episodes=500, seed=sd); perfs.append(perf); p, ok = greedy_path(Q)
        if ok: reached += 1; lens.append(len(p) - 1)
    print(f"{alg:6s} online return (last 100 eps, eps=0.1), mean over 30 seeds: {np.mean(perfs):7.1f} | greedy policy reaches goal in {reached}/30 seeds, median path length {np.median(lens):.0f}")

print("\n== REINFORCE on 3-armed bandit (softmax policy) ==")
mu = np.array([0.2, 0.5, 1.0]); th = np.zeros(3); lr = .1
for t in range(3000):
    p = np.exp(th - th.max()); p /= p.sum(); a = rng.choice(3, p=p); r = rng.normal(mu[a], 1)
    grad = -p; grad[a] += 1; th += lr * r * grad       # baseline-free REINFORCE
p = np.exp(th - th.max()); p /= p.sum(); print("policy probs:", p.round(3), "(best arm = index 2)")
