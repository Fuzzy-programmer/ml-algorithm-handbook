# Chapter 6.3 — Reinforcement Learning: Bandits → DP → MC/TD → Q-Learning/SARSA → DQN → Policy Gradients → PPO/SAC

[← Roadmap](../README.md) · Prev [6.2](02-recommender-systems.md) · Next [6.4 Causal inference](04-causal-inference.md) · Priority 🟠 · Code: [`rl_demo.py`](../code/rl_demo.py)

## 1. Identity & problem
An agent interacts with an environment, takes actions, observes states and rewards, and learns a **policy** that maximizes cumulative (discounted) reward. Games, robotics, control, recommendation/ads (bandits), resource allocation, and alignment of LLMs (RLHF, Ch 6.6). No labeled "correct action"; credit assignment over time is the core difficulty.

## 2. History — a tree of ideas
| Branch | Milestones |
|---|---|
| Bandits | Thompson sampling (1933); Robbins (1952); UCB (Auer et al. 2002); contextual bandits (LinUCB, Li et al. 2010) |
| Dynamic programming | Bellman (1957) equations; policy/value iteration (Howard 1960) — need a known model |
| Monte Carlo | Learn from full episodes, no model |
| Temporal difference | TD(λ) (Sutton 1988), **Q-learning** (Watkins 1989), **SARSA** (Rummery & Niranjan 1994) |
| Function approximation | TD-Gammon (Tesauro 1992) → **DQN** (Mnih et al. 2013/2015: replay buffer + target network), Double DQN, Dueling, Rainbow |
| Policy gradient | REINFORCE (Williams 1992), actor–critic, A2C/A3C, **TRPO** (Schulman 2015), **PPO** (Schulman 2017), DDPG, TD3, **SAC** (Haarnoja 2018) |
| Model-based & search | Dyna, AlphaGo/AlphaZero (MCTS + nets), MuZero, Dreamer |
Value-based methods learn "how good is each action"; policy-based methods directly optimize the policy; actor–critic combines both. Not a strict succession: PPO and SAC coexist, chosen by action-space and sample-efficiency needs.

## 3. Intuition
Trial and error with memory: try things, notice which led to reward, shift probability toward them—while still exploring enough to discover better options.

## 4. Math
MDP `(S,A,P,R,γ)`; return `G_t=Σγ^k r_{t+k+1}`. Value `V^π(s)=E[G_t|s]`, `Q^π(s,a)`. **Bellman optimality:** `Q*(s,a) = E[r + γ max_{a'} Q*(s',a')]`.
- **Q-learning (off-policy):** `Q(s,a) ← Q(s,a) + α[r + γ max_{a'}Q(s',a') − Q(s,a)]`. **SARSA (on-policy):** uses the action actually taken, `Q(s',a')`.
- **Policy gradient:** `∇J(θ)=E[∇log π_θ(a|s)·(G_t − b(s))]`; advantage `A=Q−V` reduces variance. **PPO** maximizes the clipped objective `E[min(ρ_t A_t, clip(ρ_t,1−ε,1+ε)A_t)]`, `ρ_t=π_θ/π_old`, keeping updates small and stable.
**Worked example (one Q-learning update).** `Q(s,a)=0`, reward `−1`, `max Q(s',·)=2`, α=0.5, γ=0.9: target `=−1+0.9·2=0.8`; `Q ← 0+0.5(0.8−0)=0.4`.
**Value iteration (verified):** corridor of 5 states, reward 1 on reaching the right end, γ=0.9 ⇒ `V* = [0.729, 0.81, 0.9, 1.0, 0]` = `γ^{steps−1}`.
Assumptions: Markov property, stationary dynamics, adequate exploration, well-specified reward (reward hacking is a real failure).
Key issues: exploration–exploitation, credit assignment, sample inefficiency, instability of off-policy + function approximation + bootstrapping ("deadly triad"), reward design, sim-to-real gap.

## 5. Cost
Tabular: `O(|S||A|)` memory, cheap updates. Deep RL: millions–billions of environment steps, GPUs and simulators; PPO is relatively robust and widely used; SAC is sample-efficient for continuous control; offline RL learns from logged data without interaction (distribution-shift challenges).

## 6. When to use / not
Use: sequential decisions with delayed reward and a simulator/safe exploration; bandits for one-step choices (ads, UI variants, recsys exploration). Avoid: when supervised learning on logged expert data suffices, when mistakes are costly and no simulator exists, when the reward can't be specified, or when a simple heuristic/optimizer (OR methods) works. Consider imitation learning, offline RL, or contextual bandits first.

## 7. Hyperparameters
γ, learning rates, exploration schedule (ε, entropy bonus, noise), replay size/batch, target-update rate (τ), clip range ε (PPO), GAE λ, number of parallel envs, network size, reward scaling/normalization.

## 8. Implementation
All from scratch in NumPy (`rl_demo.py`). Core Q-learning/SARSA loop:
```python
target = r + (0 if done else gamma * (Q[s2][a2] if alg == "sarsa" else Q[s2].max()))
Q[s][a] += alpha * (target - Q[s][a])
```
Libraries: Gymnasium (environments), Stable-Baselines3 (PPO/SAC/DQN), CleanRL, RLlib. PyTorch/SB3 templates were not executed here.

## 9. Evaluation & results (measured)
**10-armed Gaussian bandit** (300 runs × 1000 steps; last 500 steps): % pulls of the optimal arm — greedy 70.0%, ε-greedy (0.1) 80.1%, UCB (c=2) 84.1%, **Thompson sampling 90.9%**; mean true value of the chosen arm: 1.450 / 1.383 / 1.493 / **1.539**. Greedy gets stuck early on a suboptimal arm in many runs; ε-greedy keeps wasting 10% on random pulls; UCB and Thompson explore in a directed way. (Single seeds per run set; absolute numbers depend on settings.)
**Cliff walking (4×12, ε=0.1, 30 seeds, 500 episodes):**
| Algorithm | Mean online return (last 100 episodes) | Greedy policy reaches goal | Median greedy path length |
|---|---|---|---|
| Q-learning | −48.8 | 30/30 | **13** (optimal, hugs the cliff) |
| SARSA | **−26.6** | 25/30 | 17 (safer route) |
The textbook result reproduced: Q-learning learns the *optimal* path but, because it keeps exploring with ε-greedy next to the cliff, falls off more often during training (lower online return); SARSA accounts for its own exploration and learns a safer path. (SARSA's greedy policy failed to reach the goal in 5/30 seeds within 100 steps at 500 episodes: not yet converged.)
**REINFORCE** on a 3-arm bandit (means 0.2, 0.5, 1.0): final policy `[0.001, 0.002, 0.996]` — concentrates on the best arm.
Real evaluation: average return with confidence intervals over many seeds, learning curves, sample efficiency, robustness to environment perturbations; report seeds (RL results are high-variance).

## 10. Case study (template)
Dynamic pricing/ads: start with a contextual bandit (LinUCB/Thompson) with off-policy evaluation on logs; only move to full RL if actions change future states (e.g. long-term retention). Always include a safe fallback policy and constraints.

## 11. Comparison
| | Q-learning/DQN | SARSA | Policy gradient/PPO | SAC/DDPG | Bandits |
|---|---|---|---|---|---|
| Actions | Discrete | Discrete | Discrete/continuous | Continuous | Discrete |
| Policy type | Off-policy value | On-policy value | On-policy policy | Off-policy actor-critic | One-step |
| Sample efficiency | Medium | Low | Low–medium | High | Very high |
| Stability | Tricky | Better | Good (PPO) | Good (with tuning) | Excellent |

## Interview Q&A
- *Q-learning vs SARSA?* Off-policy (max over next actions) vs on-policy (action actually taken); cliff example.
- *Why a replay buffer and target network in DQN?* Break correlation of samples and stabilize bootstrapped targets.
- *Why does PPO clip?* Prevents destructively large policy updates.
- *Exploration–exploitation?* ε-greedy, UCB, Thompson sampling, entropy bonuses, curiosity.
- *When use a bandit instead of RL?* Actions don't affect future states.

## Exercises
Implement Double Q-learning and compare overestimation; add a baseline to REINFORCE and plot variance; implement LinUCB on synthetic contextual data; run PPO on CartPole with SB3.

## Further reading
Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., free); Mnih et al. (2015) *Nature*; Schulman et al. (2017); Haarnoja et al. (2018); Spinning Up in Deep RL (OpenAI); Lattimore & Szepesvári, *Bandit Algorithms* (free).
