# Chapter 6.6 — Alignment: Instruction Tuning, RLHF, DPO, AI Feedback

[← Section index](README.md) · Prev [6.5](05-efficient-adaptation.md) · Next [6.7 Diffusion](07-diffusion-models.md) · Priority 🟠 · Code: [`genai_demos.py`](../code/genai_demos.py) · Builds on RL: [Ch 6.3 (specialized)](../06-specialized-ml/03-reinforcement-learning.md)

## 1. Problem
A pretrained LM predicts web text; it doesn't reliably follow instructions, refuse harmful requests, or match human preferences. **Alignment training** shapes behavior toward being helpful, honest and harmless (and policy-compliant) using demonstrations and preference signals.

## 2. Lineage
| Stage | Idea | Landmarks |
|---|---|---|
| SFT / instruction tuning | Train on (instruction, ideal response) | FLAN (Wei et al. 2021), T0, InstructGPT SFT (Ouyang et al. 2022) |
| **RLHF** | Learn a reward model from human preference comparisons, optimize the policy with RL under a KL leash | Christiano et al. 2017; Ziegler et al. 2019; Stiennon et al. 2020; InstructGPT 2022 (PPO) |
| **AI feedback** | Use model critiques/principles instead of (some) human labels | Constitutional AI (Bai et al. 2022), RLAIF |
| **Direct preference optimization** | Skip the reward model and RL loop; optimize a closed-form preference loss | DPO (Rafailov et al. 2023), IPO, KTO, ORPO |
| **RL with verifiable rewards** | Reward = checker result (unit tests, math answers) | GRPO (DeepSeekMath 2024) and reasoning-model training recipes |
Trade-offs: SFT is simple but limited by demonstration quality; RLHF can exceed demonstrations but is complex and unstable; DPO is simpler but offline and sensitive to data quality/distribution; verifiable-reward RL is powerful where checkers exist but narrow.

## 3. Mathematics
**Reward model (Bradley–Terry):** for a prompt with chosen `y_w` and rejected `y_l`: `P(y_w ≻ y_l) = σ(r(y_w) − r(y_l))`; loss `−log σ(r_w − r_l)`.
**Worked example (executed):** `r_w=1.2`, `r_l=0.4` → `σ(0.8)=0.690`; loss `0.371`.
**RLHF objective:** `max_π E_{x,y~π}[r(x,y)] − β·KL(π(·|x) ‖ π_ref(·|x))`; optimized with PPO (Ch 6.3 specialized). The KL term prevents drifting into reward hacking/gibberish.
**DPO:** the optimal policy of the KL-regularized objective satisfies `r(x,y) = β log(π*(y|x)/π_ref(y|x)) + const`, so
`L_DPO = −E[log σ(β[(log π_θ(y_w|x) − log π_ref(y_w|x)) − (log π_θ(y_l|x) − log π_ref(y_l|x))])]`.
**Worked example (executed):** β=0.1; log-probs policy chosen −10, rejected −12; reference chosen −11, rejected −11.5. Margin `=0.1[(−10+11)−(−12+11.5)] = 0.1(1+0.5)=0.15`; loss `=−log σ(0.15)=0.621` (vs 0.693 at margin 0). Training increases the policy's relative preference for the chosen answer.
**GRPO idea:** sample a group of answers per prompt, reward each (e.g. correct/incorrect), use group-normalized advantages `A_i=(r_i−mean)/std` — no learned value network.

## 4. Pipeline (typical)
Pretrain → SFT on high-quality demonstrations → collect preference data (pairs ranked by humans/AI against guidelines) → reward model and RL **or** DPO-style optimization → safety/red-team training → evaluation → iterate. Data quality, rater guidelines and diversity matter more than algorithm choice.

## 5. Known problems
Reward hacking/overoptimization (Gao et al. 2022: gold-reward falls as proxy-reward is over-optimized), sycophancy (agreeing with the user), verbosity bias, annotator disagreement/bias, mode collapse/reduced diversity, alignment tax (capability regressions), distribution shift between preference data and the deployed policy, evaluation difficulty. Safety training also needs adversarial testing (6.11).

## 6. When to use / not
Most practitioners **consume** aligned models; you align yourself when you control a base model and have preference data: product-specific style/policy, domain safety rules. Prefer prompting/RAG/SFT first. DPO-style methods are the practical first step for preference tuning; RL (PPO/GRPO) when you have a reliable reward/verifier. Avoid RLHF without a clear reward spec and eval — you will optimize the wrong thing.

## 7. Hyperparameters
β (KL strength; higher stays near reference), learning rate (very small, ~1e-6–5e-6 for full FT), epochs (often 1–3), batch size, preference-pair quality filters, PPO clip/KL target/rollout sizes, group size (GRPO), reference-model choice.

## 8. Implementation templates (not executed)
```python
# DPO with Hugging Face TRL (illustrative)
from trl import DPOTrainer, DPOConfig
cfg = DPOConfig(output_dir="out", beta=0.1, learning_rate=5e-6, num_train_epochs=1)
trainer = DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=pref_ds, processing_class=tok)  # dataset: prompt/chosen/rejected
trainer.train()
```
The scalar loss computation is executed in `genai_demos.py`.

## 9. Evaluation
Win-rate vs a reference model under blinded human or validated LLM judges (control for position and length bias), safety evals (refusal on harmful, non-refusal on benign — track *both* error types), capability regression tests, KL to reference, reward-model accuracy on held-out pairs (typical agreement with humans is far below 100% — preference data are noisy), red-team success rates.

## Interview Q&A
- *Why the KL term?* Keeps the policy near the reference to avoid reward hacking and degradation.
- *RLHF vs DPO?* RLHF learns a reward model then runs RL; DPO optimizes a closed-form classification-style loss directly on preference pairs.
- *What is reward hacking?* Exploiting reward model flaws to score high without genuinely better outputs.
- *Why sycophancy?* Human raters often prefer agreeable answers, which the optimizer then amplifies.
- *Constitutional AI?* Uses written principles and model self-critique to generate preference/supervision data.

## Exercises
Compute DPO loss for a batch by hand; simulate reward overoptimization with a noisy proxy reward; write rater guidelines for a toy task and measure inter-annotator agreement.

## Further reading
Christiano et al. (2017); Ouyang et al. (2022) InstructGPT; Bai et al. (2022a,b) HH-RLHF and Constitutional AI; Rafailov et al. (2023) DPO; Gao, Schulman & Hilton (2022) on scaling laws for reward-model overoptimization; Shao et al. (2024) DeepSeekMath/GRPO; Casper et al. (2023) "Open problems and fundamental limitations of RLHF".
