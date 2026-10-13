# Chapter 6.5 — Efficient Adaptation: Fine-Tuning, LoRA/QLoRA, Distillation, Quantization, Efficient Inference

[← Section index](README.md) · Prev [6.4](04-rag.md) · Next [6.6 Alignment](06-alignment-rlhf-dpo.md) · Priority 🟠 · Code: [`genai_demos.py`](../code/genai_demos.py)

> PEFT/quantization libraries and recipes evolve rapidly; the mathematics and trade-offs below are stable. Library snippets are templates (not executed here — no GPU/PyTorch in the authoring environment).

## 1. Problem
Adapt a large pretrained model to a task/domain/style, and make it cheap to train and serve. Full fine-tuning of a billion-parameter model needs gradients and optimizer states for every weight (Adam stores 2 extra copies) — typically ~12–16 bytes/parameter in mixed precision, i.e. >100 GB for a 7B-parameter model before activations.

## 2. Lineage
| Idea | Landmark | What it fixes | Cost |
|---|---|---|---|
| Full fine-tuning | BERT/GPT era | Max quality | Memory, per-task model copies, forgetting |
| Feature extraction / linear probe | Transfer learning | Cheap | Limited adaptation |
| **Adapters** | Houlsby et al. 2019 | Train small inserted modules | Extra inference latency |
| **Prefix / prompt tuning** | Li & Liang 2021; Lester et al. 2021 | Train soft prompts only | Less reliable on small models |
| **LoRA** | Hu et al. 2021 | Low-rank weight updates, **mergeable** (no latency) | Rank/targets to choose |
| **QLoRA** | Dettmers et al. 2023 | LoRA on a 4-bit quantized base (NF4, double quantization, paged optimizers) | Quantization error, slower step |
| Distillation | Hinton et al. 2015 | Small student mimics large teacher | Needs teacher outputs, quality gap |
| Quantization | LLM.int8() (Dettmers 2022), GPTQ (Frantar 2022), AWQ (Lin 2023), SmoothQuant | Smaller/faster inference | Accuracy loss at low bits |
| Efficient inference | FlashAttention (Dao 2022), PagedAttention/vLLM (Kwon 2023), speculative decoding (Leviathan 2023; Chen 2023), GQA/MQA, MoE (Shazeer 2017; Switch Transformer 2021) | Throughput/latency/memory | Engineering complexity |

## 3. LoRA mathematics
Freeze `W₀∈ℝ^{d×k}`; learn `ΔW = (α/r)·B·A` with `B∈ℝ^{d×r}`, `A∈ℝ^{r×k}`, `r ≪ min(d,k)`; init `B=0` (so training starts at the pretrained model) and `A~𝒩`. Forward: `h = W₀x + (α/r)BAx`. After training merge `W = W₀ + (α/r)BA`.
**Parameter count (executed):** a 4096×4096 matrix has **16.8M** parameters; LoRA rank 8 trains `2·4096·8 =` **65.5K** (0.39%). Applying LoRA to attention projections of every layer typically trains well under 1% of the model (exact fraction depends on targets/rank).
**Toy experiment (executed).** Learn a hidden weight update of true rank 4 on a 32×32 linear map from 200 samples (full update = 1,024 parameters):
| LoRA rank | Trainable params | Final MSE |
|---|---|---|
| 1 | 64 | 1.0003 (cannot represent the update) |
| 4 | 256 | 0.00000 |
| 8 | 512 | 0.00000 |
LoRA recovers the update exactly once rank ≥ true rank, using a quarter of the parameters; too low a rank cannot. Real-model updates are only *approximately* low-rank, so rank is tuned empirically.

## 4. Quantization mathematics
Symmetric uniform quantization to b bits: `s = max|w| / (2^{b−1}−1)`, `q = round(w/s)`, `ŵ = s·q`. Per-row/per-channel scales cut error when rows differ in scale. **Executed** on a 4096×256 Gaussian matrix with row-wise scale variation:
| Bits | Per-tensor RMSE | Per-row RMSE | Memory vs fp16 |
|---|---|---|---|
| 8 | 0.0221 | 0.0084 | 0.50× |
| 4 | 0.3880 | 0.1534 | 0.25× |
Takeaways: int8 is nearly lossless relative to weight scale; naive int4 error is large and per-row scaling helps ~2.5×; real 4-bit methods use group-wise scales, outlier handling, calibration data (GPTQ/AWQ) and non-uniform codebooks (NF4) to protect quality. Always evaluate task metrics after quantizing — weight error alone isn't enough.
Memory arithmetic: 7B parameters ≈ 14 GB (fp16), 7 GB (int8), ≈3.5–4.5 GB (4-bit incl. overhead).

## 5. Distillation
Student minimizes `α·CE(y, p_s) + (1−α)·T²·KL(softmax(z_t/T) ‖ softmax(z_s/T))`. For LLMs: train a small model on a teacher's generated outputs/reasoning (sequence-level distillation), often with filtering/verification. Check the teacher provider's terms of use before distilling from proprietary outputs.

## 6. When to adapt vs alternatives (decision guide)
| Need | Try first | Then |
|---|---|---|
| Better instructions/format | Prompting (6.3) | Few-shot examples |
| Private or changing knowledge | RAG (6.4) | Don't fine-tune facts that change |
| Consistent style/format/domain jargon, lower latency/cost at volume | LoRA / full fine-tune on curated examples | Distill to a smaller model |
| Behavior preferences/safety | Preference optimization (6.6) | — |
| Serving cost/latency | Smaller model, caching, batching, quantization | Speculative decoding, MoE |
Rule: fine-tune for **behavior**, retrieve for **knowledge**. Data quality dominates: hundreds to a few thousand high-quality, diverse examples often beat tens of thousands of noisy ones.

## 7. Hyperparameters
LoRA: rank `r` (8–64 typical starting range), `α` (often 2r or r), target modules (attention q,v or all linear layers), dropout; training: LR (1e-4–2e-4 for LoRA, lower for full FT), epochs (1–3; watch overfitting), batch size, warmup, sequence length, packing; quantization: bits, group size, calibration set.

## 8. Implementation templates (not executed)
```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)   # QLoRA
model = AutoModelForCausalLM.from_pretrained("your/base-model", quantization_config=bnb)
cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, task_type="CAUSAL_LM")
model = get_peft_model(model, cfg); model.print_trainable_parameters()
```
Hosted fine-tuning services exist for some providers; availability and supported models vary — check current docs. Numerical LoRA recovery and quantization are executed in `genai_demos.py`.

## 9. Evaluation
Held-out task metrics vs the prompted base model **and** the strongest prompting/RAG baseline; regression tests on general capability (catastrophic forgetting), safety behavior, calibration, latency/throughput, cost per 1k requests. For quantization: perplexity plus task accuracy and long-context behavior. Report variance across seeds.

## 10. Efficient inference toolbox (concepts)
Batching/continuous batching; KV-cache management (paged); GQA/MQA to shrink the cache (executed: 2.15 GB → 0.54 GB, Ch 6.2); FlashAttention to avoid materializing the n×n matrix; speculative decoding (small draft model proposes k tokens, big model verifies in one pass; output distribution preserved); prefix/prompt caching; model routing (small model first, escalate on low confidence); MoE (sparse activation).

## Interview Q&A
- *Why does LoRA work?* Task-specific weight updates have low intrinsic rank; learning `BA` captures them with few parameters.
- *Why init B to zero?* So `ΔW=0` at the start; training begins from the pretrained function.
- *QLoRA's trick?* Backprop through a frozen 4-bit NF4 base into 16-bit LoRA adapters, with double quantization and paged optimizers.
- *Fine-tune vs RAG for new facts?* RAG — facts change and need citations; fine-tuning teaches behavior and is poor at reliably injecting facts.
- *Quantization trade-off?* Memory/speed vs accuracy; lower bits need smarter schemes.

## Exercises
Compute the LoRA parameter fraction for a 7B model with r=16 on q,v; reproduce the rank-recovery experiment with noisy targets; quantize a real weight matrix per-row and per-group (size 64) and compare error; estimate serving memory for a 70B model at 4-bit.

## Further reading
Hu et al. (2021) LoRA; Dettmers et al. (2023) QLoRA; Houlsby et al. (2019); Hinton, Vinyals & Dean (2015); Frantar et al. (2022) GPTQ; Lin et al. (2023) AWQ; Dao et al. (2022) FlashAttention; Kwon et al. (2023) vLLM; Leviathan et al. (2023) speculative decoding; Hugging Face PEFT docs.
