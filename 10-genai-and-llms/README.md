# Section 9 — Generative AI, LLMs and Agentic Systems

[← Roadmap](../README.md) · Prerequisite: [Section 9 — NLP](../09-nlp/README.md)

Same 16-section template as every other chapter ([`TEMPLATE.md`](../TEMPLATE.md)). Because this field changes fast, each chapter separates **durable concepts** (math, architectures, trade-offs) from **version-sensitive details** (model names, API parameters, prices), which are flagged and dated.

## Chapters (all written)

| # | Chapter |
|---|---|
| G.1 | [Generative vs. discriminative](01-generative-vs-discriminative.md) |
| G.2 | [Language models → LLMs](02-language-models-llms.md) |
| G.3 | [Prompt engineering](03-prompt-engineering.md) |
| G.4 | [RAG](04-rag.md) |
| G.5 | [Efficient adaptation](05-efficient-adaptation.md) |
| G.6 | [Alignment (SFT, RLHF, DPO)](06-alignment-rlhf-dpo.md) |
| G.7 | [Diffusion models](07-diffusion-models.md) |
| G.8 | [Multimodal](08-multimodal.md) |
| G.9 | [Tool calling](09-tool-calling.md) |
| G.10 | [Agents & agentic AI](10-agents-agentic-ai.md) |
| G.11 | [Evaluation, safety, LLMOps](11-evaluation-safety-llmops.md) |

Original plan (with lineage and trade-offs):

| # | Chapter | Core content | Evolution / lineage | Key trade-offs |
|---|---|---|---|---|
| 6.1 | **Generative vs. discriminative** | p(y\|x) vs p(x,y)/p(x); Naive Bayes vs logistic regression; likelihood-based (autoregressive, VAE, flow), implicit (GAN), score-based (diffusion) | Naive Bayes/GMM → VAE/GAN → diffusion/LLMs | Discriminative: better accuracy per label with enough data. Generative: sampling, missing data, semi-supervised use |
| 6.2 | **Language models → LLMs** | n-grams → neural LMs → word2vec → seq2seq+attention → Transformer → BERT (encoder) / GPT (decoder) / T5 (enc-dec); tokenization (BPE), next-token loss, scaling laws, context windows, KV cache, sampling (temperature, top-k/p) | RNN/LSTM → Attention → Transformer | Quadratic attention, hallucination, data/compute cost |
| 6.3 | **Prompt engineering** | Roles/instructions, zero/few-shot, chain-of-thought, structured output (JSON schema), self-consistency, prompt templates, prompt-injection basics; systematic prompt evaluation | Prompting is *not* training: no weights change | Cheap and fast, but brittle and model-specific; needs an eval set |
| 6.4 | **RAG** | Chunking, embeddings, vector search (ANN: HNSW, IVF), hybrid BM25+dense, reranking, query rewriting, citation, evaluation (retrieval recall/MRR, faithfulness, answer relevance); failure modes | Keyword search → dense retrieval (DPR) → RAG (Lewis et al. 2020) → agentic/iterative retrieval | Fresh, grounded, auditable knowledge vs. retrieval quality as ceiling, latency, chunking pitfalls |
| 6.5 | **Efficient adaptation** | Full fine-tuning; PEFT: adapters, prefix/prompt tuning, **LoRA** ($W+\tfrac{\alpha}{r}BA$), **QLoRA**; distillation; quantization (INT8/4, GPTQ/AWQ-style); pruning; inference efficiency (KV cache, batching, speculative decoding, FlashAttention, MoE) | Full FT → adapters → LoRA → QLoRA | Cost/memory ↓ vs. possible quality loss, forgetting, data requirements |
| 6.6 | **Alignment** | SFT/instruction tuning, RLHF (reward model + PPO), DPO and variants, constitutional/AI feedback | RL (Phase 5) → RLHF → preference optimization | Helpfulness vs. safety, reward hacking, data cost |
| 6.7 | **Diffusion models** | Forward noising, reverse denoising, DDPM loss, score view, classifier-free guidance, latent diffusion, samplers, flow matching, diffusion transformers | VAE/GAN → DDPM (Ho 2020) → latent diffusion → DiT/flow matching | Quality and stability vs. slow multi-step sampling |
| 6.8 | **Multimodal** | CLIP-style contrastive alignment, vision-language models (ViT encoder + LLM), image/audio/video generation, any-to-any; multimodal RAG | ViT + contrastive → VLMs → natively multimodal models | Alignment quality, hallucination about images, compute |
| 6.9 | **Tool calling** | JSON-schema tool definitions, model emits structured call → app executes → result returned; parallel calls, error handling, validation, structured outputs | Prompt-only → function calling → standardized protocols (e.g. MCP) | Reliability and security of arguments; model never executes code itself |
| 6.10 | **Agents / agentic AI** | The loop: observe → reason → act (tool) → observe; ReAct, planning, memory (short/long-term), reflection, workflows vs. autonomous agents, multi-agent patterns, human-in-the-loop, guardrails, cost/latency control | Chains → ReAct → tool-using agents → multi-agent systems | Autonomy vs. predictability; compounding errors; prompt injection; evaluation difficulty |
| 6.11 | **Evaluation, safety, LLMOps** | Benchmarks vs. task evals, LLM-as-judge (and its biases), regression test sets, tracing/observability, red-teaming, privacy, caching, routing, cost monitoring | — | Eval is the bottleneck for every other chapter |

## Choosing an LLM technique (decision guide, preview)

| Need | Try first | Escalate to |
|---|---|---|
| Better task behavior | Clearer prompt + few-shot examples | Fine-tune (LoRA) when prompts plateau and you have good data |
| Private / fresh knowledge | RAG | Fine-tune for *style/format*, not for facts that change |
| Strict output format | Structured output / schema | Constrained decoding, validation + retry |
| Needs live data or actions | Tool calling | Agent loop with guardrails |
| Fixed multi-step process | Deterministic workflow (chain) | Agent only if steps can't be predetermined |
| Latency/cost too high | Smaller model, caching, shorter prompts | Distillation, quantization |

Rule of thumb: **use the simplest system that meets the eval bar** — prompt → RAG → tools → workflow → agent → fine-tune, adding complexity only when measured results demand it.

## Priority within this section
🔴 6.1–6.4, 6.9–6.11 first (what most ML-engineer GenAI work looks like) · 🟠 6.5–6.8 next.
