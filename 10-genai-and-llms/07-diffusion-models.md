# Chapter 6.7 — Diffusion Models

[← Section index](README.md) · Prev [6.6](06-alignment-rlhf-dpo.md) · Next [6.8 Multimodal](08-multimodal.md) · Priority 🟠 · Code: [`genai_demos.py`](../code/genai_demos.py) · Related: [Ch 5.5](../05-deep-learning/05-generative-ae-vae-gan.md)

## 1. Identity & problem
Generative models that learn to **reverse a gradual noising process**. Start from pure noise and denoise step by step into an image, audio clip, video, molecule or action trajectory. Dominant for high-fidelity image/video/audio generation.

## 2. History
Non-equilibrium thermodynamics-inspired diffusion (Sohl-Dickstein et al. 2015) → score matching with Langevin dynamics (Song & Ermon 2019) → **DDPM** (Ho, Jain & Abbeel 2020) → DDIM deterministic fast sampling (Song et al. 2020) → score-SDE unification (Song et al. 2021) → classifier guidance (Dhariwal & Nichol 2021) and **classifier-free guidance** (Ho & Salimans 2022) → **latent diffusion / Stable Diffusion** (Rombach et al. 2022: diffuse in a VAE latent space) → text-to-image systems (GLIDE, DALL·E 2, Imagen) → Diffusion Transformers (DiT, Peebles & Xie 2022) → flow matching / rectified flow (Lipman et al. 2022; Liu et al. 2022) → few-step distillation (progressive distillation, consistency models Song et al. 2023). Problems solved vs GANs: training stability, mode coverage; cost: slow multi-step sampling (addressed by fewer-step methods).

## 3. Intuition
Add a little Gaussian noise to a photo many times until it's static. Train a network to look at a noisy image and predict the noise that was added. Then run it backwards from static: remove predicted noise a bit at a time and an image emerges.

## 4. Math (DDPM)
Forward: `q(x_t|x_{t−1}) = 𝒩(√(1−β_t) x_{t−1}, β_t I)`. With `α_t=1−β_t`, `ᾱ_t=Πα_s`: **closed form** `x_t = √ᾱ_t x₀ + √(1−ᾱ_t) ε`, `ε~𝒩(0,I)`.
Training loss (simplified): `E_{x₀,t,ε} ‖ε − ε_θ(x_t,t)‖²`. Sampling step: `x_{t−1} = (1/√α_t)(x_t − (β_t/√(1−ᾱ_t)) ε_θ(x_t,t)) + σ_t z`.
Score connection: `ε_θ ≈ −√(1−ᾱ_t)·∇_x log p_t(x)`; the model estimates the direction toward higher data density.
**Worked schedule (executed):** T=200, linear β from 1e-4 to 0.05: `ᾱ` at t=1,50,100,200 = **0.9999, 0.7309, 0.2831, 0.0061**; final SNR `ᾱ_T/(1−ᾱ_T)=0.0062` — the data are essentially erased at T, so starting sampling from `𝒩(0,I)` is valid.
**Toy demonstration (executed).** Data: equal mixture of two Gaussians (means −2 and +2, std 0.3). For this distribution the diffused marginal and its score are known analytically, so we plug the **exact** `ε` predictor into the DDPM sampler (no neural network needed) and sample 20,000 points from noise. Result: 50.3% of samples near −2, 49.7% near +2; fitted left-mode mean −2.00, right-mode mean 2.00, std 0.30 — the reverse process recovers the original distribution, including bimodality. A real model learns `ε_θ` from data; the quality of the learned predictor limits sample quality.
**Classifier-free guidance:** `ε̂ = ε_θ(x_t,∅) + w(ε_θ(x_t,c) − ε_θ(x_t,∅))`; w>1 strengthens conditioning (prompt adherence) at the cost of diversity.
**Latent diffusion:** encoder compresses images by ~8× per side (e.g. 512×512×3 → 64×64×4), diffusion runs there, decoder reconstructs — roughly 48× fewer elements to process in that example (786,432 → 16,384).

## 5. Cost
Training: large-scale GPU compute and data. Sampling: one network evaluation per step (tens to hundreds; CFG doubles it); fast samplers (DDIM, DPM-Solver) and distillation cut to ~1–8 steps. Memory dominated by the denoiser (U-Net or Transformer) and text encoder.

## 6. When to use / not
Use: image/video/audio synthesis and editing (inpainting, super-resolution, image-to-image), data augmentation, molecular/structure generation, robotics policies (diffusion policy). Avoid: tasks needing exact likelihoods or very low latency without distillation; text generation (autoregressive models dominate, though discrete diffusion for text is researched); when provenance/IP/safety constraints aren't managed.

## 7. Strengths / weaknesses
+ State-of-the-art fidelity and diversity, stable training, flexible conditioning (text, image, control). − Slow sampling, heavy compute, prompt-adherence/anatomy/text-rendering failures, memorization of training data in some regimes, safety/IP concerns, evaluation difficulty.

## 8. Hyperparameters
Noise schedule (linear/cosine/shifted), timesteps T, prediction target (ε, x₀, or v), network (U-Net vs DiT), sampler and step count, guidance scale w, resolution, latent compression, EMA of weights, conditioning dropout (for CFG).

## 9. Implementation
Executed scalar/1-D version in `genai_demos.py`. Using a pretrained model (template; not executed here):
```python
from diffusers import AutoPipelineForText2Image
import torch
pipe = AutoPipelineForText2Image.from_pretrained("<model-id>", torch_dtype=torch.float16).to("cuda")
image = pipe("a watercolor fox in a forest", num_inference_steps=30, guidance_scale=7.0).images[0]
```
Check each model's license and usage policy.

## 10. Evaluation
FID (distribution distance; sensitive to feature extractor and sample size), CLIP score (text–image alignment; limited), precision/recall of distributions, human preference studies, task-specific checks (identity preservation, object counts), safety filtering, memorization tests, latency/VRAM.

## 11. Comparison
| | VAE | GAN | Autoregressive | Diffusion |
|---|---|---|---|---|
| Fidelity | Medium (blurry) | High, narrow | High | Very high |
| Mode coverage | Good | Weak (collapse) | Good | Good |
| Training stability | Good | Poor | Good | Good |
| Sampling | Fast | Fast | Slow (tokens) | Slow (steps) → few-step |

## Interview Q&A
- *What does the network predict?* The noise (or equivalently the clean sample/velocity) at a given noise level.
- *Why can you sample in closed form at any t during training?* The forward process is Gaussian, so `x_t` is a direct mix of `x₀` and noise.
- *What does guidance do?* Pushes samples toward the condition at the expense of diversity.
- *Why latent diffusion?* Much cheaper than pixel space with little perceptual loss.
- *Diffusion vs GAN?* Slower but more stable, better coverage.

## Exercises
Implement the closed-form forward process and visualize `x_t` for a 1-D dataset; train a tiny MLP `ε_θ(x,t)` on the two-mode mixture and compare with the exact-score sampler; sweep guidance scale on a pretrained pipeline and plot CLIP score vs diversity.

## Further reading
Ho et al. (2020) DDPM; Song et al. (2021) score-based SDEs; Ho & Salimans (2022) classifier-free guidance; Rombach et al. (2022) latent diffusion; Peebles & Xie (2022) DiT; Lipman et al. (2022) flow matching; Lilian Weng "What are diffusion models?"; Hugging Face Diffusion Models course.
