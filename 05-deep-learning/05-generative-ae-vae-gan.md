# Chapter 5.5 — Generative Models: Autoencoders, VAE, GAN (and a pointer to Diffusion)

[← Roadmap](../README.md) · Prev [5.4](04-attention-transformers.md) · Next [5.6 Graph neural networks](06-gnn.md) · Priority 🟠 · Code: [`deep_learning_numpy.py`](../code/deep_learning_numpy.py) · Diffusion: [Ch 6.7](../10-genai-and-llms/README.md)

## 1. Identity & problem
Learn a model of the data distribution `p(x)` (or compressed representations) to **generate** new samples, denoise, compress, or detect anomalies.

## 2. History & relationships
Autoencoder (Rumelhart et al. 1986; deep AEs Hinton & Salakhutdinov 2006) → denoising/sparse/contractive AEs (Vincent et al. 2008) → **VAE** (Kingma & Welling 2013): probabilistic latent variable, trainable by backprop → **GAN** (Goodfellow et al. 2014): adversarial training for sharp samples → DCGAN, WGAN (Arjovsky 2017), StyleGAN (Karras 2018) → normalizing flows (Rezende & Mohamed 2015) → autoregressive models (PixelCNN; Transformers/LLMs) → **diffusion** (Sohl-Dickstein 2015; Ho et al. 2020), now dominant for images/audio/video. Not a single line: AE/VAE are likelihood-based, GAN is implicit, diffusion is score-based, AR models factorize the likelihood exactly.

## 3. Intuition
- **Autoencoder:** squeeze through a narrow code, reconstruct; the code is a learned compression.
- **VAE:** same, but the code is a *distribution*; a penalty keeps codes organized so you can sample from the prior and decode new data.
- **GAN:** a forger (generator) and a detective (discriminator) improve each other.

## 4. Math
**AE:** minimize `‖x − g(f(x))‖²`; with linear maps it recovers PCA subspace (Ch 4.3).
**VAE:** encoder `q_φ(z|x)=𝒩(μ,σ²)`, decoder `p_θ(x|z)`, prior `p(z)=𝒩(0,I)`. Maximize the **ELBO**: `log p(x) ≥ E_q[log p(x|z)] − KL(q(z|x)‖p(z))`. Reparameterization `z = μ + σ⊙ε`, `ε~𝒩(0,I)` makes sampling differentiable. KL for Gaussians: `½Σ(σ² + μ² − 1 − log σ²)`.
**Worked example (KL).** `μ=[1,0]`, `σ²=[1,4]`: dim1 `½(1+1−1−0)=0.5`; dim2 `½(4+0−1−ln4)=0.8069`; total KL **1.3069** (verified).
**GAN:** `min_G max_D E_x[log D(x)] + E_z[log(1−D(G(z)))]`. For fixed G the optimal discriminator is `D*(x)=p_data/(p_data+p_g)`; at the optimum `p_g=p_data`, `D*=1/2`, value `−log 4 ≈ −1.386`. In practice: non-saturating generator loss, spectral norm, WGAN-GP. Failure modes: mode collapse, unstable training, no likelihood to evaluate.
Assumptions: data lie near a low-dimensional manifold; sufficient capacity and data.

## 5. Training & cost
AE/VAE train stably with Adam; VAE balances reconstruction vs KL (β-VAE; KL annealing; posterior collapse risk). GANs need careful balance of G and D updates, normalization, tuned LRs. Sampling: VAE/GAN single forward pass (fast); diffusion needs many steps (slow, higher quality); autoregressive token by token.

## 6. When to use / not
| Need | Use |
|---|---|
| Compression, denoising, anomaly detection (reconstruction error) | (Denoising) autoencoder |
| Smooth latent space, semi-supervised, structured generation | VAE |
| Fast sharp samples, image-to-image translation (historically) | GAN |
| State-of-the-art image/audio/video generation | Diffusion (Ch 6.7) |
| Text/code generation | Autoregressive Transformers (Ch 6.2) |
Avoid: generative models when a discriminative model answers the question (Ch 6.1); GANs without capacity to tune/monitor; trusting generated data as real for evaluation leakage.

## 7. Hyperparameters
Latent dimension, architecture, β (KL weight), reconstruction loss (MSE/BCE/perceptual), LR, noise level (denoising), D/G update ratio, regularization (R1, spectral norm), batch size.

## 8. Implementation
Measured NumPy/sklearn comparison on digits (8-dimensional code, held-out test MSE, pixels in [0,1]): **PCA(8): 0.0246** vs sklearn MLP autoencoder (32-8-32, tanh): **0.0303**. The small nonlinear AE did *not* beat PCA with ~1.3k training images — nonlinear AEs need more data/tuning to pay off, and PCA is a strong baseline.
VAE template (PyTorch, not executed here):
```python
import torch, torch.nn as nn, torch.nn.functional as F
class VAE(nn.Module):
    def __init__(s, d=784, h=256, z=16): super().__init__(); s.e = nn.Linear(d, h); s.mu = nn.Linear(h, z); s.lv = nn.Linear(h, z); s.d1 = nn.Linear(z, h); s.d2 = nn.Linear(h, d)
    def forward(s, x):
        h = F.relu(s.e(x)); mu, lv = s.mu(h), s.lv(h); z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return torch.sigmoid(s.d2(F.relu(s.d1(z)))), mu, lv
def loss(xr, x, mu, lv): return F.binary_cross_entropy(xr, x, reduction="sum") - 0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
```

## 9. Evaluation
Reconstruction error/ELBO; for samples: FID and Inception Score (images), precision/recall of distributions, human preference studies; diversity vs fidelity; privacy/memorization checks. For anomaly detection: AUROC/PR-AUC on held-out anomalies.

## 10. Case study (template)
Industrial anomaly detection: train an autoencoder on normal images only; score by reconstruction error; compare with Isolation Forest on embeddings (Ch 4.5); calibrate the threshold on a validation set with some labeled defects.

## 11. Comparison
| | AE | VAE | GAN | Diffusion | Autoregressive |
|---|---|---|---|---|---|
| Likelihood | No | Bound (ELBO) | No | Bound/score | Exact |
| Sample quality | Not generative | Blurry | Sharp | Very high | High |
| Training | Easy | Easy | Hard | Moderate | Easy |
| Sampling speed | — | Fast | Fast | Slow | Slow (sequential) |

## Interview Q&A
- *Why the reparameterization trick?* Moves randomness to ε so gradients flow through μ,σ.
- *Why do VAE samples look blurry?* Pixelwise Gaussian likelihood averages plausible outputs; the KL regularization limits detail.
- *Mode collapse?* Generator produces limited variety because it can fool D with a few modes.
- *GAN vs VAE?* GAN: implicit, sharp, unstable; VAE: explicit bound, stable, blurrier.

## Exercises
Derive the Gaussian KL; show a linear AE recovers the PCA subspace numerically; train a tiny GAN on a 1-D mixture in NumPy.

## Further reading
Kingma & Welling (2013); Goodfellow et al. (2014); Arjovsky et al. (2017); Karras et al. (2019); Doersch (2016) "Tutorial on VAEs"; Lilian Weng "From autoencoder to Beta-VAE" and "What are diffusion models?".
