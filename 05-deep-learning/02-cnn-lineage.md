# Chapter 5.2 — Convolutional Networks: LeNet → AlexNet → VGG → Inception → ResNet → EfficientNet → Vision Transformers

[← Roadmap](../README.md) · Prev [5.1](01-mlp-backprop.md) · Next [5.3 Sequence models](03-sequence-models.md) · Priority 🔴 · Code: [`deep_learning_numpy.py`](../code/deep_learning_numpy.py) (conv demo; PyTorch snippets unexecuted)

## 1. Identity & problem
Networks for grid-structured data (images, spectrograms, some text/time series) using **local, weight-shared filters**. Exploit translation structure: a cat is a cat anywhere in the image.

## 2. History — the lineage with what each step changed
| Year | Model | Key advance | Remaining limitation |
|---|---|---|---|
| 1980 | Neocognitron (Fukushima) | Hierarchical local feature detectors | Not trained by backprop |
| 1989–98 | **LeNet-5** (LeCun) | Conv + pooling trained end-to-end by backprop; digit recognition | Small data/compute |
| 2012 | **AlexNet** (Krizhevsky, Sutskever, Hinton) | GPU training, ReLU, dropout, data augmentation; ImageNet error leap | Large dense layers; hand-tuned |
| 2014 | **VGG** (Simonyan & Zisserman) | Deep stacks of 3×3 convs | ~138M params (VGG-16), slow |
| 2014 | **Inception/GoogLeNet** (Szegedy et al.) | Multi-scale parallel branches, 1×1 bottlenecks, efficiency | Complex design |
| 2015 | **ResNet** (He et al.) | **Residual connections** `y=F(x)+x` make 50–150+ layer nets trainable | Still fixed-size design |
| 2017 | DenseNet, MobileNet (depthwise-separable convs), SENet | Feature reuse / efficiency / channel attention | — |
| 2019 | **EfficientNet** (Tan & Le) | Compound scaling of depth, width, resolution (NAS-found base) | Training cost |
| 2020 | **Vision Transformer (ViT)** (Dosovitskiy et al.) | Patches + Transformer; scales well with big data/pretraining | Needs lots of data or pretraining; less inductive bias |
| 2022 | ConvNeXt (Liu et al.) | Modernized CNN matching Transformers | — |
Pattern: each step fixed a *training or efficiency* bottleneck of the previous one. CNNs did not "lose" to ViTs outright; hybrids and modern CNNs remain competitive, especially with limited data/compute.

## 3. Intuition
Slide a small filter over the image; each position produces a "how much does this pattern appear here?" score. Early layers detect edges, later ones textures, parts, objects.

## 4. Math
2-D convolution (cross-correlation in practice): `Y[i,j] = Σ_{u,v} K[u,v]·X[i+u, j+v] + b`. Output size `⌊(W − K + 2P)/S⌋ + 1`. Parameters of a conv layer `K·K·C_in·C_out + C_out` — independent of image size. Pooling reduces resolution/adds invariance; receptive field grows with depth.
**Worked example.** Image `[[1,2,0,1],[3,1,2,0],[0,1,3,2],[1,0,2,1]]`, kernel `[[1,0],[0,−1]]` (stride 1, no padding) → 3×3 output `[[0,0,0],[2,−2,0],[0,−1,2]]` (top-left: `1·1 + 2·0 + 3·0 + 1·(−1) = 0`).
**Parameter efficiency (computed):** a 3×3 conv from 64→64 channels has **36,928** parameters; a fully connected layer mapping a 32×32×64 map to a same-size map would need **4.29×10⁹**.
**Residual block:** `y = ReLU(F(x) + x)` — gradient has an identity path `∂y/∂x = I + ∂F/∂x`, so signals/gradients flow through very deep nets; if the optimal map is near identity, F just learns ≈0.
Assumptions/inductive bias: locality, translation equivariance (not rotation/scale invariance), hierarchical composition.

## 5. Training & cost
SGD+momentum or AdamW, large batches, augmentation (crop, flip, color jitter, mixup/CutMix, RandAugment), label smoothing, BatchNorm, weight decay, cosine LR. FLOPs ∝ `H·W·K²·C_in·C_out`. **Transfer learning** (pretrain on ImageNet-scale, fine-tune) is the default workflow for small datasets.

## 6. When to use / not
Use: image classification/detection/segmentation, audio spectrograms, medical imaging, 1-D CNNs for signals/text, feature extractors. Prefer pretrained models/fine-tuning over training from scratch. Avoid: tabular data, tiny datasets without transfer learning, tasks needing global/long-range reasoning without attention (consider ViT/hybrids), strict latency on CPU without efficient variants (MobileNet, EfficientNet-Lite, quantization).

## 7. Strengths / weaknesses
+ Few parameters per capability, strong priors, mature tooling, fast inference. − Not rotation/scale invariant, adversarial sensitivity, needs lots of labeled data (or pretraining), texture bias, black-box.

## 8. Hyperparameters
Kernel size, channels, depth, stride/padding, pooling, normalization, augmentation strength, learning rate/schedule, weight decay, resolution, batch size, pretrained backbone choice, fine-tuning scope (freeze vs unfreeze, discriminative LRs).

## 9. Implementation
From scratch: the sliding-window loop above. PyTorch fine-tuning template (not executed here):
```python
import torch, torch.nn as nn, torchvision
m = torchvision.models.resnet18(weights="DEFAULT")            # pretrained backbone
m.fc = nn.Linear(m.fc.in_features, num_classes)               # new head
for p in list(m.parameters())[:-2]: p.requires_grad = False   # stage 1: train head only; later unfreeze with a small LR
```

## 10. Evaluation
Top-1/top-5 accuracy, per-class precision/recall (imbalance), confusion matrix, mAP/IoU for detection/segmentation, calibration, robustness tests (corruptions, distribution shift), Grad-CAM for sanity checks, latency/memory on target hardware. Hold out by **source** (patient/site/camera) to avoid leakage.

## 11. Case study (template)
Defect detection with ~2k labeled images: baseline = HOG/color features + logistic regression; model = pretrained ResNet-18/EfficientNet fine-tune with augmentation; report PR-AUC, threshold at required recall, error gallery. Expect transfer learning to beat scratch training by a wide margin at this data scale (a typical, not guaranteed, result—measure).

## 12. Comparison
CNN vs ViT: CNNs have inductive bias and win with less data; ViTs scale better with huge pretraining and unify modalities. CNN vs MLP: weight sharing and locality. Modern default: pretrained backbone (CNN or ViT) + task head.

## Interview Q&A
- *Why do convs work for images?* Locality + weight sharing → few params, translation equivariance.
- *What problem do residual connections solve?* Degradation/vanishing gradients in very deep nets via identity shortcuts.
- *Why 3×3 stacks (VGG)?* Same receptive field as large kernels with fewer params and more nonlinearity.
- *1×1 convolution?* Per-pixel channel mixing; bottlenecks and dimension reduction.
- *Equivariance vs invariance?* Conv is translation-equivariant; pooling/global pooling adds (approximate) invariance.

## Exercises
Compute output sizes and parameter counts for a given stack by hand; implement conv with `im2col`; fine-tune a small pretrained CNN on a 10-class subset and compare to training from scratch.

## Further reading
LeCun et al. (1998) "Gradient-based learning applied to document recognition"; Krizhevsky et al. (2012); Simonyan & Zisserman (2014); Szegedy et al. (2014); He et al. (2015) "Deep residual learning"; Tan & Le (2019); Dosovitskiy et al. (2020); Liu et al. (2022) "A ConvNet for the 2020s"; CS231n notes.
