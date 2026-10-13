# Chapter 5.3 — Sequence Models: RNN → LSTM/GRU → TCN → Seq2Seq

[← Roadmap](../README.md) · Prev [5.2](02-cnn-lineage.md) · Next [5.4 Attention & Transformers](04-attention-transformers.md) · Priority 🟠 · Code: [`deep_learning_numpy.py`](../code/deep_learning_numpy.py)

## 1. Identity & problem
Model ordered data (text, audio, time series, events) where context from earlier steps matters. Predict next element, classify a sequence, or map sequence → sequence (translation).

## 2. History and what each step fixed
| Year | Model | Fixes | Leaves |
|---|---|---|---|
| 1986–90 | Jordan/**Elman RNN** | Hidden state carries memory; weight sharing across time | Vanishing/exploding gradients; short memory (Hochreiter 1991; Bengio et al. 1994) |
| 1997 | **LSTM** (Hochreiter & Schmidhuber); forget gate (Gers et al. 2000) | Additive cell state with gates preserves gradients | Sequential compute, many parameters |
| 2014 | **GRU** (Cho et al.) | Fewer gates, similar performance | Same sequential bottleneck |
| 2014 | **Seq2Seq** (Sutskever et al.; Cho et al.) | Encoder–decoder for variable-length input/output | Fixed-size bottleneck vector |
| 2014 | **Attention** (Bahdanau et al.) | Decoder looks at all encoder states | Still recurrent |
| 2016–18 | WaveNet, **TCN** (Bai et al.) | Dilated causal convolutions: parallel training, long receptive field | Fixed receptive field |
| 2017 | **Transformer** (Vaswani et al.) | Drops recurrence; fully parallel (Ch 5.4) | Quadratic attention cost |
| 2021– | State-space models (S4, Mamba) | Linear-time long-range modeling | Newer, less mature ecosystem |
Siblings rather than a ladder: LSTMs/GRUs still serve small, streaming, low-latency and edge cases.

## 3. Intuition
An RNN reads left to right, updating a "memory" vector. LSTM adds valves: forget what's stale, write what's new, reveal what's relevant. Plain RNNs forget because gradients are multiplied by the same weights many times.

## 4. Math
Vanilla RNN: `h_t = tanh(W_h h_{t−1} + W_x x_t + b)`; output `y_t = W_y h_t`. BPTT unrolls time; gradient through T steps involves `Π ∂h_t/∂h_{t−1}`, which vanishes/explodes depending on the largest singular value of `W_h` and the activation derivative (clip gradients for exploding case).
**LSTM:** `f=σ(W_f[x,h]+b_f)`, `i=σ(W_i·)`, `o=σ(W_o·)`, `g=tanh(W_g·)`; `c_t = f⊙c_{t−1} + i⊙g`; `h_t = o⊙tanh(c_t)`. The additive `c_t` update gives a gradient highway when `f≈1`.
**GRU:** update gate `z`, reset gate `r`: `h_t = (1−z)⊙h_{t−1} + z⊙h̃_t`.
**Worked example (one LSTM step, scalars, all weights 1).** `x=0.5`, `h_{t−1}=0.1`, `c_{t−1}=0.2`. Pre-activation `0.6` for every gate: `f=i=o=σ(0.6)=0.6457`, `g=tanh(0.6)=0.5370`. `c_t = 0.6457·0.2 + 0.6457·0.5370 = 0.4759`; `h_t = 0.6457·tanh(0.4759) = 0.2860` (verified in code).
Param counts: LSTM layer with input d, hidden h: `4·h·(d+h+1)`; GRU `3·h·(d+h+1)`.
Training objectives: teacher forcing for generation; CTC for alignment-free speech; seq2seq cross-entropy.

## 5. Cost
Compute `O(T·h²)`, but **inherently sequential** in T (no parallelism across time) ⇒ slow to train on GPUs vs Transformers. Inference is cheap and constant-memory per step — good for streaming. Truncated BPTT limits memory.

## 6. When to use / not
Use: streaming/online inference, small datasets, low-latency edge devices, sensor/time-series with modest context, as a baseline. Avoid: very long dependencies, large-scale language modeling, tasks where parallel training matters — use Transformers/SSMs; for tabular lag features use boosted trees (Ch 6.1).

## 7. Strengths / weaknesses
+ Natural for sequences, variable length, constant-memory inference. − Hard to parallelize, long-range memory still limited, tricky to train (clipping, init), largely superseded for NLP.

## 8. Hyperparameters
Hidden size, layers, bidirectional, dropout (variational/recurrent), sequence length/truncation, gradient clipping (norm 1–5), embedding dimension, teacher-forcing ratio, optimizer/LR, batch size (pad + mask).

## 9. Implementation
NumPy: the LSTM step in `deep_learning_numpy.py`. PyTorch template (not executed here):
```python
import torch.nn as nn
class Net(nn.Module):
    def __init__(s, d, h, k): super().__init__(); s.rnn = nn.LSTM(d, h, num_layers=2, batch_first=True, dropout=0.2); s.out = nn.Linear(h, k)
    def forward(s, x): o, _ = s.rnn(x); return s.out(o[:, -1])        # last hidden state -> prediction
# clip: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

## 10. Evaluation
Task metrics (accuracy/F1, perplexity for language models, BLEU/ROUGE/chrF for generation, WER for speech, RMSE/MASE for forecasting). Use **chronological splits** for time series; compare with a naive baseline (last value/seasonal naive) and a lag-feature gradient-boosting model before trusting a deep model.

## 11. Case study (template)
Predictive maintenance on sensor streams: windowed features + LightGBM baseline vs 1-D CNN/LSTM on raw windows; measure recall at fixed false-alarm rate; examine latency on the edge device.

## 12. Comparison
RNN < LSTM/GRU on long context; GRU ≈ LSTM with fewer parameters; TCN is parallel and stable; Transformers dominate large-scale NLP; SSMs target linear-time long contexts.

## Interview Q&A
- *Why do RNNs have vanishing gradients?* Repeated multiplication by Jacobians with norm < 1 across time.
- *How does LSTM help?* Additive cell state and forget gate near 1 pass gradient unchanged.
- *LSTM vs GRU?* GRU merges gates/states — fewer params, often similar accuracy.
- *Why attention for seq2seq?* The fixed-size context vector bottlenecks long inputs; attention lets the decoder look back at all encoder states.

## Exercises
Compute an LSTM parameter count for d=128,h=256; implement a vanilla RNN forward/backward in NumPy and observe gradient norms vs sequence length; compare an LSTM with seasonal-naive on a periodic series.

## Further reading
Hochreiter & Schmidhuber (1997); Cho et al. (2014); Sutskever, Vinyals & Le (2014); Bahdanau, Cho & Bengio (2014); Bai, Kolter & Koltun (2018); Gu et al. (2021) S4; Olah, "Understanding LSTM Networks".
