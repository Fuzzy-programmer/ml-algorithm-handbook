# Chapter 5.6 — Graph Neural Networks: GCN, GraphSAGE, GAT, Message Passing

[← Roadmap](../README.md) · Prev [5.5](05-generative-ae-vae-gan.md) · Next [5.7 Self-supervised learning](07-self-supervised.md) · Priority 🟡 · Code: [`deep_learning_numpy.py`](../code/deep_learning_numpy.py)

## 1. Identity & problem
Learn from data shaped as graphs: molecules, social/citation networks, road networks, knowledge graphs, fraud rings, recommender interaction graphs. Tasks: node classification, link prediction, graph classification/regression.

## 2. History
Graph spectral theory and early GNN models (Gori 2005; Scarselli et al. 2009) → spectral CNNs (Bruna et al. 2013), ChebNet (Defferrard 2016) → **GCN** (Kipf & Welling 2016) simplification → **GraphSAGE** (Hamilton et al. 2017; inductive, sampling) → **GAT** (Veličković et al. 2017; attention over neighbors) → unifying **message-passing NN** framework (Gilmer et al. 2017) → **GIN** (Xu et al. 2018; expressive power ~ Weisfeiler–Leman test) → graph transformers, equivariant GNNs for 3-D molecules. Problems that motivated each step: scalability (sampling), inductiveness, anisotropic weighting, expressiveness, over-smoothing.

## 3. Intuition
Each node repeatedly collects its neighbors' feature vectors, mixes them, and updates itself. After k rounds a node "knows" its k-hop neighborhood.

## 4. Math
Message passing: `h_v^{(l+1)} = UPDATE(h_v^{(l)}, AGG({h_u^{(l)} : u ∈ N(v)}))`.
**GCN layer:** `H' = σ(Â H W)`, `Â = D̃^{-1/2}(A+I)D̃^{-1/2}` (self-loops + symmetric normalization). **GraphSAGE:** `h_v' = σ(W·[h_v ‖ mean(h_u)])` with neighbor sampling. **GAT:** attention coefficients `α_vu = softmax_u(LeakyReLU(aᵀ[Wh_v‖Wh_u]))`.
**Worked example (computed).** Path graph 1–2–3, `A=[[0,1,0],[1,0,1],[0,1,0]]`; `A+I` degrees (2,3,2). `Â=[[0.5,0.408,0],[0.408,0.333,0.408],[0,0.408,0.5]]`. Features `H=[[1,0],[0,1],[1,1]]`, `W=I`: `ÂH = [[0.5,0.408],[0.816,0.742],[0.5,0.908]]` — node 2 now mixes all three nodes (verified in code).
**Over-smoothing:** many layers average features until nodes are indistinguishable; **over-squashing:** long-range info compressed through bottlenecks. Remedies: residual/jumping connections, normalization, graph rewiring, transformers with positional/structural encodings.

## 5. Training & cost
Full-batch for small graphs (`O(|E|·d)` per layer); mini-batch with neighbor sampling (GraphSAGE), cluster-GCN, or subgraph sampling for large graphs. Semi-supervised node classification trains on few labeled nodes using the whole graph structure.

## 6. When to use / not
Use: relational/structured data where relationships carry signal—molecular property prediction, fraud rings, recommendations, traffic, knowledge graph completion. Avoid: when features alone suffice (compare to GBM on tabular + graph features such as degree/PageRank/node2vec embeddings), when the graph is noisy/arbitrary, or when simple label propagation already works.

## 7. Strengths / weaknesses
+ Uses structure explicitly, inductive variants generalize to new nodes. − Over-smoothing, scalability complexity, heterophily (neighbors differ) hurts basic GCNs, evaluation pitfalls (leakage across splits via edges), limited expressiveness beyond 1-WL.

## 8. Hyperparameters
Layers (2–3 typical), hidden size, aggregator (mean/sum/max/attention), sampling fanout, dropout, residuals, normalization, learning rate, readout for graph tasks (sum/mean/attention pooling).

## 9. Implementation
NumPy GCN layer verified above. PyTorch Geometric template (not executed here):
```python
import torch, torch.nn.functional as F
from torch_geometric.nn import GCNConv
class GCN(torch.nn.Module):
    def __init__(self, d, h, c):
        super().__init__(); self.c1 = GCNConv(d, h); self.c2 = GCNConv(h, c)
    def forward(self, x, edge_index):
        x = F.dropout(F.relu(self.c1(x, edge_index)), p=0.5, training=self.training)
        return self.c2(x, edge_index)
```
Libraries: PyTorch Geometric, DGL.

## 10. Evaluation
Node tasks: accuracy/F1 on public splits (Planetoid: Cora/Citeseer/Pubmed) but also random/ time-based splits; link prediction: AUC/AP/Hits@k with proper negative sampling and **no edge leakage**; graph tasks: scaffold splits for molecules (random splits overstate performance). Baselines: MLP on features, label propagation, node2vec+GBM.

## 11. Case study (template)
Transaction fraud: baseline LightGBM on tabular features; add graph features (degree, shared device/IP neighbors, PageRank); then GraphSAGE embeddings; evaluate on a forward-in-time split, measuring PR-AUC and investigator workload.

## 12. Comparison
GCN: simple, transductive-leaning. GraphSAGE: inductive, scalable. GAT: learned neighbor weights. GIN: expressive for graph classification. Graph Transformers: global attention, costlier.

## Interview Q&A
- *Why add self-loops and normalize?* Keep a node's own features and prevent high-degree nodes from dominating scale.
- *Over-smoothing?* Repeated averaging makes node embeddings converge.
- *Transductive vs inductive?* Test nodes seen at training time vs generalizing to unseen nodes/graphs.
- *Can GNNs beat feature-only models?* Only if neighbors add information—test against strong tabular baselines.

## Exercises
Compute two GCN layers by hand on a 4-node graph; implement mean aggregation in NumPy; compare Cora accuracy of MLP vs GCN.

## Further reading
Kipf & Welling (2016); Hamilton, Ying & Leskovec (2017); Veličković et al. (2017); Gilmer et al. (2017); Xu et al. (2018); Bronstein et al. (2021) "Geometric Deep Learning"; Hamilton, *Graph Representation Learning* (free).
