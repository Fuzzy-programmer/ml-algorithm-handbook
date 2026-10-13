# Chapter 6.4 — Retrieval-Augmented Generation (RAG)

[← Section index](README.md) · Prev [6.3](03-prompt-engineering.md) · Next [6.5 Efficient adaptation](05-efficient-adaptation.md) · Priority 🔴 · Code: [`genai_demos.py`](../code/genai_demos.py)

## 1. Identity & problem
RAG gives a language model **external, up-to-date, private knowledge at inference time**: retrieve relevant passages, put them in the prompt, generate an answer grounded in them (and cite them). It addresses stale knowledge, hallucination on proprietary facts, and auditability without retraining.

## 2. History
Keyword retrieval (TF-IDF; **BM25**, Robertson & Zaragoza) → latent semantic indexing (Deerwester 1990) → dense retrieval (**DPR**, Karpukhin et al. 2020) → **RAG** (Lewis et al. 2020) → retrieve-then-read pipelines with rerankers (cross-encoders, ColBERT late interaction) → hybrid sparse+dense with Reciprocal Rank Fusion (Cormack et al. 2009) → query rewriting/HyDE → contextual chunking → **agentic RAG** (iterative retrieval via tool calls, Ch 6.9–6.10) and long-context alternatives. Each step addresses recall, precision or flexibility; none removes the retrieval-quality ceiling.

## 3. Pipeline
```
Ingest:  documents → clean/parse → chunk (+metadata) → embed → vector index (+ BM25 index)
Query:   user question → (rewrite) → retrieve top-K (sparse + dense) → fuse → rerank → select N chunks
Answer:  prompt = instructions + question + chunks (delimited, with IDs) → LLM → answer + citations → verify
```
- **Chunking:** by structure (headings/paragraphs) with overlap (e.g. 10–20%); chunk size trades precision (small) vs context (large); attach title/section/date metadata; contextual chunk headers help.
- **Embeddings:** bi-encoder maps text → vector; similarity = cosine/dot (Ch 0.1). Choose by retrieval benchmarks (MTEB) *and* your data; keep query/document prefixes if the model requires them.
- **Index:** exact for small corpora; ANN (HNSW, IVF-PQ; Ch 2.5) for millions; metadata filters for access control.
- **Hybrid + fusion:** `RRF(d)=Σ_rankers 1/(k+rank_r(d))`, k≈60. **Worked example:** doc A ranks 1st in BM25 and 5th in dense → `1/61+1/65=0.0318`; doc B ranks 3rd and 2nd → `1/63+1/62=0.0320`; B edges A since it is consistently high.
- **Reranking:** a cross-encoder scores (query, chunk) pairs jointly — more accurate, slower; apply to top 50–100.
- **Generation:** instruct to answer only from the context, say "not found" otherwise, cite chunk IDs; place important chunks at the start/end (long-context position effects).

## 4. Math
BM25 score: `Σ_{t∈q} IDF(t)·tf(t,d)(k₁+1)/(tf(t,d)+k₁(1−b+b·|d|/avgdl))`, with `IDF=ln((N−n_t+0.5)/(n_t+0.5)+1)`, typical `k₁=1.5, b=0.75`.
**Worked BM25 term:** N=1,000 docs, term appears in 10 → IDF≈4.6; tf=3, doc length = average, k₁=1.5 → saturation factor `3·2.5/(3+1.5)=1.667`; contribution ≈ 7.6. Raising tf from 3 to 30 gives `30·2.5/31.5=2.38` — diminishing returns by design.
Retrieval metrics: Recall@K, Precision@K, MRR, NDCG@K. End-to-end: faithfulness (is the answer supported by the context?), answer relevance, context precision/recall, citation accuracy.

## 5. Measured retrieval comparison (executed)
Setup: 8 newsgroup topics, ~4.5k training posts as the corpus; 300 held-out posts; query = first 12 words of a post; a retrieved document counts as "relevant" if it has the **same topic** (a weak, topic-level relevance proxy — not question answering). Metric: precision@10.
| Retriever | P@10 |
|---|---|
| Random | 0.122 |
| LSA-150 (dense, SVD of TF-IDF) | 0.383 |
| TF-IDF cosine | 0.400 |
| Hybrid RRF (BM25 + LSA) | 0.425 |
| **BM25** | **0.432** |
Reading: lexical BM25 is a strong baseline on short, keyword-rich queries; the hybrid did **not** beat BM25 here (the LSA component was weaker than neural embeddings would be — LSA is only a stand-in for a dense retriever; no neural embedding model was run). Hybrid typically helps when queries mix exact terms with paraphrase; verify on your data. Always compare against BM25 before paying for embeddings and a vector DB.

## 6. Failure modes and fixes
| Failure | Cause | Fix |
|---|---|---|
| Right doc never retrieved | Chunking splits answer; vocabulary mismatch | Better chunks, hybrid search, query rewriting, bigger K + rerank |
| Retrieved but ignored | Long/noisy context, position effects | Fewer, better chunks; rerank; instruct to quote evidence |
| Confident wrong answer from partial context | Model fills gaps | "Answer only from context," abstention, verification |
| Stale/conflicting sources | No metadata/versioning | Date/source metadata, freshness filters, conflict handling |
| Permission leaks | Index ignores ACLs | Filter by user entitlements at retrieval time |
| Multi-hop questions | Single retrieval | Iterative/agentic retrieval, decomposition, graph-based RAG |
| Prompt injection via documents | Untrusted text in context | Treat as data, sanitize, constrain tools (6.3, 6.11) |

## 7. When to use / not (and RAG vs alternatives)
| Need | Choice |
|---|---|
| Private/fresh/large knowledge base, citations | **RAG** |
| Behavior, tone, format consistency | Fine-tuning / prompting (6.5/6.3) |
| Corpus fits in context and queries are few | Long-context stuffing (+prompt caching) — simpler |
| Structured data / exact computation | Text-to-SQL, tools (6.9) |
| Reasoning across many documents | Agentic retrieval (6.10) |
Avoid RAG when the answer needs holistic reasoning over the whole corpus (summarization of everything) without map-reduce strategies, or when retrieval cannot be evaluated.

## 8. Hyperparameters
Chunk size/overlap, top-K, reranker depth, hybrid weights or RRF k, similarity threshold, embedding model/dimension, query rewrite strategy, number of chunks in context, generation instructions, citation format.

## 9. Implementation
BM25 from scratch (executed) is in `genai_demos.py`. Minimal template with an embedding library (not executed here):
```python
from sentence_transformers import SentenceTransformer
import numpy as np
emb = SentenceTransformer("all-MiniLM-L6-v2")                       # example open model
D = emb.encode(chunks, normalize_embeddings=True)                    # (n_chunks, dim)
def retrieve(q, k=5): s = D @ emb.encode([q], normalize_embeddings=True)[0]; return np.argsort(-s)[:k]
context = "\n\n".join(f"[{i}] {chunks[i]}" for i in retrieve(question))
prompt = f"Answer using ONLY the sources. Cite as [id]. If not found, say so.\n<sources>\n{context}\n</sources>\nQuestion: {question}"
```
Production: vector DB (FAISS, pgvector, Qdrant, Weaviate, Elasticsearch/OpenSearch hybrid), orchestration libraries (LlamaIndex, LangChain), or hosted retrieval tools.

## 10. Evaluation recipe
Build a **question set with gold evidence passages** (50–200 questions incl. unanswerable ones). Score retrieval (Recall@K, MRR) separately from generation (faithfulness, correctness, citation accuracy); use LLM-as-judge only after calibrating against human labels (6.11). Ablate: BM25 only, dense only, hybrid, +rerank, K, chunk size. Track latency and cost per query. Monitor online with user feedback and sampled audits.

## 11. Case study (template)
Internal policy assistant: parse PDFs by headings, chunk 300–500 tokens with titles, hybrid retrieval + cross-encoder rerank, answer with citations and abstention, ACL filters. Success = answer correctness ≥ target on the gold set, citation precision, p95 latency, and zero permission leaks in red-team tests.

## Interview Q&A
- *RAG vs fine-tuning?* RAG injects knowledge at inference (updatable, citable); fine-tuning changes behavior/style/weights.
- *Why hybrid search?* Sparse captures exact terms/IDs; dense captures paraphrase; fusion is more robust.
- *How evaluate a RAG system?* Separate retrieval and generation metrics; gold evidence sets; faithfulness checks.
- *What limits RAG quality?* Retrieval recall — the generator cannot use what isn't retrieved.
- *Why rerank?* Cross-encoders model query–doc interaction, improving precision of the final few chunks.

## Exercises
Reproduce the retrieval table and add a neural embedding retriever; test chunk sizes 100/300/800 words; build 20 unanswerable questions and measure abstention; implement RRF and tune k.

## Further reading
Lewis et al. (2020); Karpukhin et al. (2020); Robertson & Zaragoza (2009) "The probabilistic relevance framework: BM25 and beyond"; Cormack, Clarke & Buettcher (2009) RRF; Khattab & Zaharia (2020) ColBERT; Gao et al. (2023) "Retrieval-augmented generation for LLMs: a survey"; MTEB (Muennighoff et al. 2022).
