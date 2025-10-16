# Part 2: Sparse Embedding Models

## Overview

Sparse embeddings output **high-dimensional vectors with mostly zeros** (99%+ sparsity). Unlike dense embeddings, sparse vectors are interpretable (non-zero dimensions correspond to specific terms/concepts) and compatible with traditional inverted indexes, combining neural semantic understanding with the efficiency of lexical search.

### Why Sparse?

**Advantages:**
1. **Interpretability:** Non-zero dimensions are explicit terms with weights
2. **Efficiency:** Inverted index compatible (proven at web scale)
3. **Generalization:** Better out-of-domain than dense models
4. **Hybrid potential:** Easy combination with BM25
5. **Scalability:** Standard IR optimizations apply

**Key Innovation:** Models like SPLADE use BERT's MLM (Masked Language Model) head to expand beyond input terms, achieving semantic understanding while maintaining sparse structure.

---

## 1. SPLADE Family (State-of-the-Art Sparse)

### 1.1 SPLADE v3

**HuggingFace ID:** `naver/splade-v3`

**Release Date:** March 2024

**Organization:** Naver Labs Europe

**Architecture Type:** SPLADE-style sparse neural encoder with term expansion

**Key Specifications:**
- **Sparsity Level:** ~99.82% (average 56 non-zero dimensions out of 30,522)
- **Vocabulary Size:** 30,522 (BERT WordPiece vocabulary)
- **Context Length:** 512 tokens
- **Base Model:** BERT (BertForMaskedLM)

**Paper:** arXiv:2403.06789

**How It Works:**
1. Input text → BERT encoder
2. MLM head → log(1 + ReLU(predictions)) for each vocab term
3. FLOPS regularization enforces sparsity
4. Output: sparse vector over full vocabulary

**Training Method:**
- Mix of KL-Div and MarginMSE loss
- 8 hard negatives per query from SPLADE++SelfDistil
- FLOPS regularization for sparsity control
- Term expansion via MLM head

**Use Cases:**
- First-stage retrieval
- Semantic search with keyword precision
- Efficient neural ranking
- Inverted index deployment

**Unique Features:**
- Automatic query and document expansion (neural term weighting)
- Interpretable weights (can see which terms matter)
- Better out-of-domain generalization than dense models
- Statistically significant improvement over SPLADE++

**Performance:**
- MS MARCO dev set: >40 MRR@10
- BEIR: Defeats all dense embedding methods
- Better generalization than dense models

**Availability:** HuggingFace, research license

---

### 1.2 SPLADE v3 Variants

#### SPLADE v3 DistilBERT
- **HuggingFace ID:** `naver/splade-v3-distilbert`
- **Key Feature:** Smaller, faster (DistilBERT base)
- **Use Case:** Resource-constrained deployments
- **Performance:** Comparable to SPLADE v3 with reduced cost

#### SPLADE v3 Lexical
- **HuggingFace ID:** `naver/splade-v3-lexical`
- **Key Feature:** Enhanced lexical matching, higher sparsity
- **Use Case:** High-precision keyword search with semantic understanding

#### SPLADE v3 Doc (Asymmetric)
- **HuggingFace ID:** `naver/splade-v3-doc`
- **Sparsity:** ~99.4% (184 avg non-zero for documents vs 56 for queries)
- **Key Feature:** Separate document encoder for faster query processing
- **Latency:** 4ms query encoding (nearly as fast as BM25)

---

### 1.3 Efficient SPLADE Series

#### Efficient SPLADE V Large (Doc/Query)
- **HuggingFace ID:** `naver/efficient-splade-V-large-doc`, `naver/efficient-splade-V-large-query`
- **Release Date:** 2024
- **Base Model:** BERT-large
- **Query Sparsity:** ~99.97% (7.7 avg non-zero dimensions)
- **Document Sparsity:** ~99.5%
- **Training:** L1 regularization for queries, FLOPS for docs
- **Performance:** <4ms latency difference vs BM25, superior accuracy

#### Efficient SPLADE VI BT Large
- **HuggingFace ID:** `naver/efficient-splade-VI-BT-large-query`, `naver/efficient-splade-VI-BT-large-doc`
- **Release Date:** 2024
- **Key Feature:** Version VI with Balanced Training, latest efficiency techniques
- **Performance:** Best-in-class SPLADE variant

---

### 1.4 SPLADE++ Models

#### SPLADE++ EnsembleDistil
- **HuggingFace ID:** `naver/splade-cocondenser-ensembledistil`
- **Base Model:** CoCondenser
- **Training:** Ensemble knowledge distillation, hard negative mining
- **Performance:** SOTA on BEIR out-of-domain tasks
- **Key Feature:** Benefits from ensemble teacher knowledge

#### SPLADE++ SelfDistil
- **HuggingFace ID:** `naver/splade-cocondenser-selfdistil`
- **Training:** Self-distillation (no teacher model needed)
- **Use:** Foundation for SPLADE v3, strong baseline

---

### 1.5 Commercial SPLADE Models

#### prithivida/Splade_PP_en_v1
- **HuggingFace ID:** `prithivida/Splade_PP_en_v1`
- **Release Date:** 2023-2024
- **License:** **Apache 2.0** (commercial-friendly!)
- **Key Feature:** Permissive licensing for commercial use
- **Integration:** FastEmbed, Qdrant
- **Performance:** Competitive with official SPLADE++

#### prithivida/Splade_PP_en_v2
- **HuggingFace ID:** `prithivida/Splade_PP_en_v2`
- **Release Date:** 2024
- **Key Feature:** Improved version, Apache 2.0
- **Performance:** Enhanced accuracy over v1

---

## 2. Learned Sparse Models

### 2.1 uniCOIL

**HuggingFace ID:** `castorini/unicoil-msmarco-passage` (via Pyserini)

**Release Date:** 2021 (NAACL 2021, still widely used in 2024-2025)

**Organization:** University of Waterloo (Castorini)

**Architecture Type:** Learned sparse with scalar term importance

**Key Specifications:**
- **Sparsity Level:** ~98-99% (20-200 non-zero terms per document)
- **Vocabulary Size:** 30,522 (BERT vocabulary)
- **Context Length:** 512 tokens
- **Base Model:** BERT-base

**Paper:** arXiv:2106.14807

**How It Works:**
1. BERT contextualizes input tokens
2. 2-layer neural network → scalar importance score per token
3. Only original tokens scored (no expansion)
4. Optional: DocT5Query expansion for documents

**Training:**
- MS MARCO passage ranking
- Relevance-based objectives
- Can use document expansion techniques

**Use Cases:**
- First-stage retrieval
- Efficient sparse retrieval
- Inverted index deployment

**Advantages:**
- Simpler than COIL (scalar scores vs 32-dim vectors per term)
- Inverted index compatible
- Strong performance, especially with document expansion
- Easier to implement than full multi-vector COIL

**Performance:**
- Strong MRR@10 on MS MARCO
- Efficient inference
- Good baseline for learned sparse methods

**Availability:** Via Pyserini, research implementations

**Why This Matters:** Simpler alternative to multi-vector sparse models, still effective.

---

### 2.2 miniCOIL-v1 (Qdrant)

**HuggingFace ID:** `Qdrant/minicoil-v1`

**Release Date:** 2024

**Organization:** Qdrant

**Architecture Type:** Ultra-lightweight sparse neural retriever

**Key Specifications:**
- **Sparsity Level:** Sparse across vocabulary, **4 dimensions per word stem**
- **Vocabulary:** Word stems (not full BERT vocab)
- **Context Length:** 512 tokens
- **Base Model:** Lightweight neural model

**Paper:** Qdrant blog post 2024

**How It Works:**
1. Tokenize to word stems (not WordPiece)
2. Generate 4-dimensional embedding per word stem
3. Sparse activation across vocabulary
4. Combines BM25 formula scaled by semantic similarity

**Training Method:**
- **Self-supervised** - NO labeled data needed!
- Learns from unlabeled text
- Contextual understanding of keywords

**Use Cases:**
- Hybrid search (better than BM25)
- Resource-constrained environments
- When labeled data is unavailable

**Advantages:**
- **No labeled training data required** (self-supervised)
- Slightly better than BM25 in 4/5 BEIR domains
- Understands contextual meaning of keywords
- Permissive licensing (Apache 2.0)
- Only 4-dim per term (ultra-compact)

**Performance:**
- Better than BM25 on BEIR (4/5 domains)
- Not biased to BEIR (trained without it)
- Efficient for hybrid search

**Availability:** HuggingFace, Apache 2.0 license

**Why This Matters:** Self-supervised sparse model without labeled data, beats BM25.

---

### 2.3 DeepImpact

**Implementation:** Research code (not directly on HuggingFace)

**Release Date:** 2021 (still referenced in 2024 production systems)

**Architecture Type:** Learned contextualized term weighting

**Key Specifications:**
- **Sparsity:** ~99%
- **Vocabulary:** 30,522
- **Context:** 512 tokens
- **Base:** BERT

**How It Works:**
- BERT embeddings → 2-layer NN → scalar weight per token
- Relevance-based training
- Context-aware term importance

**Advantages:**
- Context-aware weighting vs frequency-based (BM25)
- Compatible with standard inverted indexes
- Better than BM25 on semantic queries

**Use:** Referenced by Pinecone sparse implementation

---

### 2.4 Pinecone Sparse English v0

**Model ID:** `pinecone-sparse-english-v0`

**Release Date:** December 2024

**Organization:** Pinecone

**Architecture:** Based on DeepImpact approach

**Key Specifications:**
- **Sparsity:** High sparsity
- **Vocabulary:** Dynamic (whole-word tokenization, not WordPiece)
- **Context:** 512 tokens

**How It Works:**
- Whole-word tokenization preserves complete terms
- Context-aware lexical importance (DeepImpact-style)
- Dynamic vocabularies (content-dependent)

**Use Cases:**
- High-precision keyword search
- Hybrid search (sparse + dense)
- Production retrieval

**Advantages:**
- **44% better than BM25** on TREC DL (up to, 23% average)
- **24% better than BM25** on BEIR (up to, 8% average)
- Preserves complete terms (not subword pieces)
- Context-aware weighting

**Performance:**
- Significant improvement over BM25 on standard benchmarks
- Production-ready via Pinecone API

**Pricing:** $0.08 per 1M tokens

**Metric:** Dot product

**Availability:** Pinecone Inference API (proprietary)

---

## 3. OpenSearch Neural Sparse Series

### 3.1 OpenSearch Neural Sparse Encoding v2-distill

**HuggingFace ID:** `opensearch-project/opensearch-neural-sparse-encoding-v2-distill`

**Release Date:** August 2024

**Organization:** OpenSearch Project

**Architecture:** Distilled neural sparse encoder

**Key Specifications:**
- **Parameters:** 67M (50% reduction from v1's 133M)
- **Sparsity:** ~99.8%
- **Vocabulary:** 30,522
- **Context:** 512 tokens
- **Base:** Distilled BERT

**Training:**
- Knowledge distillation from larger model
- Pre-training on MS MARCO, eli5, squad_pairs, WikiAnswers, Yahoo, gooaq
- Supports doc-only and bi-encoder modes

**Performance:**
- **1.39x faster on GPU, 1.74x faster on CPU** than v1
- Better search relevance than v1
- Outperforms v1 on BEIR subset

**Availability:** HuggingFace, Apache 2.0

---

### 3.2 OpenSearch Neural Sparse Encoding v2-mini

**HuggingFace ID:** `opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini`

**Release Date:** August 2024

**Architecture:** Ultra-lightweight neural sparse (doc-only mode)

**Key Specifications:**
- **Parameters:** ~33M (75% reduction from v1's 133M)
- **Base:** MiniLM

**Performance:**
- **1.74x faster on GPU**
- **4.18x faster on CPU** than v1
- Smallest OpenSearch sparse model

**Use Case:** Ultra-efficient sparse search, resource-constrained environments

**Availability:** HuggingFace, Apache 2.0

---

## 4. Hybrid and Novel Sparse Approaches

### 4.1 SPLATE (Sparse Late Interaction)

**Organization:** Naver Labs Europe

**Release Date:** SIGIR 2024 (April 2024)

**Paper:** arXiv:2404.13950

**Architecture:** ColBERT + SPLADE fusion

**How It Works:**
- Freezes ColBERTv2 token embeddings
- Learns "MLM adapter" to map dense embeddings to sparse vocabulary space
- Late interaction with sparse representations

**Performance:**
- Can retrieve 50 documents in <10ms
- Same effectiveness as PLAID ColBERTv2
- CPU-friendly sparse candidate generation

**Why This Matters:** Bridges multi-vector (ColBERT) and sparse (SPLADE) paradigms.

---

### 4.2 PromptReps (LLM-Generated Sparse)

**Release Date:** EMNLP 2024

**Organization:** Academic research

**Paper:** arXiv:2404.18424

**Architecture:** LLM-generated dense + sparse hybrid

**How It Works:**
- Prompt LLM to represent text as single word
- Extract last token hidden states (dense)
- Extract logits over vocabulary (sparse)
- **Zero-shot** - no task-specific training

**Use Cases:** Zero-shot document retrieval without training data

**Performance:** Similar or better than trained LLM embeddings on MS MARCO, TREC DL, BEIR

**Why This Matters:** Zero-shot sparse embeddings from any LLM.

---

## 5. Sparse Model Comparison Table

| Model | Release | Organization | Sparsity | Vocab Size | Context | License | Key Feature |
|-------|---------|--------------|----------|------------|---------|---------|-------------|
| **SPLADE v3** | 2024-03 | Naver | 99.82% | 30,522 | 512 | Research | Latest SPLADE, >40 MRR@10 |
| **SPLADE v3-distil** | 2024 | Naver | 99.8% | 30,522 | 512 | Research | Faster, smaller |
| **Efficient SPLADE VI** | 2024 | Naver | 99.6% | 30,522 | 512 | Research | Best-in-class variant |
| **uniCOIL** | 2021 | Waterloo | 98-99% | 30,522 | 512 | Research | Simpler than COIL |
| **miniCOIL-v1** | 2024 | Qdrant | High | Stems | 512 | Apache 2.0 | Self-supervised, 4-dim |
| **prithivida/SPLADE v1** | 2023-24 | Independent | 99.8% | 30,522 | 512 | Apache 2.0 | Commercial use |
| **prithivida/SPLADE v2** | 2024 | Independent | 99.8% | 30,522 | 512 | Apache 2.0 | Improved v1 |
| **Pinecone Sparse** | 2024-12 | Pinecone | High | Dynamic | 512 | Proprietary | 44% > BM25 |
| **OpenSearch v2-mini** | 2024-08 | OpenSearch | 99.8% | 30,522 | 512 | Apache 2.0 | 75% param reduction |
| **OpenSearch v2-distil** | 2024-08 | OpenSearch | 99.8% | 30,522 | 512 | Apache 2.0 | 50% param reduction |
| **BGE-M3 (sparse)** | 2024-01 | BAAI | 99.9% | 250,002 | 8,192 | MIT | Dense+Sparse+Multi |
| **SPLATE** | 2024-04 | Naver | 99% | 30,522 | 512 | Research | Sparse late interaction |
| **PromptReps** | 2024 | Academic | LLM-dep | LLM-dep | Variable | Research | Zero-shot LLM sparse |

---

## Performance Highlights

**MS MARCO (MRR@10):**
- SPLADE v3: >40
- SPLADE++: ~38-39
- uniCOIL: Strong performance

**BEIR (Out-of-Domain):**
- SPLADE++ EnsembleDistil: Defeats all dense methods
- miniCOIL-v1: Better than BM25 in 4/5 domains
- SPLADE v3: Strong generalization

**Latency:**
- SPLADE v3 Doc: 4ms query encoding
- SPLATE: <10ms for 50 documents
- OpenSearch v2-mini: 4.18x faster than v1 on CPU

**Storage Efficiency:**
- All models: 99%+ sparsity
- Inverted index compatible
- Proven scalability to billions of documents

---

## Key Insights: Sparse Embeddings

1. **SPLADE dominates** - v3 is current state-of-the-art for learned sparse
2. **Commercial options exist** - prithivida variants (Apache 2.0), Pinecone API
3. **Self-supervised possible** - miniCOIL needs no labeled data
4. **Hybrid approaches emerging** - SPLATE combines sparse + late interaction
5. **LLMs as sparse models** - PromptReps shows zero-shot potential
6. **Efficiency improving** - OpenSearch v2 series shows 4x CPU speedups
7. **Better than BM25** - All neural sparse models outperform traditional lexical

---

*End of Part 2: Sparse Embedding Models*

**Continue to Part 3: Time Series Foundation Models...**
