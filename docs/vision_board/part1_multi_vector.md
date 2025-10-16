# The Embedding Models Vision Board
## A Comprehensive Survey of Exotic, Unusual, and Cutting-Edge Embedding Models (2024-2025)

**Compiled:** January 2025
**Focus:** NEW models and approaches from 2024-2025
**Scope:** Multi-vector, Sparse, Time Series, Dense, and Exotic Geometric Embeddings
**Total Models Documented:** 100+ models across all categories

---

## Table of Contents

1. [Introduction](#introduction)
2. [Multi-Vector Embeddings](#multi-vector-embeddings)
3. [Sparse Embedding Models](#sparse-embedding-models)
4. [Time Series Foundation Models](#time-series-foundation-models)
5. [Exotic and Geometric Embeddings](#exotic-and-geometric-embeddings)
6. [Dense Embedding Models](#dense-embedding-models)
7. [Implementation Roadmap](#implementation-roadmap)
8. [Conclusion](#conclusion)

---

## Introduction

This document serves as a comprehensive vision board for the embedding landscape in 2024-2025, focusing on models and approaches that are NOT widely available in existing libraries like FastEmbed or Embed Anything. The research emphasizes:

- **Recency:** Models released in 2024-2025 (old models can be found elsewhere)
- **Availability:** Actually usable implementations (not just theoretical papers)
- **Novelty:** Unique architectures and approaches
- **Diversity:** Spanning multiple modalities and geometric spaces

### Why This Matters

The embedding space is evolving rapidly beyond traditional dense single-vector approaches:

- **Multi-vector models** (ColBERT, ColPali) achieve state-of-the-art retrieval through token-level interactions
- **Sparse models** (SPLADE, uniCOIL) offer interpretability and efficiency with inverted index compatibility
- **Non-Euclidean geometries** (hyperbolic, spherical) provide massive capacity improvements for hierarchical data
- **Time series foundation models** (TimesFM, Granite) bring zero-shot capabilities to temporal data
- **Extreme quantization** (binary, 1-bit) enables billion-scale search with 32x compression

### Document Structure

Each section covers a major category of embedding models with detailed specifications including:
- Model names and HuggingFace IDs
- Release dates and organizations
- Architecture types and technical specifications
- Dimensions, context lengths, and parameter counts
- Unique features and use cases
- Performance metrics and benchmarks
- Availability and licensing

---

## Multi-Vector Embeddings

Multi-vector embeddings output **multiple vectors per input** (typically one per token) rather than pooling to a single vector. This preserves fine-grained semantic information and enables sophisticated matching mechanisms like MaxSim (late interaction).

### Why Multi-Vector?

**Traditional Dense Embeddings:**
```
Document → BERT → Pool (mean/CLS) → Single 768-dim vector
```

**Multi-Vector (ColBERT-style):**
```
Document → BERT → Linear Projection → N × 128-dim vectors (one per token)
```

**Key Advantage:** Token-level matching captures nuanced similarities that single vectors miss.

**Trade-off:** Higher storage cost (N vectors vs 1) offset by compression techniques.

---

### Category 1: ColBERT Variants (Text-Only)

#### 1.1 Jina-ColBERT-v2

**HuggingFace ID:** `jinaai/jina-colbert-v2`, `jinaai/jina-colbert-v2-96`, `jinaai/jina-colbert-v2-64`

**Release Date:** August 30, 2024

**Organization:** Jina AI

**Architecture Type:** ColBERT with modified XLM-RoBERTa backbone

**Key Specifications:**
- **Input Modalities:** Text only
- **Dimensions per Token:** 128 (default), 96, or 64 (Matryoshka variants)
- **Context Length:** 8,192 tokens
- **Number of Vectors:** Variable (one per token)
- **Base Model:** Modified XLM-RoBERTa (560M parameters)
- **Enhancements:** Rotary Position Embeddings (RoPE), Flash Attention

**Paper:** arXiv:2408.16672, MRL 2024 Workshop

**Use Cases:**
- Multilingual retrieval (89 languages)
- Long-document search (8K tokens)
- General-purpose semantic search
- Reranking pipelines

**Unique Features:**
- Matryoshka representation learning enables flexible embedding dimensions
- 50% storage reduction with 64-dim variant (only 1.5% performance loss)
- Supports 89 languages with strong performance on 30
- 8x longer context than original ColBERT (8192 vs 512)

**Performance:**
- BEIR average: 0.521 across 14 benchmarks (+6.5% over ColBERTv2)
- Outperforms BM25 across all MIRACL benchmark languages
- 128-dim: 0.521 nDCG@10
- 96-dim: 0.518 nDCG@10
- 64-dim: 0.513 nDCG@10

**Availability:** HuggingFace, Apache 2.0 license

---

#### 1.2 Answer.AI ColBERT Small (answerai-colbert-small-v1)

**HuggingFace ID:** `answerdotai/answerai-colbert-small-v1`

**Release Date:** August 13, 2024

**Organization:** Answer.AI (Jeremy Howard's lab)

**Architecture Type:** Miniaturized ColBERT

**Key Specifications:**
- **Input Modalities:** Text only
- **Dimensions per Token:** 96
- **Context Length:** 512 tokens
- **Number of Vectors:** Variable (one per token)
- **Base Model:** Mix of gte-small and bge-small-en-v1.5 (33M parameters)
- **Training Recipe:** JaColBERTv2.5 methodology

**Paper:** Blog post: https://www.answer.ai/posts/2024-08-13-small-but-mighty-colbert.html

**Use Cases:**
- Efficient retrieval on CPU
- Resource-constrained environments
- Edge deployment
- Fast prototyping

**Unique Features:**
- **Only 33M parameters** (3x smaller than ColBERTv2's 110M)
- Can search hundreds of thousands of documents in milliseconds on CPU
- Outperforms the 110M parameter ColBERTv2 on all benchmarks tested
- Proof of concept that small ColBERT models are viable

**Performance:**
- Outperforms e5-large-v2 and bge-base-en-v1.5 despite being much smaller
- Superior performance on LoTTe benchmark (unseen during training)
- Downloads: 1,247,942+ on HuggingFace

**Availability:** HuggingFace, Apache 2.0 license

**Implementation Status:** ✅ **Already implemented in Hypiler!** This is the model we successfully tested.

---

#### 1.3 GTE-ModernColBERT-v1

**HuggingFace ID:** Available via LightOn AI

**Release Date:** 2025 (based on ModernBERT released December 2024)

**Organization:** LightOn AI + Alibaba-NLP (GTE series)

**Architecture Type:** ColBERT with ModernBERT backbone

**Key Specifications:**
- **Input Modalities:** Text only
- **Dimensions per Token:** 128
- **Context Length:** 8,192 tokens
- **Number of Vectors:** Variable (one per token)
- **Base Model:** ModernBERT

**Paper:** Blog post: https://www.lighton.ai/lighton-blogs/lighton-releases-gte-moderncolbert

**Use Cases:**
- Long-document retrieval (8K context)
- Semantic search
- RAG (Retrieval-Augmented Generation) systems

**Unique Features:**
- First model to surpass ColBERT-small on BEIR benchmark
- Extended 8192-token context window (vs 512 for original ColBERT)
- Token-level matching maintains precision across long documents
- Optimized for Qdrant, LanceDB, Weaviate, Vespa

**Performance:**
- BEIR average: 54.89 (vs ColBERT-small: 53.79)
- LEMB Narrative QA Retrieval: 88.39 and 78.82
- Significantly outperforms voyage-multilingual-2 (79.17) and bge-m3 (58.73) on long-context tasks

**Availability:** HuggingFace, open source

---

#### 1.4 Reason-ModernColBERT

**HuggingFace ID:** `lightonai/Reason-ModernColBERT`

**Release Date:** 2025 (announced early 2025)

**Organization:** LightOn AI

**Architecture Type:** ColBERT optimized for reasoning-intensive retrieval

**Key Specifications:**
- **Input Modalities:** Text only
- **Dimensions per Token:** 128
- **Context Length:** 8,192 tokens (likely)
- **Number of Vectors:** Variable (one per token)
- **Base Model:** ModernBERT (150M parameters)

**Paper:** Blog post: https://www.lighton.ai/lighton-blogs/lighton-releases-reason-colbert

**Training:** Trained on reasonir-hq dataset, optimized for BRIGHT benchmark

**Use Cases:**
- Deep research requiring reasoning
- Reasoning-intensive retrieval (mathematical, scientific)
- Complex query understanding
- Academic/technical search

**Unique Features:**
- Specialized for reasoning-intensive retrieval (not just semantic similarity)
- Trained in less than 3 hours on 8 H100 GPUs (extremely efficient)
- Outperforms models 45x its size (effective parameter efficiency)
- First ColBERT variant specifically optimized for reasoning tasks

**Performance:**
- BRIGHT benchmark: Outperforms all models up to 7B parameters
- +2.5 NDCG@10 improvement over ReasonIR-8B on Stack Exchange splits
- State-of-the-art for reasoning-intensive retrieval at 150M parameters

**Availability:** HuggingFace, open source

---

#### 1.5 ColBERT-XM (Zero-Shot Multilingual)

**HuggingFace ID:** `antoinelouis/colbert-xm`

**Release Date:** February 2024

**Organization:** Academic research

**Architecture Type:** Modular ColBERT for zero-shot multilingual retrieval

**Key Specifications:**
- **Input Modalities:** Text (multilingual)
- **Dimensions per Token:** 128
- **Context Length:** 512 tokens
- **Number of Vectors:** Variable (one per token)
- **Base Model:** XMOD (modular transformer)

**Paper:** arXiv:2402.15059

**Use Cases:**
- Zero-shot multilingual information retrieval
- Low-resource language retrieval
- Cross-lingual search without parallel data

**Unique Features:**
- Learns from **single high-resource language** (English) data only
- Zero-shot transfer to multiple languages without retraining
- **Post-hoc language addition** without retraining entire model
- Modular architecture reduces "curse of multilinguality"
- Highly data-efficient and energy-efficient

**Performance:**
- Competitive with models trained on extensive multilingual datasets
- Effective adaptation to out-of-distribution data
- Demonstrates strong zero-shot transfer

**Availability:** HuggingFace, open source

---

#### 1.6 JaColBERTv2.5 (Japanese)

**HuggingFace ID:** Via Answer.AI

**Release Date:** August 2, 2024

**Organization:** Answer.AI

**Architecture Type:** ColBERT for Japanese language

**Key Specifications:**
- **Input Modalities:** Text (Japanese)
- **Dimensions per Token:** 128
- **Context Length:** 512 tokens
- **Number of Vectors:** Variable (one per token)
- **Base Model:** 110M parameters

**Paper:** Blog post: https://www.answer.ai/posts/2024-08-02-jacolbert-v25.html

**Use Cases:**
- Japanese language retrieval
- Multilingual search (Japanese component)
- Asian language semantic search

**Unique Features:**
- State-of-the-art Japanese retrieval model
- Trained with only 40% of data used by JaColBERTv2 (highly efficient)
- Trained in less than 15 hours on 4 A100 GPUs
- Post-training on high-quality Japanese datasets (v2.5 improvement over v2.4)

**Performance:**
- Outperforms bge-m3 on all Japanese benchmarks
- Best performance at any evaluation scale
- Demonstrates efficiency of Answer.AI training recipes

**Availability:** Available through Answer.AI

---

#### 1.7 Original ColBERTv2 (Baseline Reference)

**HuggingFace ID:** `colbert-ir/colbertv2.0`

**Release Date:** 2022 (included as foundational reference)

**Organization:** Stanford NLP (Omar Khattab)

**Architecture Type:** ColBERT (Contextualized Late Interaction over BERT)

**Key Specifications:**
- **Input Modalities:** Text only
- **Dimensions per Token:** 128
- **Context Length:** 512 tokens
- **Number of Vectors:** Variable (one per token)
- **Base Model:** BERT-base (110M parameters)

**Paper:** NAACL 2022

**Use Cases:**
- General-purpose neural search
- Baseline for all subsequent ColBERT variants

**Unique Features:**
- Introduced residual compression for storage efficiency
- Knowledge distillation training
- PLAID indexing engine for fast retrieval

**Performance:**
- Strong baseline for retrieval benchmarks
- Foundation for all modern ColBERT variants

**Availability:** HuggingFace, MIT license

**Implementation Status:** ✅ **Already implemented in Hypiler!** This is fully working with 128-dim output.

---

### Category 2: ColPali and Vision-Language Multi-Vector Models

These models extend the ColBERT late-interaction paradigm to **visual documents** (PDFs, images of pages, infographics). This is a major innovation from 2024.

#### 2.1 ColPali

**HuggingFace ID:** `vidore/colpali`, `vidore/colpali-v1.1`, `vidore/colpali-v1.2`

**Release Date:** July 2024 (v1.0), updated through v1.2

**Organization:** ViDoRe (Visual Document Retrieval) Research Group

**Architecture Type:** Vision-Language ColBERT (PaliGemma-based)

**Key Specifications:**
- **Input Modalities:** Visual documents (images of entire pages, PDFs, infographics)
- **Dimensions per Token:** Variable (per image patch)
- **Context Length:** Image-based (not text tokens)
- **Number of Vectors:** Multiple vectors per image patch (variable based on image resolution)
- **Base Model:** PaliGemma-3B
  - Vision: SigLIP-So400m (400M parameter vision encoder)
  - Text: Gemma-2B text decoder
- **Vision Patches:** 14×14 pixel patches

**Paper:** arXiv:2407.01449, Accepted at ICLR 2025

**Use Cases:**
- **OCR-free document retrieval** (no text extraction needed)
- PDF search (technical papers, reports, presentations)
- Visual RAG (retrieval-augmented generation with images)
- Infographic search
- Handwritten document retrieval

**How It Works:**
1. Input document page rendered as image
2. Vision encoder processes image into patch embeddings
3. Text query tokenized normally
4. **Late interaction:** MaxSim between query tokens and image patch embeddings
5. Highest scoring documents returned

**Unique Features:**
- **No OCR required** - works directly on document images
- **No chunking** - entire pages processed holistically
- Treats document pages as images end-to-end
- Introduced ViDoRe benchmark alongside the model
- Drastically simpler pipeline than OCR → chunk → embed → search

**Performance:**
- ViDoRe v1: 81.3 nDCG@5
- Outperforms traditional OCR-based pipelines on:
  - InfographicVQA
  - ArxivQA
  - TabFQuAD
- Faster and more accurate than OCR-based systems

**Availability:** HuggingFace, open source

**Why This Matters:** FastEmbed and Embed Anything do NOT have vision-document models. This is a completely new capability.

---

#### 2.2 ColQwen2

**HuggingFace ID:** `vidore/colqwen2-v0.1`, `vidore/colqwen2-v1.0`

**Release Date:** 2024

**Organization:** ViDoRe Research Group

**Architecture Type:** Vision-Language ColBERT (Qwen2-VL-based)

**Key Specifications:**
- **Input Modalities:** Visual documents
- **Dimensions per Token:** Variable (per image patch)
- **Context Length:** Image-based with dynamic resolution support
- **Number of Vectors:** Multiple per image patch
- **Base Model:** Qwen2-VL-2B-Instruct (2B parameters)
  - **1B fewer parameters** than PaliGemma (2B vs 3B)
- **Vision Patches:** Smaller than ColPali

**Paper:** Available through ColPali repository

**Use Cases:**
- Visual document retrieval
- Multimodal search
- Efficiency-focused visual RAG

**Unique Features:**
- **Smaller model** than ColPali (2B vs 3B parameters)
- **Smaller image patch size** → lower storage and compute requirements
- Dynamic resolution support
- **Apache 2.0 license** (more permissive than PaliGemma)
- Trained with larger batch size (256 vs ColPali's smaller batches)

**Performance:**
- **+5.3 nDCG@5 improvement over ColPali** on ViDoRe benchmark
- Better storage efficiency
- Better computational efficiency
- State-of-the-art for visual document retrieval

**Availability:** HuggingFace, Apache 2.0 license

**Why This Matters:** Outperforms ColPali while being smaller and more efficient.

---

#### 2.3 ColSmol

**HuggingFace ID:** `vidore/colSmol-256M`

**Release Date:** 2024

**Organization:** ViDoRe Research Group

**Architecture Type:** Small Vision-Language ColBERT

**Key Specifications:**
- **Input Modalities:** Visual documents
- **Dimensions per Token:** Variable (per image patch)
- **Context Length:** Image-based
- **Number of Vectors:** Multiple per image patch
- **Base Model:** ColIdefics3 (256M parameters)

**Paper:** GitHub: illuin-tech/colpali

**Use Cases:**
- Efficient visual document retrieval
- Resource-constrained environments
- Edge deployment for document search

**Unique Features:**
- **Very small model** (256M parameters - smaller than ColQwen and ColPali)
- Part of ColVision model family
- OCR-free visual retrieval
- Optimized for efficiency over absolute accuracy

**Performance:**
- Evaluated on ViDoRe benchmark
- Trade-off: smaller footprint vs ColPali/ColQwen accuracy

**Availability:** HuggingFace, open source

---

#### 2.4 ModernVBERT

**HuggingFace ID:** ModernVBERT organization on HuggingFace

**Release Date:** October 2024

**Organization:** Academic research

**Architecture Type:** Vision-Language Bidirectional Encoder for Document Retrieval

**Key Specifications:**
- **Input Modalities:** Vision + Text (multimodal)
- **Dimensions per Token:** Variable
- **Context Length:** Image-based
- **Number of Vectors:** Multiple (late interaction)
- **Base Model:**
  - Text: ModernBERT (150M parameters)
  - Vision: SigLIP2-16B-512 (100M parameters)
  - **Total: 250M parameters**

**Paper:** arXiv:2510.01149 (October 2024)

**Use Cases:**
- Visual document retrieval
- Multimodal search
- Document understanding

**Unique Features:**
- **State-of-the-art at 250M parameters** for document retrieval
- Principled recipe for improving visual document retrievers
- Attention masking optimization
- Enhanced with text-only training pairs (+1.7 nDCG@5 improvement)
- **MIT license** (fully open-source, unlike PaliGemma)

**Performance:**
- Outperforms models 10x larger when fine-tuned
- +1.7 nDCG@5 improvement from text-only augmentation
- State-of-the-art for its size category

**Availability:** HuggingFace, MIT license

---

#### 2.5 ColFlor

**HuggingFace ID:** `ahmed-masry/ColFlor`

**Release Date:** 2024

**Organization:** Academic research (Ahmed Masry)

**Architecture Type:** Efficient Vision-Language Document Retrieval

**Key Specifications:**
- **Input Modalities:** Vision + Text
- **Dimensions per Token:** Variable
- **Context Length:** Image-based
- **Number of Vectors:** Multiple per image patch
- **Base Model:** Florence-2-base (174M parameters)

**Paper:** OpenReview: https://openreview.net/forum?id=DrvZsa2GpN

**Use Cases:**
- OCR-free document retrieval
- Efficient visual search
- GPU-poor environments (works well on CPU)

**Unique Features:**
- **17x smaller than ColPali** (174M vs 3B parameters)
- **5.25x faster image encoding**
- **9.8x faster query encoding**
- Text-aware contextual embeddings using special OCR token
- Designed for efficiency and accessibility

**Performance:**
- Only 1.8% performance decrease vs ColPali on text-rich English documents
- **Outperforms ColPali** on TatDQA and Health datasets
- Comparable performance with massive efficiency gains

**Availability:** HuggingFace, open source

**Why This Matters:** Proves visual document retrieval doesn't require 3B parameter models.

---

### Category 3: Advanced Multi-Vector Architectures

#### 3.1 MetaEmbed (Meta AI)

**HuggingFace ID:** Not yet released (Meta research)

**Release Date:** September 2024 (paper)

**Organization:** Meta AI Research

**Architecture Type:** Flexible Late Interaction with Meta Tokens

**Key Specifications:**
- **Input Modalities:** Multimodal (text + vision)
- **Dimensions per Token:** Variable via Meta Tokens
- **Context Length:** Variable
- **Number of Vectors:** Flexible (test-time scalable)
- **Base Model:** Qwen2.5-VL (3B, 7B, 32B variants tested)

**Paper:** arXiv:2509.18095

**Use Cases:**
- Test-time scalable multimodal retrieval
- Quality-efficiency trade-off adjustment
- Multi-scale retrieval

**How It Works:**
- **Matryoshka Multi-Vector Retrieval (MMR):** Extends Matryoshka to multi-vector setting
- **Learnable Meta Tokens:** Compress variable-length representations to fixed budgets
- **Test-time scaling:** Adjust number of vectors used at inference (1-1024)
- **Coarse-to-fine:** Cascade retrieval from few vectors → many vectors

**Unique Features:**
- First to combine Matryoshka + Multi-Vector paradigms
- Test-time quality control (adjust budget based on needs)
- State-of-the-art at scale (up to 32B parameter models)
- Learned meta tokens rather than simple pooling

**Performance:**
- MMEB scores with (16, 64) budget:
  - 3B model: 69.1
  - 7B model: 76.6
  - 32B model: 78.7
- ViDoRe benchmark: State-of-the-art results
- Monotonic performance gains with increased budget and model scale

**Availability:** Research paper (implementation not yet public)

**Why This Matters:** Shows future direction for adaptive multi-vector retrieval.

---

#### 3.2 MM-Embed (NVIDIA)

**HuggingFace ID:** `nvidia/MM-Embed`

**Release Date:** November 4, 2024

**Organization:** NVIDIA

**Architecture Type:** Universal Multimodal Retriever with MLLM

**Key Specifications:**
- **Input Modalities:** Multimodal (text, images, and more)
- **Dimensions:** Based on NV-Embed architecture
- **Context Length:** Extended context
- **Number of Vectors:** Multiple (MLLM-based multi-vector approach)
- **Base Model:** Extension of NV-Embed-v1

**Paper:** arXiv:2411.02571, Accepted at ICLR 2025

**Use Cases:**
- Universal multimodal retrieval
- Cross-modal search (text→image, image→text)
- Mixed-modality queries

**Unique Features:**
- **First multimodal retriever achieving SOTA on M-BEIR**
- Modality-aware hard negative mining during training
- Continual fine-tuning preserves multimodal capability
- Zero-shot reranking with off-the-shelf MLLMs (no training needed)

**Performance:**
- UniIR benchmark: 52.7 (vs 48.9 previous best)
- MTEB text retrieval: 60.3 (improved from NV-Embed-v1's 59.36)
- State-of-the-art on multimodal M-BEIR benchmark

**Availability:** HuggingFace (NVIDIA), open source

---

#### 3.3 NV-Embed-v2 (with Latent Multi-Vector Attention)

**HuggingFace ID:** `nvidia/NV-Embed-v2`

**Release Date:** August 30, 2024

**Organization:** NVIDIA

**Architecture Type:** Generalist LLM-based embedding with latent attention

**Key Specifications:**
- **Input Modalities:** Text (primarily)
- **Dimensions:** 4096 total, compressed via 512 latent vectors
- **Context Length:** 32,768 tokens
- **Number of Vectors:** 512 latent vectors (via 8 attention heads × 64 per head)
- **Base Model:** Mistral-7B decoder (fine-tuned)

**Paper:** arXiv:2405.17428

**How It Works:**
- Input tokens → Mistral-7B processing
- **Latent attention layer:** 512 learnable latent vectors attend to all tokens
- Compresses variable-length token sequence → 512 fixed vectors
- Final pooling → single 4096-dim vector (or use 512 vectors for multi-vector mode)

**Use Cases:**
- General-purpose text embedding
- Retrieval for RAG
- Long document processing

**Unique Features:**
- **Latent attention mechanism** - novel approach to multi-vector compression
- Two-stage instruction tuning
- Novel hard-negative mining considering positive relevance scores
- Can be used as single-vector OR multi-vector
- **#1 on MTEB as of August 30, 2024**

**Performance:**
- MTEB score: **72.31** across 56 tasks (record-breaking at release)
- Retrieval sub-category: 62.65 across 15 tasks (#1 position)
- AIR Benchmark: Highest Long Doc scores

**Availability:** HuggingFace, open source

**Why This Matters:** Shows hybrid single/multi-vector approach via latent attention.

---

#### 3.4 BGE-M3 (Multi-Functionality: Dense + Sparse + Multi-Vector)

**HuggingFace ID:** `BAAI/bge-m3`

**Release Date:** January 30, 2024

**Organization:** Beijing Academy of Artificial Intelligence (BAAI)

**Architecture Type:** Unified model supporting THREE retrieval methods simultaneously

**Key Specifications:**
- **Input Modalities:** Text (100+ languages)
- **Dimensions:**
  - Dense: 1024
  - Sparse: Vocabulary-sized (~250K)
  - Multi-vector: Multiple × 1024 (ColBERT-style)
- **Context Length:** 8,192 tokens
- **Number of Vectors:** Variable (supports all three modes)
- **Base Model:** XLM-RoBERTa-large

**Paper:** arXiv:2402.03216

**Use Cases:**
- Multilingual retrieval (100+ languages)
- Hybrid search (combining dense + sparse + multi-vector)
- Versatile embedding for different retrieval scenarios

**How It Works:**
- Single model with three output heads:
  1. **Dense:** Standard pooling → 1024-dim vector
  2. **Sparse:** Linear + ReLU → vocabulary-sized sparse vector
  3. **Multi-vector:** Token-level embeddings (ColBERT-style)

**Unique Features:**
- **First model supporting all three retrieval paradigms in one**
- M3 = Multi-linguality, Multi-functionality, Multi-granularity
- Self-knowledge distillation training
- No additional computational cost for sparse weights when generating dense
- Can mix retrieval methods for hybrid search

**Performance:**
- SOTA on MIRACL (multilingual retrieval)
- SOTA on MKQA (cross-lingual retrieval)
- C-MTEB: Evaluated on 31 Chinese datasets
- Hybrid search (dense + sparse) significantly outperforms BM25

**Availability:** HuggingFace, MIT license

**Why This Matters:** Unifies three different embedding paradigms in a single model.

---

### Multi-Vector Model Comparison Table

| Model | Release | Modality | Dims/Token | Context | Params | Key Innovation |
|-------|---------|----------|------------|---------|--------|----------------|
| **ColPali** | 2024-07 | Vision+Text | Variable | Image | 3B | OCR-free PDF retrieval |
| **ColQwen2** | 2024 | Vision+Text | Variable | Image | 2B | Smaller, faster ColPali |
| **ColFlor** | 2024 | Vision+Text | Variable | Image | 174M | 17x smaller than ColPali |
| **Jina-ColBERT-v2** | 2024-08 | Text | 128/96/64 | 8192 | 560M | 89 languages, Matryoshka |
| **Answer.AI Small** | 2024-08 | Text | 96 | 512 | 33M | Outperforms 110M ColBERT |
| **GTE-ModernColBERT** | 2025 | Text | 128 | 8192 | 150M+ | ModernBERT backbone |
| **Reason-ModernColBERT** | 2025 | Text | 128 | 8192 | 150M | Reasoning-optimized |
| **ColBERT-XM** | 2024-02 | Text | 128 | 512 | N/A | Zero-shot multilingual |
| **MM-Embed** | 2024-11 | Multi | 4096 | 32K | 7B+ | Universal multimodal |
| **NV-Embed-v2** | 2024-08 | Text | 4096 | 32K | 7B | Latent attention (512 vectors) |
| **BGE-M3** | 2024-01 | Text | 1024 | 8192 | Large | Dense+Sparse+Multi unified |

---

### Key Insights: Multi-Vector Models

1. **Vision-language is the frontier** - ColPali family revolutionizing document retrieval
2. **Efficiency matters** - Smaller models (33M-256M) competitive with larger ones
3. **Context length growing** - 8192 becoming standard (vs 512 original)
4. **Matryoshka + Multi-vector** - Flexible dimensions for each token vector
5. **Late interaction proven** - MaxSim consistently beats single-vector approaches

---

*End of Part 1: Multi-Vector Embeddings*

**Document continues in subsequent sections...**
