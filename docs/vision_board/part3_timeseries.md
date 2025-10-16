# Part 3: Time Series Foundation Models

## Overview

Time series foundation models bring the pre-training + zero-shot paradigm from NLP to temporal data. These models are trained on **billions to trillions of time points** and can forecast, classify, detect anomalies, or impute missing values without task-specific training.

### The Revolution (2024-2025)

**Pre-2024:** Time series models were task-specific, domain-specific, required labeled data

**2024-2025:** Foundation models emerge with:
- Zero-shot forecasting across domains
- Training on 100B - 1T time points
- Transformer and MLP architectures
- Context lengths: 512 → 16,384 time points

**Key Players:** Google (TimesFM), IBM (Granite), Amazon (Chronos), Salesforce (Moirai), Tsinghua (Timer/Sundial)

---

## 1. Foundation Models (Priority: Specifically Requested)

### 1.1 Google TimesFM (Time Series Foundation Model)

**Model IDs:**
- `google/timesfm-1.0-200m` (v1.0)
- `google/timesfm-2.0-500m-pytorch` (v2.0)
- TimesFM 2.5 (latest, 200M)

**Release Dates:**
- v1.0: February 2024
- v2.0: Mid-2024
- v2.5: September 2024

**Organization:** Google Research

**Architecture Type:** Decoder-only Transformer (GPT-style for time series)

**Model Versions:**
- **TimesFM 1.0:** 200M parameters, 512 context
- **TimesFM 2.0:** 500M parameters, 2,048 context
- **TimesFM 2.5:** 200M parameters, **16,384 context** (16K!)

**Technical Specifications:**
- **Embedding Dimensions:** 1,280
- **Number of Layers:** 20 (v1.0), 50 (v2.0)
- **Patch Size:** Input 32, Output 128
- **Maximum Forecast Horizon:** 256 time points
- **Architecture:** Stacked transformer decoder blocks with causal attention

**Pre-training Data:** 100 billion real-world time points from diverse domains

**Input Format:** Univariate time series, arbitrary length

**Zero-shot Capability:** Yes - strong zero-shot performance without fine-tuning

**Use Cases:**
- Time series forecasting (retail, finance, web traffic, energy)
- Point predictions
- Probabilistic predictions with uncertainty
- Cross-domain generalization

**How It Works:**
1. **Patching:** Time series divided into patches (treated as tokens)
2. **No discrete tokenization:** Continuous values (unlike Chronos)
3. **Decoder-only:** Autoregressively predicts future patches
4. **Just-in-time inference:** Milliseconds per forecast

**Unique Features:**
- Patching approach (patches = tokens, no value quantization)
- State-of-the-art zero-shot forecasting
- **Currently leads GIFT-Eval benchmark** (v2.5)
- Available in Google BigQuery for cloud deployment
- Fast inference (milliseconds)

**Performance:**
- Competitive with supervised models despite zero-shot
- 5-10x faster than transformer baselines
- GIFT-Eval: Rank #1 (v2.5)

**Availability:**
- HuggingFace (open-weights)
- GitHub: google-research/timesfm
- Google BigQuery integration

**Paper:** ICML 2024

---

### 1.2 IBM Granite Time Series Family

**Repository:** GitHub: ibm-granite/granite-tsfm

**Release Date:** 2024 (multiple releases), Updated October 2024

**Organization:** IBM Research

**Availability:** HuggingFace (`ibm-granite/granite-timeseries-*`)

The Granite time series family consists of three complementary models, each with different architectures and strengths:

---

#### 1.2.1 TinyTimeMixer (TTM)

**Model Sizes:**
- TTM_B, TTM_E, TTM_A variants
- **Less than 1M parameters** (introduces "tiny" pre-trained concept)
- TTM r2: ~700M training samples
- TTM r2.1: ~1B training samples

**Architecture Type:** Lightweight TSMixer-based (MLP, NO attention mechanism)

**Mechanism:**
- Fully-connected layers only
- Channel-independent and channel-mixing approaches
- **Decoder Channel-Mixing** during fine-tuning

**Technical Innovations:**
- **Adaptive patching:** Dynamic patch sizes
- **Diverse resolution sampling:** Training across time scales
- **Resolution prefix tuning:** Adapt to different sampling frequencies

**Input Format:** Multivariate time series, flexible length

**Zero-shot Capability:** Yes - state-of-the-art zero-shot forecasting

**Use Cases:**
- Multivariate time series forecasting
- Few-shot learning (competitive with 5% training data)
- Channel-mixing for correlated variables
- **CPU-only capable** (resource-constrained environments)

**Unique Features:**
- **10x smaller than competing foundation models** yet outperforms them
- Outperforms billion-parameter models from Google and Alibaba
- 2-3x faster execution than transformers
- 2-3x lower memory usage
- Competitive with only 5% training data (highly sample-efficient)

**Performance:**
- Outperforms baselines by 4-40% in zero/few-shot settings
- NeurIPS 2024
- Beats TimesFM, Chronos on efficiency metrics

**Availability:** HuggingFace, Apache 2.0 license

---

#### 1.2.2 FlowState

**Architecture Type:** State Space Model (SSM) + Functional Basis Decoder

**Key Innovation:** Time-scale adjustable foundation model

**Technical Specifications:**
- SSM architecture (not transformer)
- Flexible forecasting across temporal scales
- Seamless timescale shifting

**Unique Features:**
- Analyze at one timescale, predict at another
- **Ranked #2 on GIFT-Eval leaderboard** for zero-shot forecasting
- State space models provide different inductive biases than transformers

**Use Cases:**
- Multi-resolution forecasting
- Cross-scale analysis
- Long-range dependencies

**Performance:** Strong on GIFT-Eval, complementary to transformer approaches

---

#### 1.2.3 TSPulse

**Model Size:** 1M parameters (ultra-lightweight)

**Architecture Type:** Dual-space masked reconstruction (time + frequency domains)

**How It Works:**
- Joint learning from **time domain** and **frequency domain** (FFT)
- Shared embedding space
- Masked reconstruction in both domains simultaneously
- GPU-free inference

**Use Cases:**
- Forecasting
- **Classification**
- **Anomaly detection** (SOTA)
- **Imputation**
- Similarity search

**Unique Features:**
- **Ultra-lightweight** (1M parameters)
- **Dual-space approach** leverages complementary time/frequency signals
- **GPU-free** inference possible
- Multi-task capable (rare for time series models)

**Performance:**
- **TSB-AD benchmark:** Outperforms statistical models by 24%, larger foundation models by 33%
- **State-of-the-art for anomaly detection**
- Demonstrates that 1M parameter models can compete with large models on specific tasks

**Paper:** arXiv:2505.13033

**Availability:** HuggingFace (`ibm-granite/granite-timeseries-tspulse`), Apache 2.0

---

**Summary:** IBM Granite offers **three complementary models** - TTM for efficiency and multivariate, FlowState for timescale flexibility, TSPulse for multi-task with dual-space learning. All are tiny (<1M to few M params) yet highly effective.

---

### 1.3 Amazon Chronos Family

**Repository:** GitHub: amazon-science/chronos-forecasting

**Release Dates:**
- Original Chronos: March 2024
- Chronos-Bolt: November 2024

**Organization:** Amazon Science

**Availability:** HuggingFace (`amazon/chronos-*`), AWS SageMaker JumpStart

---

#### 1.3.1 Original Chronos

**Architecture Type:** T5-based Transformer (repurposed language model)

**Model Sizes:** Based on T5
- t5-tiny to t5-large
- Largest: ~710M parameters

**Key Innovation:** **Tokenization of time series**
- Time series values → discrete tokens (via quantization)
- Treat time series as "language" of numbers
- Train with T5 seq2seq objective

**Input Format:** Univariate and multivariate time series

**Zero-shot:** Yes - framework for zero-shot probabilistic forecasting

**Unique Features:**
- First to successfully apply language model architecture to time series
- Probabilistic predictions with uncertainty intervals
- Vocabulary: 4,096 tokens (reduced from T5's 32,128)

**Performance:** Strong zero-shot forecasting, basis for Chronos-Bolt

---

#### 1.3.2 Chronos-Bolt

**Release Date:** November 2024

**Model Sizes:**
- Tiny: 9M parameters
- Mini: 21M parameters
- Small: 48M parameters
- Base: 205M parameters

**Key Innovation:** Optimized architecture and training

**Technical Specifications:**
- Vocabulary: 4,096 tokens
- Context: 512 observations (benchmarks)
- Prediction Horizon: 64 steps (benchmarks)
- Pre-training: **Nearly 100 billion time series observations**

**Performance:**
- **250x faster** than original Chronos
- **20x more memory-efficient**
- **5% lower forecasting error**

**Use Cases:**
- Probabilistic forecasting
- Zero-shot predictions
- Uncertainty quantification
- AWS cloud deployment

**Unique Features:**
- Massive speedup over original while improving accuracy
- Efficient enough for real-time applications
- Integrated with AWS SageMaker

**Paper:** AWS ML Blog, arXiv 2403.*

**Availability:** HuggingFace, AWS SageMaker, Apache 2.0

---

### 1.4 Nixtla TimeGPT

**Model ID:** `timegpt-1` (API only, closed-source)

**Release Date:** October 2023 (first foundation model), continued updates 2024

**Organization:** Nixtla (Commercial)

**Architecture Type:** Custom encoder-decoder transformer (NOT based on GPT/LLM)

**Model Size:** Not publicly disclosed (proprietary)

**Technical Specifications:**
- Self-attention encoder-decoder with residual connections
- Layer normalization
- Linear output layer for forecasting window
- **Optimized for forecasting, NOT language modeling**

**Pre-training Data:**
- Over 100 billion data points
- Domains: Finance, economics, demographics, healthcare, weather, IoT, energy, web traffic, sales, transport, banking

**Input Format:** Univariate and multivariate time series

**Zero-shot:** Yes - first commercial zero-shot time series API

**Use Cases:**
- Time series forecasting
- Anomaly detection
- Cross-domain predictions
- Production forecasting at scale

**Unique Features:**
- **First time series foundation model** (October 2023 - pioneering)
- Purpose-built for time series (not adapted from NLP)
- Commercial API service
- Proprietary architecture

**Performance:**
- Ranks in top 3 performers across 300K+ unique series
- Zero-shot inference: 0.6ms per series on GPU
- Outperforms statistical and deep learning baselines

**Pricing:** Commercial API (subscription-based)

**Paper:** arXiv:2310.03589

**Availability:** Nixtla API (https://docs.nixtla.io/), closed-source

---

### 1.5 Salesforce Moirai Family

**Repository:** GitHub: SalesforceAIResearch/uni2ts

**Release Dates:**
- Original Moirai: March 2024
- Moirai-MoE: October 2024

**Organization:** Salesforce AI Research

**Availability:** HuggingFace (`Salesforce/moirai-*`)

---

#### 1.5.1 Original Moirai

**Architecture Type:** Masked Encoder Transformer

**Model Sizes:** Small, Base, Large variants

**Pre-training Data:**
- 27 billion observations
- 9 distinct time series domains

**Input Format:** **Any-variate** (flexible variable count), multiple frequencies

**Zero-shot:** Yes - universal zero-shot forecasting

**Novel Concepts:**
- **Multi Patch Size Projection Layers:** Handle different temporal resolutions
- **Any-Variate Attention:** Variable number of time series
- **Mixture Distribution:** Probabilistic outputs

**Use Cases:**
- Universal forecasting (any domain)
- Multivariate time series
- Multiple sampling frequencies

---

#### 1.5.2 Moirai-MoE (Mixture-of-Experts)

**Release Date:** October 2024

**Model Sizes:**
- **Moirai-MoE-Small:** 11M active parameters
- **Moirai-MoE-Base:** 86M active parameters
  - **3x larger in total than Moirai-Large** but only activates subset

**Architecture Type:** Decoder-only with **Mixture-of-Experts (32 experts)**

**How It Works:**
- Single input/output projection layer
- Sparse MoE transformers (32 expert networks)
- **Token-level routing:** Each token routed to subset of experts
- Only 2-4 experts activated per token (sparse activation)

**Unique Features:**
- **First mixture-of-experts time series foundation model**
- Token-level model specialization (data-driven)
- Each expert specializes in unique time series characteristics
- Sparse activation (only subset activated per token)
- 65x fewer activated parameters than dense models

**Performance:**
- **Moirai-MoE-Small:**
  - 17% better than Moirai-Small
  - 8% better than Moirai-Base
  - 7% better than Moirai-Large
- Outperforms Chronos and TimesFM with **65x fewer activated parameters**

**Paper:** arXiv:2410.10469

**Availability:** HuggingFace, Apache 2.0 license

**Why This Matters:** MoE architecture enables specialized experts while maintaining efficiency.

---

### 1.6 ServiceNow/Morgan Stanley Lag-Llama

**HuggingFace ID:** `time-series-foundation-models/Lag-Llama`

**Release Date:** February 2024

**Organizations:** ServiceNow, Morgan Stanley, Université de Montréal, Mila-Quebec, McGill University

**Architecture Type:** Decoder-only Transformer (inspired by LLaMA architecture)

**Key Specifications:**
- **Architecture Components:**
  - RMSNorm (pre-layer normalization)
  - Rotary Positional Encoding (RoPE) in attention
  - Distribution head for probabilistic predictions
- **Tokenization:** Lagged features (e.g., lags at {1, 4, 7, ..., L})
- **Covariates:** Temporal features (second, minute, month, etc.)

**Pre-training Data:**
- 27 datasets across 6 domains
- Domains: Energy, transportation, economics, nature, air quality, cloud operations
- ~8K univariate time series
- 352M tokens

**Input Format:** Univariate time series with lagged dependencies

**Zero-shot:** Yes - comparable zero-shot performance

**Use Cases:**
- Probabilistic time series forecasting
- Long-range forecasting
- Uncertainty quantification
- Modeling lagged dependencies (temporal correlations)

**Unique Features:**
- **Focus on lagged dependencies** (important for economics, finance)
- LLaMA-inspired architecture adapted to time series
- Probabilistic predictions with uncertainty intervals
- Immediate uncertainty quantification (no ensembles needed)
- Academic-industry collaboration

**Performance:**
- Zero-shot: Average rank 6.714 among baselines (respectable)
- **Fine-tuning: State-of-the-art on 3 datasets**
- Significant performance increase with fine-tuning
- Uncertainty calibration is strong

**Paper:** arXiv:2310.08278, NeurIPS 2023 Workshops

**Availability:** HuggingFace, open source

---

### 1.7 Datadog Toto

**Repository:** GitHub: DataDog/toto

**HuggingFace ID:** Available on HuggingFace

**Release Date:** July 2024

**Organization:** Datadog

**Architecture Type:** Transformer optimized for observability

**Model Size:** 151M parameters (open-weights version)

**Pre-training Data:**
- **Nearly 1 TRILLION data points** (largest among published models!)
- 750 billion anonymous metric data points from Datadog platform
- Time series from LOTSA datasets

**Input Format:** Time series (especially observability/monitoring metrics)

**Zero-shot:** Yes

**Use Cases:**
- **Observability metrics forecasting** (primary)
- IT infrastructure monitoring
- Cloud operations metrics
- DevOps time series
- Application performance monitoring (APM)

**Unique Features:**
- **Largest pre-training dataset** (1 trillion points)
- **First foundation model specifically tuned for observability**
- Domain-specific specialization (IT metrics)
- Open-weights release
- BOOM benchmark introduced for observability

**Performance:**
- Matches or beats state-of-the-art on standard benchmarks
- **Significantly improves observability metric forecasting** (domain-specific)
- Practical impact: Reduces false positives in anomaly detection

**Paper:** arXiv:2407.07874

**Blog:** https://www.datadoghq.com/blog/datadog-time-series-foundation-model/

**Availability:** HuggingFace, GitHub, Apache 2.0 license

**Why This Matters:** First domain-specific (observability) foundation model, trained on real production data at massive scale.

---

## 2. Tsinghua THUML Family (Timer, Sundial)

**Organization:** THUML Lab, Tsinghua University

**Repository:** GitHub: thuml/Large-Time-Series-Model

These models represent cutting-edge academic research with impressive benchmark results.

---

### 2.1 Timer

**HuggingFace ID:** `thuml/timer-base-84m`

**Release Date:** 2024 (ICML 2024)

**Architecture Type:** Generative Pre-trained Transformer (GPT-style, Decoder-only)

**Model Size:** 84M parameters (base model)

**Technical Specifications:**
- **Layers:** 8
- **Model Dimension:** 1,024
- **Feed-forward Dimension:** 2,048
- **Patch Length:** 96
- **Lookback Length:** 2,880 time points
- **Pre-training Task:** Next token prediction (autoregressive)

**Pre-training Data:**
- UTSD (Unified Time Series Dataset)
- 7 domains
- Up to 1 billion time points (UTSD-12G variant)
- Larger versions: 260 billion points

**Input Format:** Arbitrary-length, any-variable time series

**Zero-shot:** Yes

**Use Cases:**
- Zero-shot forecasting
- Task-specific fine-tuning
- Scalable pre-training research

**Unique Features:**
- GPT-style architecture for time series
- Autoregressive generation
- Flexible sequence length
- Focus on generalization across domains

**Paper:** arXiv:2402.02368, ICML 2024

**Availability:** HuggingFace, GitHub, open source

---

### 2.2 Timer-XL

**Release Date:** 2024, Accepted ICLR 2025

**Architecture Type:** Decoder-only Transformer for long-context

**Focus:** Long-context time series forecasting

**Key Insight:** Encoder-only transformers degrade on long contexts; decoder-only maintains performance

**Input Format:** Arbitrary-length, any-variable time series

**Zero-shot:** Yes - pre-trained for zero-shot

**Use Cases:**
- Long-context forecasting (thousands of time points)
- Unified forecasting across different lengths

**Performance:** Competitive on long-context benchmarks

**Paper:** arXiv:2410.04803, ICLR 2025

---

### 2.3 Sundial

**HuggingFace ID:** `thuml/sundial-base-128m`

**Release Date:** 2025 (ICML 2025 Oral, Top 1%)

**Model Size:** 128M parameters (base model)

**Architecture Type:** Generative Transformer with Flow Matching

**Pre-training Data:**
- **TimeBench:** 10^12 = **1 TRILLION time points**
- Largest pre-training dataset alongside Datadog Toto

**Technical Innovation:** **TimeFlow Loss** based on flow-matching
- No discrete tokenization (continuous values)
- Flow-matching for continuous distributions
- No prior distribution specification needed
- Native continuous-valued time series (unlike Chronos)

**Input Format:** Arbitrary-length time series

**Zero-shot:** Yes - point and probabilistic forecasting

**Use Cases:**
- Zero-shot point forecasting
- **Probabilistic forecasting** (multiple probable futures)
- Fast inference applications

**Unique Features:**
- **First "trillion-level" pre-training** for time series
- Flow-matching approach (novel for time series)
- Just-in-time inference (few milliseconds)
- **State-of-the-art probabilistic forecasting**
- **ICML 2025 Oral (Top 1% of submissions)**

**Performance:**
- **1st place MSE/MAE on Time-Series-Library** (zero-shot)
- **1st place MASE on GIFT-Eval**
- **Fastest inference** among foundation models

**Paper:** arXiv:2502.00816, ICML 2025

**Availability:** HuggingFace, GitHub, open source

**Why This Matters:** Trillion-point pre-training + flow-matching = state-of-the-art probabilistic forecasting.

---

## 3. Task-Versatile Models (Beyond Forecasting)

### 3.1 MOMENT (Carnegie Mellon University)

**HuggingFace ID:** `AutonLab/MOMENT-1-small`, `AutonLab/MOMENT-1-base`, `AutonLab/MOMENT-1-large`

**Release Date:** February 2024 (ICML 2024)

**Organization:** Carnegie Mellon University AutonLab

**Architecture Type:** T5-based Transformer (encoder-only, repurposed)

**Model Sizes:**
- Small: 40M parameters
- Base: 125M parameters
- Large: 385M parameters

**Technical Specifications:**
- **Architecture:** T5 encoder-only
- **Patching:** Time series broken into disjoint patches
- **Embedding:** Each patch → D-dimensional embedding (linear projection)
- **Masking:** Learnable [MASK] embedding for masked patches
- **Pre-training Task:** Masked time series prediction

**Pre-training Data:** "Time Series Pile" - large, diverse public time series collection

**Input Format:** Univariate and multivariate time series

**Zero-shot:** Yes

**Use Cases:**
- **Forecasting**
- **Classification**
- **Anomaly detection**
- **Imputation**
- **Multi-task applications**

**Unique Features:**
- **Multiple tasks with single model** (rare for time series)
- Family of models (Small/Base/Large) for different resource constraints
- Masked prediction pre-training (BERT-style)
- Open-source and accessible

**Performance:**
- Strong generalization across tasks
- Good baseline for multiple time series tasks

**Paper:** arXiv:2402.03885, ICML 2024

**Website:** https://moment-timeseries-foundation-model.github.io/

**Availability:** HuggingFace, GitHub, open source

---

### 3.2 UniTS (Unified Time Series)

**Release Date:** March 2024

**Paper:** arXiv:2403.00131

**Architecture:** Unified architecture with prompt learning

**Key Innovation:** **Learned dictionary of prompts** retrieved at inference

**Use Cases:**
- Imputation
- Anomaly detection
- Out-of-domain forecasting
- Classification

**Unique Features:**
- Universal task specification without specialized modules
- Prompt-based adaptation (similar to TEMPO)
- Handles diverse inputs seamlessly
- Few-shot transfer learning

**Performance:** Exceptional few-shot performance across tasks

---

## 4. Cutting-Edge Architectures (2024-2025)

### 4.1 PatchTST (Patch Time Series Transformer)

**Release Date:** ICLR 2023, continued adoption 2024-2025

**HuggingFace:** Integrated into HuggingFace Transformers library

**Repository:** GitHub: yuqinie98/PatchTST

**Architecture:** Transformer with patching

**Key Concept:** "A Time Series is Worth 64 Words"

**How It Works:**
- Segment time series into **patches** (e.g., 64 patches)
- Each patch treated as a token
- Channel-independent approach
- Patch-wise tokenization reduces sequence length dramatically

**Input Format:** Multivariate time series

**Use Cases:**
- Long-term forecasting
- Multivariate time series

**Performance:**
- **Overall champion** in multiple benchmarks
- Superior at modeling intricate temporal dynamics
- Foundational work inspiring many subsequent models

**Paper:** ICLR 2023, arXiv:2211.14730

---

### 4.2 iTransformer (Inverted Transformer)

**Release Date:** ICLR 2024

**Organization:** Academic research

**Repository:** Time-Series-Library (THUML)

**Architecture:** Vanilla Transformer applied to **inverted** input shape

**Key Innovation:**
- Traditional: Transformer across time (tokens = time steps)
- iTransformer: Transformer across variables (tokens = variables)
- Each entire time series becomes one token

**Input Format:** Multivariate time series

**Use Cases:**
- Long-term forecasting
- Multivariate forecasting

**Performance:**
- Competitive with PatchTST
- Slightly below PatchTST in some benchmarks
- Above TSMixer

**Paper:** ICLR 2024

**Why This Matters:** Simple inversion of input shape yields strong results.

---

### 4.3 TSMixer (Google)

**Release Date:** 2023, active use 2024

**Organization:** Google Research

**Architecture:** **MLP-Mixer based (NOT Transformer!)**

**Key Innovation:** Fully-connected layers only, no attention mechanism

**Design:** Lightweight, efficient alternative to transformers

**Performance:**
- Slightly better than iTransformer overall
- **2-3x faster execution** than transformers
- **2-3x lower memory usage**
- State-of-the-art despite simplicity

**Use Cases:**
- Forecasting with efficiency constraints
- Resource-limited environments
- Fast training and inference

**Why This Matters:** Proves transformers aren't always necessary - simple MLPs can excel.

---

### 4.4 TiDE (Time-series Dense Encoder)

**Release Date:** April 2023, updated 2024

**Organization:** Google Research

**Repository:** GitHub: google-research/google-research/tree/master/tide

**Architecture:** **MLP-based Encoder-Decoder (NOT Transformer!)**

**How It Works:**
- **Encoder:** MLP encodes past time points + covariates
- **Decoder:** MLP combines encoding + future covariates
- Residual connections
- No attention mechanism

**Input Format:** Univariate and multivariate with covariates

**Use Cases:**
- Long-term forecasting
- Forecasting with external covariates (holidays, events, etc.)
- Non-linear dependencies

**Unique Features:**
- "Embarrassingly simple" MLP architecture beats transformers
- Handles covariates effectively
- Near-optimal for linear dynamical systems (theoretically grounded)
- Simplicity enables interpretability

**Performance:**
- Matches or outperforms prior transformer approaches
- **5-10x faster** than best transformer models
- **10x faster training** than transformer baselines

**Paper:** arXiv:2304.08424

**Availability:** GitHub, Vertex AI, open source

**Why This Matters:** Another example of MLPs beating transformers with better efficiency.

---

## 5. Vision-Based Time Series Models

### 5.1 ViTime

**Release Date:** July 2024

**Repository:** GitHub: IkeYang/ViTime

**Architecture:** Vision intelligence framework for time series

**Key Innovation:** Numerical time series → **binary images**

**How It Works:**
1. Transform numerical temporal correlations → pixel spatial patterns
2. Binary image representation
3. Synthesize authentic periodic and trend patterns
4. Forecast in binary image space

**Input Format:** Univariate time series

**Zero-shot:** Yes

**Use Cases:**
- Time series forecasting
- Handling missing data (no imputation needed)
- Robustness to data perturbations

**Unique Features:**
- Vision intelligence framework applied to time series
- Binary image representation (novel)
- Synthetic data synthesis for pattern diversity
- **Exceptional robustness to perturbations**

**Performance:**
- **Zero-shot: Outperforms TimesFM by 9-15%**
- **10% fine-tuning:** Surpasses fully-supervised benchmarks
- **20-30% better than TimesFM under data perturbations**

**Paper:** arXiv:2407.07311

**Availability:** GitHub, open source

**Why This Matters:** Crossover from computer vision to time series, better robustness.

---

## 6. Foundation Model Comparison

| Model | Org | Release | Params | Context | Data | Architecture | Specialty |
|-------|-----|---------|--------|---------|------|--------------|-----------|
| **TimesFM 2.5** | Google | 2024-09 | 200M | 16,384 | 100B | Decoder | Long context leader |
| **TTM** | IBM | 2024 | <1M | Flex | 1B | TSMixer | Tiny, efficient |
| **TSPulse** | IBM | 2024 | 1M | Flex | Large | Dual-space | Anomaly detection |
| **FlowState** | IBM | 2024 | N/A | Flex | Large | SSM | Timescale flexibility |
| **Chronos-Bolt** | Amazon | 2024-11 | 9-205M | 512 | 100B obs | T5 | 250x faster |
| **TimeGPT** | Nixtla | 2023-24 | N/A | N/A | 100B | Custom | First (commercial) |
| **Moirai-MoE** | Salesforce | 2024-10 | 11-86M* | Flex | 27B | MoE (32) | 65x fewer params* |
| **Lag-Llama** | ServiceNow+ | 2024-02 | LLaMA | Flex | 352M | Decoder | Lagged dependencies |
| **MOMENT** | CMU | 2024-02 | 40-385M | Flex | Pile | T5 | Multi-task |
| **Toto** | Datadog | 2024-07 | 151M | Flex | 1T | Transformer | Observability |
| **Timer** | THUML | 2024 | 84M | 2,880 | 260B | Decoder | GPT-style |
| **Sundial** | THUML | 2025 | 128M | Flex | 1T | Flow | ICML Oral, 1st place |
| **ViTime** | Academic | 2024-07 | N/A | Flex | Synth | Vision | 9-15% > TimesFM |

**Active params for MoE*

---

## Key Trends: Time Series (2024-2025)

1. **Scale explosion:** 1 trillion time point pre-training (Toto, Sundial)
2. **Tiny models work:** <1M parameters competitive with billion-param models
3. **Context growth:** TimesFM 2.5 reaches 16K context
4. **MLP renaissance:** TSMixer, TiDE prove transformers aren't always best
5. **Domain specialization:** Toto for observability shows value of domain-specific models
6. **Zero-shot standard:** All major 2024 models support zero-shot
7. **Vision crossover:** ViTime shows computer vision techniques work for time series
8. **MoE emerging:** Moirai-MoE first sparse MoE for time series

---

## Recommendations by Use Case

**Need best zero-shot accuracy?**
→ Sundial (1st on benchmarks), TimesFM 2.5 (Google Cloud)

**Resource-constrained / CPU-only?**
→ TTM (<1M params), TSPulse (1M params)

**Need speed?**
→ Chronos-Bolt (250x faster), TTM (2-3x faster than transformers)

**Observability/IT metrics?**
→ Toto (domain-specific, 1T training points)

**Multi-task (forecast + anomaly + classification)?**
→ MOMENT, TSPulse

**Probabilistic forecasting?**
→ Sundial (flow-matching), Chronos-Bolt, Lag-Llama

**Commercial deployment?**
→ TimeGPT API (Nixtla), TimesFM (BigQuery)

**Long context (16K+)?**
→ TimesFM 2.5

---

*End of Part 3: Time Series Foundation Models*

**Continue to Part 4: Exotic and Geometric Embeddings...**
