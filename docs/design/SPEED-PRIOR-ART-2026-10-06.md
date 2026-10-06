# Embedding speed: prior art and what Tessera should copy (2026-10-06)

Box R1 (Daisy, 2026-10-06 06:25Z). Research only. Nothing was built, run, installed or downloaded on
Tom's Mac. Tessera source is unchanged. Where a source was read, the commit or date is given.
Anything marked **NOT CONFIRMED** is my reading or estimate, not a measured or quoted fact.

Starting point: Tessera main `1d600f4` uses Candle 0.11 on the CPU in f32 with Apple Accelerate. It
runs BERT-family dense models plus ColBERT, SPLADE and miniCOIL. bge-base gives about 50 short texts/s
on a quiet M-series Mac, and 3.4 texts/s at 512 tokens.

## 0. A ceiling check on our own number (estimate, NOT CONFIRMED)

bge-base has about 110M parameters. One 512-token text costs roughly 2 × 110M × 512 ≈ 113 GFLOP in
the dense layers, plus about 10 GFLOP in attention (4 × 512² × 768 × 12 layers), so about 123 GFLOP in
total. At 3.4 texts/s that is about 0.4 TFLOP/s of f32. The f32 matrix throughput of Accelerate (the
AMX units) on M-series is quoted informally at 1 to 2 TFLOP/s per cluster, but I found no primary
source. If that is right, the 512-token path already runs within a small factor of what the CPU can
do in f32. Large gains on long text then have to come from:

- less precision (f16 or int8);
- another engine (GPU or ANE);
- or less work (padding removed, repeats cached).

They will not come from tuning the f32 CPU path. The short-text figure (50/s) is far below what the
FLOPs allow, so short texts are dominated by overhead (per-call cost, padding, small batches). Trixie's
hot-path map should confirm or refute this.

This same arithmetic is used below to reject one published figure (§2.13).

## 1. EmbedAnything, read at `74a68f2ed00071ef9c88449101aec4c414a9c38d` (2026-09-25)

Repository: github.com/StarlightSearch/EmbedAnything (shallow clone, read in place).

**What "vector streaming" is in code.** It lives in `rust/src/lib.rs`, `embed_directory_stream` (line 815):

- `:836` and `:837`: two `tokio::sync::mpsc::unbounded_channel()`s, one carrying chunks and one
  carrying results.
- `:846`: a `tokio::spawn` consumer collects chunks into a buffer of `buffer_size` (default 100,
  `rust/src/config.rs:70`). At `:857` (`if chunk_buffer.len() == buffer_size`) it calls
  `process_chunks` (`:1175`), which calls `embedder.embed(&chunk_refs, batch_size, late_chunking)`.
- `:881` and `:917`: the results are sent on the collector channel.
- `:927` to `:946`: the producer runs file extraction and chunking synchronously in the caller
  (`files.into_iter().for_each`, then `tx.send((chunk, metadata))`).
- `:952`: `drop(tx)`.
- `:956`: the caller drains `collector_rx.recv()` and hands each buffer's embeddings to the
  vector-database adapter.

**Pipeline and memory, not compute.** Vector streaming gives two things:

- Pipeline: parsing and chunking overlap with embedding.
- Memory: embeddings go to the adapter one buffer at a time, instead of the whole corpus being held.

The model forward is unchanged. Both channels are **unbounded**. The README and blog say "only the
chunks and embeddings in the buffer are stored in the system memory", but nothing in the code bounds
the input side. A fast producer can queue every chunk ahead of a slow model (my reading of the code;
not run).

**Backends.**

- Candle (`rust/src/embeddings/local/bert.rs`, `jina.rs`, `modernbert.rs`), with Cargo features
  `accelerate`, `mkl`, `metal`, `cuda`, and `flash-attn` (CUDA only).
- ONNX Runtime (`ort = 2.0.0-rc.10` with `half`), in `rust/src/embeddings/local/ort_bert.rs`:
  - execution providers `[CUDAExecutionProvider, CoreMLExecutionProvider]` with default CoreML
    options (`:146` to `:150`);
  - `GraphOptimizationLevel::Level3`, intra-op threads = logical cores / 2, inter-op threads 1
    (`:151` to `:153`);
  - the session sits behind an `RwLock` and takes `write()` for each embed, so inference is
    serialised.

**Batching.**

- Fixed-count `.chunks(batch_size)` with default 32 (`ort_bert.rs:168`, `:182`; `bert.rs` the same).
- `PaddingStrategy::BatchLongest` (`ort_bert.rs:116`).
- No sorting by length and no token budget was found.

**Long text.**

- Text is chunked before embedding (default `chunk_size` 1000).
- Optional late chunking (`ort_bert.rs:240`; also `bert.rs:147`, `jina.rs:141`, `modernbert.rs:145`,
  `ort_jina.rs:173`):
  1. concatenate the token ids of a mini-batch's chunks into one sequence;
  2. run one forward;
  3. split by cumulative lengths before pooling.

**Published throughput.** The README benchmark is an image with the caption "Only measures embedding
model inference speed, on onnx-runtime". It shows EmbedAnything 0.44 s, FastEmbed 0.49 s and
SentenceTransformers 0.63 s. It names no model, input size or hardware; it links a Colab, which I did
not run. The vector-streaming blog post
(embed-anything.com/blog/2024/03/31/vector-streaming/) gives no figures. **There is no published
EmbedAnything throughput with hardware.**

**What to copy.**

- The pipeline idea (overlap extraction and tokenisation with the forward pass), but with *bounded*
  channels.
- Not the batching: it is plainer than ours would need to be.

## 2. Techniques

Each entry gives the source read, what it gains, on what hardware, and its cost to correctness.

### 2.1 ONNX Runtime with the CoreML execution provider (Apple Neural Engine)

- **Sources.**
  - ORT CoreML EP docs (onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html, read
    2026-10-06).
  - Apple, "Deploying Transformers on the Apple Neural Engine"
    (machinelearning.apple.com/research/neural-engine-transformers).
  - agmem issue #139 (github.com/AlfoldiMate/agmem/issues/139).
- **What the docs say.**
  - `ModelFormat` is MLProgram (macOS 12+) or NeuralNetwork.
  - `MLComputeUnits` is CPUOnly, CPUAndNeuralEngine, CPUAndGPU or ALL.
  - `RequireStaticInputShapes`: "performance may be negatively impacted by inputs with dynamic
    shapes".
  - Without `ModelCacheDirectory`, compiling "may cost significant time (even minutes)".
  - Gelu is supported only under MLProgram.
- **Gain.**
  - Apple: distilbert at sequence 128, batch 1, on an iPhone 13 (A15) was "up to 10 times faster"
    with "14 times less memory" (3.47 ms at 0.454 W). This needed a *rewritten* model: (B,C,1,S)
    layout, Conv2d in place of Linear, attention split per head, and no reshapes or transposes.
  - agmem #139 cites ane_transformers giving "MiniLM 0.53 ms on M5 Pro via a hand-converted Core ML
    model". It also states that "no one has published a BERT/Qwen3 encoder through ORT's CoreML EP".
    I found none either.
- **Cost to correctness.**
  - The ANE computes in f16, so outputs will not match f32 certified references bit for bit.
  - Static shapes force padding to fixed buckets.
  - First load can take minutes without a cache, which hurts start-up.
- **For Tessera.** A second runtime (ort) and per-model conversion. The ANE gain needs Apple's
  model rewrite, not just the EP.

### 2.2 MLX and mlx-embeddings

- **Sources.**
  - mlx-embeddings at `9b28270be81211f2b8daed0041aec65ea5dc4b28` (2026-05-13), read in place.
  - LucasSte/MLX-vs-Pytorch (search result).
  - contracollective.com (§2.13).
- **Code.**
  - `models/bert.py:89` uses plain `mx.matmul` attention.
  - `models/modernbert.py:208` uses the fused `mx.fast.scaled_dot_product_attention`.
  - `convert.py` offers `quantize_model` with a group size.
  - Inputs are padded to `max_length`.
  - The README gives no throughput figures.
- **Gain.**
  - LucasSte gives BERT *training* times: M1 Max 793.67 s with PyTorch MPS against 499.34 s with MLX;
    M3 Max 550.29 s against 408.45 s. That is MLX against MPS, not against our CPU path, and it is
    training, not inference.
  - No primary MLX BERT inference figure on M-series was found.
- **Cost to correctness.** In f32 on the GPU, results differ from CPU at the rounding level. f16 or
  quantised output differs more.
- **For Tessera.** Large: a C++/Python runtime, or Rust bindings (not evaluated).

### 2.3 Candle Metal

- **Sources.**
  - candle #3302: "Candle's Metal backend GEMM is 2-11x slower than PyTorch MPS and MLX", opened
    2026-01-14, closed by PR #3313.
  - candle #4021: Metal SDPA head dimensions.
  - candle #1780: slow Metal-to-CPU `to_device` (MiniLM, M1 Pro; search result, not opened).
- **Figures from #3302 (M-series, chip not named).**

  | Case | Shape | PyTorch MPS | MLX | Candle |
  |---|---|---|---|---|
  | 4D attention | (484,6,144,32)×(484,6,32,144) | 2.46 ms | 2.20 ms | 24.65 ms |
  | Linear | (17424,768)×(768,3072) | 12.56 ms | 12.97 ms | 31.94 ms |

  The cause was a hard-coded tile of (32,32,16,2,2). The fix adds shape-chosen tiles.
- **Gain.**
  - No Candle Metal against Candle CPU figure for BERT embedding was found.
  - Candle 0.11 (ours) contains PR #3313 (§5a).
- **Cost to correctness.** In f32 on the GPU, results differ from Accelerate at the rounding level, so
  they must stay inside each reference's tolerance (§3).
- **For Tessera.** Medium. The feature exists and the models are already Candle. Work needed:
  - device plumbing;
  - copying back from GPU to CPU;
  - certifying a second device.

### 2.4 f16 and bf16

- **Source.** sentence-transformers `docs/sentence_transformer/usage/efficiency.rst` at repo HEAD
  `4a3b5cd6ec718e421f57e824a41ed3fd99595df6`, and its charts. Hardware: i7-13700K CPU and RTX 3090
  GPU. Models: MiniLM-L6, bge-base, mxbai-large and bge-m3.
- **Gain.**

  | Setting | Speed against fp32 | Quality kept |
  |---|---|---|
  | CPU torch-fp16 | 0.25x | — |
  | CPU torch-bf16 | 0.28x | 99.87% |
  | CPU llama.cpp f16 | 0.89x | 99.98% |
  | GPU fp16 | 2.92x | — |

  On the CPU, half precision was slower. On the GPU it was faster.
- **Apple.** No f16 BERT figure on M-series GPU or CPU was found. Apple GPUs run f16 at about twice
  the f32 rate (general knowledge, **NOT CONFIRMED** from a primary source here).
- **Cost to correctness.** Outputs change. ST reports "without reducing average task quality" on its
  GPU runs, but vectors are not equal to f32 references.

### 2.5 int8 quantisation

- **Source.** ST efficiency.rst, as above.
- **Gain (i7-13700K CPU).**

  | Setting | Speed | Quality kept |
  |---|---|---|
  | onnx-qint8 | 1.92x | 99.58% |
  | openvino-qint8 | 2.84x | 99.37% |
  | llama.cpp q8_0 | 1.01x | 99.92% |
  | llama.cpp q4_k_m | 1.23x | 99.40% |

  ST's own conclusion: "Always measure on your own model, hardware, and inputs."
- **Apple.** No int8 BERT figure on M-series was found. OpenVINO is x86-oriented. Candle has quantised
  (GGUF) matmul for its LLMs. **NOT CONFIRMED** whether it applies cleanly to Candle's BERT, or is fast
  on Accelerate or Metal.
- **Cost to correctness.** About 0.4 to 0.6% task-quality loss. Vectors differ, so a quantised model is
  a different certified model.

### 2.6 Sorting by length and token-budget batching

- **Sources.**
  - TEI (huggingface/text-embeddings-inference) at `98b7ea2ddb928eccbfde41d96e9576f876d045f4`
    (2026-09-23), read in place.
  - ST efficiency.rst.
- **TEI's code.**
  - `core/src/queue.rs:148` to `:160` packs requests in arrival order until `max_batch_tokens` is
    reached (default 16384).
  - For padded models, the budget charges `max_length × count` (`:150`), so one long text cannot be
    paired with many short ones beyond the budget.
  - TEI does not sort.
- **ST.** `encode` sorts its inputs by length before batching and restores the order afterwards (known
  ST behaviour; **NOT CONFIRMED** at the commit read, since I read efficiency.rst, not
  `SentenceTransformer.py`). Token-budget mini-batches (`mini_batch_num_tokens`) are described for
  training.
- **Gain.** No figure on its own. The saving equals the share of padding removed, which is large when a
  batch mixes 10 and 500 tokens.
- **Cost to correctness.** None in exact arithmetic, because padding is masked. In floating point,
  batch shape can change the last bits. Tessera's own test
  `mixed_length_bert_batch_matches_sequential_forward` (`src/encoding/dense/tests.rs:246`) already
  allows 1e-5 between a padded batch item and the same text alone. That is far inside the reference
  tolerance (§3).

### 2.7 Padding-free (variable-length) attention

- **Sources.**
  - TEI `backends/candle/src/models/flash_bert.rs`: uses `flash_attn_varlen` with `cu_seqlens`
    (`:95` to `:100`). It is used only on CUDA, in f16, compute capability ≥ 8.0
    (`backends/candle/src/lib.rs:45` to `:63`, `:380`).
  - TEI `backends/candle/src/models/bert.rs:706` to `:790`: on CPU and Metal, TEI builds a **padded**
    batch with an additive mask.
  - ST efficiency.rst: on the GPU, "fp16+FA2+unpadding" gives 3.87x and bf16+FA2 3.84x against fp32
    (RTX 3090), with the largest gain on mixed lengths.
  - ModernBERT paper (arXiv 2412.13663): unpadding gives 10 to 20% over other unpadding methods.
- **Apple.** No varlen attention kernel for Metal or CPU was found in Candle or TEI.
- **The cheap variant for us** (design idea, **NOT CONFIRMED** by any source):
  1. Pack all tokens of a batch into one `[T, H]` tensor for embeddings, every Linear, LayerNorm and
     FFN. These work row by row and never see padding.
  2. Run only the attention per sequence (or per group of equal-length sequences).
  This removes padding FLOPs from about 95% of the compute without a new kernel.
- **Cost to correctness.** Same as §2.6.

### 2.8 Flash attention on Metal

- **Sources.**
  - candle #4021 (search result): Candle's Metal SDPA kernel supports head dimensions 32, 64, 72, 80,
    96, 128, 256 and 512, not 48. Unsupported shapes fall back to matmul and softmax.
  - mlx-embeddings `modernbert.py:208` uses MLX's fused SDPA.
- **Coverage.** bge-base has head dimension 64 and bge-small 32, so both are covered. A 384-wide
  model with 8 heads (48) is not.
- **Gain.** No figure found. At 512 tokens, attention is about 8% of bge-base FLOPs (§0), so the fused
  kernel mostly saves memory traffic. It is not a large compute win at our lengths.
- **Cost to correctness.** It sums in a different order, so the last bits change.

### 2.9 Overlapping tokenisation with the forward pass

- **Sources.**
  - TEI `router/src/lib.rs:237`: tokenisation workers default to `(num_cpus − 1).clamp(1, 64)`.
  - The TEI README: "control the number of tokenizer workers used for payload tokenization,
    validation and truncation".
  - EmbedAnything §1 (producer and consumer).
- **Gain.** At most the tokeniser's share of wall time. Unknown for us until Trixie's map.
- **Cost to correctness.** None, if the order of the vectors is kept.

### 2.10 Caching by content hash

- **Source.** LangChain `CacheBackedEmbeddings`
  (`libs/langchain/langchain_classic/embeddings/cache.py`, master, read 2026-10-06):
  - the key is `namespace + hash(text)`;
  - the default hash is SHA-1, with a warning that it is "*not* collision-resistant", and SHA-256 or
    BLAKE2b available;
  - queries are not cached by default;
  - the namespace is meant to separate models.
- **Gain.** Unbounded on repeats (re-running a backfill, unchanged documents), zero on new text. No
  figure, because it depends on the workload.
- **Cost to correctness.** None, if the key covers everything that changes the vector:
  - model id, pinned revision and manifest digest (`ModelIdentity`);
  - role and prompt;
  - the limit settings;
  - the exact text bytes, hashed with SHA-256.
  LangChain's namespace-only-by-convention design is the trap to avoid.

### 2.11 Static embedding models (model2vec, Static Embeddings)

- **Source.** Hugging Face blog "Train 400x faster Static Embedding Models", 2025-01-15. Hardware:
  i7-13700K and RTX 3090.
- **How they work.** "the Encoder step is as simple as a dictionary lookup". In practice this is an
  `EmbeddingBag` (token lookup plus mean pooling), with no attention.
- **Gain.**
  - static-retrieval-mrl-en-v1 runs 107,419 sentences/s on CPU against 270/s for all-mpnet-base-v2
    (397x).
  - It reaches 87.4% of mpnet on NanoBEIR (NDCG@10 0.5032 against 0.5757).
  - The multilingual model is about 125x faster on CPU than multilingual-e5-small.
- **Cost to correctness.**
  - It is a *different model* with lower quality, not a speed-up of bge.
  - It would be a new registry entry with its own certification.
  - It is useful as a fast first-pass or fallback model.

### 2.12 Late chunking for long texts

- **Source.** Günther et al., arXiv 2409.04701 (v3, 2025-07-07); EmbedAnything code §1.
- **What it does.**
  1. Embed all tokens of a long text once, with a long-context model.
  2. Split into chunks after the transformer and before pooling.
- **Gain.** The abstract claims quality ("superior results") and no speed figure. For speed, it makes
  one forward over N tokens in place of k forwards over N/k tokens. Attention cost grows with length,
  so it is not faster per token. It needs a long-context model (for us, jina v2 or nomic at 8192).
- **Cost to correctness.** Chunk vectors differ from separately embedded chunks by design. It would be
  a new, opt-in output, not a replacement.

### 2.13 Newer methods (2025 to 2026) and figures rejected

- **ModernBERT-style unpadding and sequence packing.** These are GPU-kernel designs; see §2.7.
  gte-modernbert is already in Tessera. One search summary claimed the gains "vanish" on CPU (source
  not opened, **NOT CONFIRMED**).
- **Matryoshka (MRL) truncation.** Smaller vectors make storage and search cheaper, not embedding.
  Out of scope for speed.
- **llama.cpp GGUF embeddings.**
  - ST's efficiency doc says llama.cpp leads on cloud CPUs.
  - On the i7, llama.cpp gives f16 0.89x and q8_0 1.01x (§2.4, §2.5).
  - nullmirror (2026-02-28 blog) measured bge-m3 through Ollama: MacBook Air (Mac14,2) 0.92 req/s,
    p50 300 ms; Mac mini (Mac16,11) 5.86 req/s, p50 159 ms. Batch and lengths were not stated.
- **Rejected figure: contracollective.com, "Local embeddings on Apple Silicon … M5 Max 2026".** It
  claims MLX 0.21 Q8 bge-m3 at 4,800 passages/s (batch 64, 512 tokens) and nomic-embed-text-v2 at
  19,800/s, on an M5 Max.
  - bge-m3 is about 568M parameters. 2 × 568M × 512 ≈ 0.58 TFLOP per passage, so 4,800/s would be
    about 2.8 PFLOP/s. That is two orders of magnitude beyond any Apple chip.
  - "MLX 0.21" is also a 2024 version number.
  - **Treated as unreliable; not used in the table.**
- **Fused fp32 attention for Candle** (github.com/mi-for-the-rust-of-us/candle-fused-attn). CUDA only.
  Not applicable.

## 3. Ranked for Tessera on Apple Silicon

Ranking is by expected gain for our journeys, weighed against size and risk to the certified
references. Certification compares within a tolerance, not by hash (Daisy's ruling, checked in code):

- `xtask/src/certification/reference_compare.rs:21-32` checks each value against
  `absolute + relative × |expected|` and a minimum cosine, both read from the reference document
  (`:91-106`).
- `xtask/src/certification/reference.rs:263-266` compares the hash of the *expected* output only.
  For the *observed* output it checks only that the hash is valid hex. No observed hash is ever
  compared.
- All 68 tolerance blocks under `certification/references/` have `absolute` 0.001 and
  `minimum_cosine` 0.999. `relative` is 0.01 in 64 of them, and 0.001 in the 4 contract references
  (`certification/references/contract/{dense,sparse,multi-vector,vision}.json`).
- **No reference requires an exact hash.**

So rounding-level changes (batch shape, Metal f32, fused attention) are low risk while they stay
inside those bounds. f16, int8, ANE and MLX quantisation change vectors by more than rounding and
must be measured against the bounds before anyone calls them low.

| # | Technique | Expected gain on M-series (source of figure) | Size | Risk to certified references | Journeys helped |
|---|---|---|---|---|---|
| 1 | Content-hash cache (SHA-256 of text + `ModelIdentity` + role/prompt + limits) | Unbounded on repeats, 0 on new text. No figure found (workload-dependent). | Small | None if the key is complete | Backfill throughput (re-runs), one-query latency (repeat queries) |
| 2 | Sort by length within a call, then token-budget batches (restore order) | Removes the padding share. No M-series figure found; ST GPU unpadding up to 3.87x total with fp16 (RTX 3090) | Small | Low: batch-shape rounding, already bounded at 1e-5 by an existing test, against a tolerance of 1e-3 | Backfill throughput, long text (memory bound per batch) |
| 3 | Overlap tokenisation with the forward pass (bounded channel, worker pool) | At most the tokeniser's share of wall time. No figure found; size it from Trixie's map | Small to medium | None | Backfill throughput |
| 4 | Packed `[T,H]` linears with per-sequence attention (padding-free without a kernel) | Same saving as #2 but within mixed batches. No figure found | Medium | Low (same as #2) | Backfill throughput, long text |
| 5 | Candle Metal, f32 | No BERT figure found. GEMM was 2-11x behind MPS/MLX before PR #3313 (candle #3302); 0.11 contains the fix (§5a) | Medium | Low: f32 GPU rounding, inside tolerance (to be measured) | Backfill throughput, long text |
| 6 | Fused SDPA on Metal (with #5) | No figure found. Attention ≈ 8% of FLOPs at 512 (§0 estimate) | Small once #5 exists | Low beyond #5 | Long text |
| 7 | f16 on Metal (with #5) | No M-series figure found. RTX 3090 2.92x; CPU f16 0.25x (ST) | Small once #5 exists | High: new vectors, re-certify | Backfill throughput, long text |
| 8 | Static embedding model (model2vec / static-retrieval-mrl) as a new registry entry | 397x CPU against mpnet at 87.4% NanoBEIR (HF blog, i7-13700K); no M-series figure | Small to medium | None to existing models (new certified model) | One-query latency, backfill (as a fast tier) |
| 9 | int8 (quantised weights) | No M-series figure. onnx-qint8 1.92x, openvino-qint8 2.84x on i7-13700K (ST) | Large | High: new vectors, about 0.5% quality | Backfill throughput |
| 10 | ORT + CoreML EP / ANE | No ORT BERT figure on M-series exists (agmem #139). Apple: distilbert up to 10x on A15 after a model rewrite | Large | High: f16, static shapes, new runtime | One-query latency (if warm). Hurts start-up (compile, minutes uncached) |
| 11 | MLX (mlx-embeddings) | No reliable inference figure found (contracollective rejected, §2.13) | Large | Medium to high | Backfill throughput |
| 12 | Late chunking (opt-in, long-context models) | Quality, not speed. No speed figure | Medium | None to existing outputs (new output kind) | Long text (quality) |
| 13 | EmbedAnything-style streaming to the caller | Pipeline and memory, not compute. No figure published | Small (mostly the caller's side) | None | Backfill throughput (memory) |

**Start-up.** Nothing found speeds Tessera's start-up except avoiding the ANE compile (#10). TEI loads
weights with `from_mmaped_safetensors` (`backends/candle/src/lib.rs:278`). Tessera already does the
same: `src/encoding/dense/loading.rs:126` and `src/backends/candle/encoder.rs:130` at `1d600f4`
(grep only).

**What to copy first.** Items 1 to 4 keep the f32 CPU references and attack the short-text gap and
the padding waste. Item 5 is the only route seen to more raw compute that keeps the model files we
already have. It needs a decision on how certification treats a second device.

## 4. Not confirmed, and where I looked

- **§0 ceiling.** Partly answered in §5b. There is no published M1 Max figure; M1 and M1 Pro figures
  are given there.
- **Whether Tessera sorts by length.** A grep of `src/` at `1d600f4` found no length sort (the only
  sort is `src/encoding/minicoil/vector.rs:288`, by index). The call path was not traced; that is
  Trixie's hot-path map.
- **Batch shape under Accelerate.** Bounded by the existing 1e-5 test only. Not run here.
- **ST sorts by length in `encode`.** Known behaviour, not re-read at the commit.
- **Candle #4021 head dimensions.** From the search-result summary; the issue page was not opened.
- **Candle #1780** (Metal to CPU copies slow) and **#3052** (Candle against PyTorch). Seen only in
  search results, not opened.
- **contracollective.com figures.** Read, then rejected by arithmetic (§2.13).
- **"TDS: How Fast Is MLX" (8 Apple chips).** Seen in search results, not opened.
- **EmbedAnything's Colab benchmark.** Not run, and no hardware stated.
- **Candle #2877** (8.5x slower than PyTorch on CPU). Windows i9-13900HX with MKL, Candle 0.8.4. Not
  Apple; open with no maintainer answer. Noted, not used.
- **Paywalls and credentials.** No source needed them. Nothing was skipped for payment.

## 5. Follow-up R2 (Daisy, 2026-10-06 06:34Z)

Read-only. Nothing was built or run.

### 5a. Does our Candle 0.11 contain PR #3313? **Yes.**

- **What we build against.** `Cargo.lock` pins `candle-core` 0.11.0 (`:330`) and
  `candle-metal-kernels` 0.11.0 (`:372`) from crates.io. The vendored crate's `.cargo_vcs_info.json`
  names git sha `31f35b147389700ed2a178ee66a91c3cc25cc80d`. That is the commit the GitHub tag API lists
  for tag `0.11.0`.
- **The PR.** huggingface/candle #3313, "Metal GEMM Dynamic Tile Selection and Batch Collapse
  Optimization", merged 2026-01-21 as `06cb7134370e1e7790960b061c98d0be7616bf96`. It touches:
  - `candle-metal-kernels/src/kernels/mlx_gemm.rs`
  - `candle-metal-kernels/src/metal_src/mlx_gemm.metal`
  - `candle-metal-kernels/src/metal/device.rs`
  - a matmul bench
- **Ancestry.** GitHub's compare of `06cb713...31f35b1` reports `ahead`, ahead by 101 and behind by 0.
  The PR's merge commit is therefore an ancestor of the 0.11.0 release.
- **The PR's code in the vendored source**
  (`~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/candle-metal-kernels-0.11.0/src/kernels/mlx_gemm.rs`):

  | What the PR adds | Line |
  |---|---|
  | `struct TileConfig` | `:23` |
  | `TILE_64_64_16_2_2` | `:42` |
  | `fn select_tile_config` | `:59` |
  | `fn check_batch_collapse` | `:194` |
  | `fn should_use_split_k` | `:248` |
  | call site `check_batch_collapse(b, m, k, a_trans, lhs_stride, rhs_stride)` | `:540` |
  | call site `let tile = select_tile_config(dtype, m, n, k, b, a_trans, b_trans, device_type);` | `:549` |

  The names and call sites are the PR diff's added lines.
- **Caveat.** Tessera's `metal` feature (`Cargo.toml:56`) is not in the default build we certify, so
  none of this runs today.

### 5b. Accelerate (AMX) f32 GEMM peak on M1 Max: **NOT CONFIRMED.** No published M1 Max figure was found.

Closest published figures:

- **danieldk/gemm-benchmark README** at `b4f6ed3fcb09bf3c88ca6f3ceda7ddcc050fdd97` (2024-05-18).
  Accelerate sgemm, matrix size 768, 1000 iterations, in GFLOPS:

  | Chip | 1 thread | 2 threads | 4 threads | Best |
  |---|---|---|---|---|
  | M1 | 1340 | — | — | 1340 at 1 thread |
  | M1 Pro | 2061 | 2583 | 2685 | 2685 at 4 threads |
  | M1 Ultra | — | — | — | 4376 at 16 threads |
  | M2 | — | — | — | 1730 at 4 threads |

  There is no M1 Max row.
- **Bhan (Georgia Tech), arXiv 2606.25426v1** (2026-06-24): plain M1, macOS 26.5.1.
  - One AMX block per cluster (P and E).
  - About 1,525 GFLOPS f32 load-free on one thread.
  - 610 to 680 GFLOPS with operand loads interleaved.
  - About 1,480 GFLOPS aggregate at eight threads.
- **jott.live "1.5 TFLOPs on a single M1 core"** (via HN item 34259213): M1. An HN comment says M1 Pro
  and M1 Max have two P-clusters, each with an AMX unit.
- **Not M1.** scalable.uni-jena.de/opt/sme/gemm.html measures the M4 with SME: Accelerate at 1825.1
  GFLOPS for M=N=K=512.

**Inference, NOT CONFIRMED.** The M1 Max has the same two P-clusters as the M1 Pro, so roughly
2.0 to 2.7 TFLOP/s f32 in Accelerate at 768-square matrices.

**Effect on §0.** bge-base at 3.4 texts/s × ~123 GFLOP ≈ 0.42 TFLOP/s is about 15 to 20% of that, at
our real (thinner, smaller-M) shapes. The 512-token path has headroom of perhaps 2 to 4x before the
f32 CPU ceiling, not the "small factor" §0 guessed. Where the loss sits (GEMM shapes, non-GEMM ops,
threading) is for Trixie's hot-path measurement.

### 5c. Length sort plus token budget, for both batch paths

Two invariants hold for both paths:

- The caller's order of outputs never changes.
- Every index reported to the caller (`EmbedFailure::OutputInvalid { index }`, the worker's per-item
  outcomes) names the caller's position, never the sorted position.

Never-cut still holds: an item that alone exceeds the budget runs as a group of one, or is refused by
the existing preflight. It is never truncated.

#### Path A: worker per-item outcomes

The path is `crates/tessera-worker/src/engine.rs:126-142` → `encode_batch_outcomes`
(`src/encoding/dense/inference.rs:205-240`).

**How it works today.**

1. Tokenise once (`:212`).
2. Preflight activations at `inputs.len() × longest` (`:219`).
3. Run one forward **per text at batch 1** on the bounded Rayon pool
   (`into_par_iter().enumerate()`, `:231`).

There is no padding, so sorting saves no padded FLOPs. It does two other things:

- **Load balance.** Rayon splits an indexed iterator by index ranges. A cluster of long texts in one
  range leaves a long tail.
- **Grouping.** Batch-1 forwards on short texts give thin GEMMs (M = token count, perhaps 10 to 30
  rows), which use AMX poorly. This is the likely cause of the 50 short texts/s; it is a
  **hypothesis for Trixie's measurement**.

**Design.**

1. **Sort.** Straight after `encode_batch_with_prompt` (`:212`), build
   `order: Vec<usize> = (0..n)` sorted *stably* by `token_ids.len()`, descending. A stable sort keeps
   ties in the caller's order, so the run is deterministic.
2. **Group.** Walk `order` and cut groups so that `group_len × group_max_tokens` stays within the
   activation budget. Each group is checked with the existing
   `resource_policy.validate_transformer_activations(profile, group_len, group_max, dtype)`. This uses
   no new knob. It replaces the one call at `:219`, which charges every item at the longest length.
   Charging per group is TEI's padded-model rule (`core/src/queue.rs:150`).
3. **Run.**
   - A group of one keeps today's batch-1 forward.
   - A group of more than one runs one padded forward through the shared helper of Path B below.
   - Groups go on the Rayon pool longest-first, with `with_max_len(1)`, under the same single
     inference permit (`:228`).
4. **Restore.** Every forward carries its original index.
   - `counted_embedding(original_index, …)` (`:259`) is called with the *original* index, so
     `OutputInvalid` names the caller's item.
   - Results are written into `let mut out: Vec<Option<_>> = vec![None; n]` at `out[original_index]`.
   - Then `out.into_iter().map(Option::unwrap)`. Every slot is filled by construction; a debug
     assertion checks it.
   - `engine.rs:130` (the count check) and `:140` (the zip with `request.items`) stay as they are.

#### Path B: `encode_batch`, the padded tensor path

The path is `src/encoding/dense/inference.rs:469-560`.

**How it works today.**

- `tokenizer.encode_batch(texts, true)` pads every text to the longest in the call.
- One activation preflight runs at `batch × max` (`:497`).
- One forward runs on `[batch, max]`.
- The JinaBERT branch (`supports_padded_batch == false`, `:489`) already runs text by text. It is
  left as it is.

**Design.**

1. **Sort.** Tokenise once *without* padding: `encode_batch(texts, false)`, at `src/core/tokenizer.rs:540`.
   Then build the same stable `order` by length. For Path B, sort ascending or descending; only the
   grouping matters here.
2. **Group.** Use the same rule as Path A: the largest runs whose `len × group_max` pass
   `validate_transformer_activations`.
3. **Pad per group.** Pad each group to its own max with the pad id and mask 0 that
   `encode_batch(…, true)` uses today. **NOT CONFIRMED** where the tokenizer exposes the pad id. If it
   does not, call `encode_batch(group_texts, true)` per group instead. That tokenises twice, which is
   cheap next to the forward.
4. **Run.** Build tensors and run the forward and pooling as now, then mask-normalise each group. The
   DistilBERT mask inversion (`:530`) applies per group unchanged.
5. **Restore.** Write each pooled row to `out[order[k]]`.
6. **Keep these unchanged.** The one-text fast path (`:475`) stays. The whole function stays under one
   inference permit (`:553`).

This turns Path B into the shared helper "padded forward of one group", which Path A reuses in its
step 3.

#### Tests that prove order and vectors are unchanged

Write each test first and see it red, per our rule.

**Pure-function tests** (new module beside `inference.rs`; no model needed):

1. **Permutation restores the caller's order.** `length_order` plus `restore` returns the input
   unchanged for:
   - all lengths equal;
   - strictly increasing;
   - strictly decreasing;
   - ties mixed with distinct lengths;
   - n = 0 and n = 1;
   - a deterministic sweep of 1,000 shuffled cases (fixed seed, no new dependency).
2. **Ties keep the caller's order.** For equal lengths `[5, 5, 5]`, the order is `[0, 1, 2]`.
3. **Groups respect the budget.** Each group passes `validate_transformer_activations`.
4. **Groups cover every item exactly once.** Their concatenation is a permutation of `0..n`.
5. **An item over the budget alone is a group of one.** The existing preflight then refuses it with
   the same error as today. Nothing is cut.
6. **The error index is the caller's.** A forced `OutputInvalid` from sorted position k reports
   `index == order[k]`.

**Model-level tests** (in `src/encoding/dense/tests.rs`, beside
`mixed_length_bert_batch_matches_sequential_forward` at `:246`, with the same tiny BERT fixture):

7. **Sorted batch matches each text alone.** `encode_batch` of
   `[long, short, mid, short2, long2]` equals `encode(text)` for each text, in caller order, within
   1e-5. That is the same bound the existing test uses, and well inside the references' 1e-3.
8. **A uniform batch is bit-identical.** When every text has the same token count and the budget holds
   them in one group, the output is **bit-identical** to the unsorted path. There is one group, the
   same shape and the same forward, so this asserts `==`, not a tolerance. Keep the old function under
   `#[cfg(test)]` as the oracle for that one test.
9. **Same for Path A.** `encode_batch_outcomes` gives the same order and parity for the
   mixed-length list in test 7, including each outcome's `tokens_total`.

**Worker protocol tests** (`crates/tessera-worker/tests/`):

10. These existing tests stay green unchanged:
    - `installed_worker_reports_identity_roles_and_ordered_refusals` (`protocol.rs:203`);
    - `long_ids_leave_whole_ordered_items_and_an_omitted_count` (`protocol/never_cut.rs:33`).
11. **New: ids stay in request order.** A request with distinct ids and mixed lengths returns items in
    request id order. Each vector equals the single-item request for that text within 1e-5.

**Certification** (on Tom's Mac, at a later cargo turn):

12. Run `cert run` for each dense profile before and after. Every comparison passes, and the report
    shows `max_absolute_error` before and after.

**Expected counts.** The handback states expected and observed counts as usual: +6 pure-function,
+3 model-level and +1 protocol, so 10 new tests. The 2 existing protocol tests and the existing parity
test stay green.
