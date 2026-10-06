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
  - **NOT CONFIRMED** whether Candle 0.11 (ours) contains PR #3313.
- **Cost to correctness.** In f32 on the GPU, results differ from Accelerate at the rounding level, so
  they need a tolerance, not equal hashes.
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
  batch shape can change the last bits through BLAS blocking. **NOT CONFIRMED** for Accelerate. This
  matters only if references are compared by hash.

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
references. The risk column assumes references compare vectors by hash (`observed_output_sha256`).
If they compare within a tolerance, every "rounding" risk drops to low.

| # | Technique | Expected gain on M-series (source of figure) | Size | Risk to certified references | Journeys helped |
|---|---|---|---|---|---|
| 1 | Content-hash cache (SHA-256 of text + `ModelIdentity` + role/prompt + limits) | Unbounded on repeats, 0 on new text. No figure found (workload-dependent). | Small | None if the key is complete | Backfill throughput (re-runs), one-query latency (repeat queries) |
| 2 | Sort by length within a call, then token-budget batches (restore order) | Removes the padding share. No M-series figure found; ST GPU unpadding up to 3.87x total with fp16 (RTX 3090) | Small | Low: last-bit changes possible from batch shape (NOT CONFIRMED for Accelerate) | Backfill throughput, long text (memory bound per batch) |
| 3 | Overlap tokenisation with the forward pass (bounded channel, worker pool) | At most the tokeniser's share of wall time. No figure found; size it from Trixie's map | Small to medium | None | Backfill throughput |
| 4 | Packed `[T,H]` linears with per-sequence attention (padding-free without a kernel) | Same saving as #2 but within mixed batches. No figure found | Medium | Low (same as #2) | Backfill throughput, long text |
| 5 | Candle Metal, f32 | No BERT figure found. GEMM was 2-11x behind MPS/MLX before PR #3313 (candle #3302); fix in 0.11 NOT CONFIRMED | Medium | Medium: GPU rounding needs a tolerance or separate references per device | Backfill throughput, long text |
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

- **§0 ceiling.** The Accelerate f32 peak on M-series is not from a primary source. Searched only
  general results; no Apple figure found.
- **Candle 0.11 and PR #3313.** Not checked against Candle's changelog or tags.
- **Whether Tessera sorts by length.** A grep of `src/` at `1d600f4` found no length sort (the only
  sort is `src/encoding/minicoil/vector.rs:288`, by index). The call path was not traced; that is
  Trixie's hot-path map.
- **Batch shape and last bits under Accelerate.** Not tested (testing is not in the box).
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
