# ColPali Shape Incompatibility Fix

## Issue Summary

The ColPali PDF demo was failing with a shape mismatch error when computing similarity between text queries and image embeddings:

```
Error: ShapeError/IncompatibleShape: incompatible shapes
```

## Root Cause Analysis

### Problem
ColPali v1.2-merged has asymmetric projection behavior:
- **Text embeddings**: PaliGemma output (2048 dims) → projected to 128 dims via `custom_text_proj`
- **Image embeddings**: PaliGemma output (2048 dims) → **NOT projected** (stayed at 2048 dims)

This caused MaxSim to fail because it requires matching embedding dimensions:
- Query: `[num_tokens, 128]`
- Document: `[num_patches, 2048]` ❌ Dimension mismatch!

### Model Architecture Details

**ColPali v1.2-merged** (vidore/colpali-v1.2-merged):
- Has only `custom_text_proj` layer (2048 → 128)
- No separate vision projection layer
- Intended behavior: Use the SAME projection for both modalities

**ColPali v1.3** (vidore/colpali-v1.3-hf):
- Has unified `embedding_proj_layer` (2048 → 128)
- Cleaner architecture with explicit unified projection

### Why It Mattered

ColPali uses MaxSim (late interaction) for retrieval:
```
MaxSim(Q, D) = Σᵢ max_ⱼ (qᵢ · dⱼ)
```

This requires:
1. Query tokens qᵢ and document patches dⱼ must have the same embedding dimension
2. Both must be L2-normalized for proper similarity scoring
3. Lower dimensions (128) are more efficient than high dimensions (2048)

## Solution

Apply the `custom_text_proj` layer to **both** text and vision embeddings, since ColPali v1.2-merged uses the same projection layer for both modalities.

### Code Changes

**File:** `src/encoding/vision.rs`

**Before (lines 269-279):**
```rust
// 9. Convert to CPU and extract as Vec<Vec<f32>>
let embeddings = self.tensor_to_vec2(&patch_embeddings)
    .context("Failed to convert patch embeddings to Vec<Vec<f32>>")?;

// 10. Create VisionEmbedding
Ok(VisionEmbedding::new(
    embeddings,
    self.num_patches,
    self.embedding_dim,
    Some(image_path.to_string_lossy().to_string()),
))
```

**After (lines 269-293):**
```rust
// 9. Apply custom text projection to image embeddings (2048 -> 128)
// Note: In ColPali v1.2-merged, the same projection layer is used for both
// text and vision embeddings to project from PaliGemma's hidden size (2048)
// to ColPali's embedding dimension (128) for efficient late interaction.
let projected = self.custom_text_projection.forward(&patch_embeddings)
    .context("Failed to apply projection to image embeddings")?;

// 10. Apply L2 normalization
let norms = projected.sqr()?
    .sum_keepdim(1)?  // Sum over embedding dimension
    .sqrt()?;
let normalized = projected.broadcast_div(&norms)
    .context("Failed to normalize image embeddings")?;

// 11. Convert to CPU and extract as Vec<Vec<f32>>
let embeddings = self.tensor_to_vec2(&normalized)
    .context("Failed to convert patch embeddings to Vec<Vec<f32>>")?;

// 12. Create VisionEmbedding with correct embedding dimension (128)
Ok(VisionEmbedding::new(
    embeddings,
    self.num_patches,
    self.embedding_dim,
    Some(image_path.to_string_lossy().to_string()),
))
```

### Key Changes:
1. **Added projection**: Apply `custom_text_projection` to image embeddings
2. **Added normalization**: L2-normalize projected image embeddings
3. **Updated comments**: Clarify the unified projection behavior

## Verification

### Shape Verification Test
Created `examples/verify_colpali_shapes.rs` to confirm dimensions match:

```
✓ SUCCESS: Text and image embedding dimensions match!
  Both use embedding_dim = 128

Text embeddings: projected from 2048 → 128
Image embeddings: projected from 2048 → 128
```

### PDF Demo Results
The `colpali_pdf_demo` now successfully computes similarity:

```
Query: "transformer architecture"

Top 5 most relevant pages:
  1. Page 13 - Score: 0.9154
  2. Page 14 - Score: 0.8450
  3. Page 3 - Score: 0.7733
  4. Page 6 - Score: 0.7679
  5. Page 5 - Score: 0.7462
```

These results make sense for the "Attention is All You Need" paper, with the highest scores on pages containing architecture diagrams and descriptions.

## Technical Details

### Embedding Pipeline

**Text:**
1. Tokenize text → token IDs
2. PaliGemma language model forward → `[num_tokens, 2048]`
3. Apply `custom_text_proj` → `[num_tokens, 128]`
4. L2 normalize → final text embeddings

**Vision:**
1. Preprocess image → `[3, 448, 448]`
2. PaliGemma vision encoder → `[1024, 2048]` (32×32 patches)
3. Apply `custom_text_proj` → `[1024, 128]` ✅ **NEW**
4. L2 normalize → final vision embeddings ✅ **NEW**

### MaxSim Computation

Now that dimensions match:
```rust
// Query: [num_query_tokens, 128]
// Document: [num_patches, 128]
// Similarity matrix: [num_query_tokens, num_patches]
let similarity_matrix = query_matrix.dot(&doc_matrix.t());

// For each query token, find max similarity across document patches
let max_sims = similarity_matrix.map_axis(Axis(1), |row| row.max());

// Sum all maximum similarities
let total_score = max_sims.sum();
```

## Impact

✅ **Fixed:** ColPali PDF document search now works correctly
✅ **Verified:** Text and image embeddings have matching dimensions (128)
✅ **Tested:** MaxSim similarity computation produces reasonable scores
✅ **Architecture:** Correctly implements ColPali v1.2-merged's unified projection

## Files Changed

1. `src/encoding/vision.rs` - Added projection and normalization for image embeddings
2. `examples/verify_colpali_shapes.rs` - Created verification test (NEW)
3. `COLPALI_SHAPE_FIX.md` - This documentation (NEW)

## Related Examples

- `examples/colpali_pdf_demo.rs` - PDF document search (now working)
- `examples/verify_colpali_shapes.rs` - Shape verification test
- `examples/test_colpali_shapes.rs` - Integration test

## Future Considerations

For ColPali v1.3 support, consider:
- Loading `embedding_proj_layer` instead of `custom_text_proj`
- Using the unified projection layer for both modalities
- Updating model registry to support both v1.2 and v1.3 architectures
