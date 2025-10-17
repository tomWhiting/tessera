# ColPali Model Registry Update Report

## Summary

Successfully updated the Tessera model registry with accurate ColPali metadata sourced from HuggingFace. Updated the existing ColPali v1.2 entry and added the new ColPali v1.3-hf model with verified specifications.

## Changes Made

### 1. Updated ColPali v1.2 Entry (lines 226-268)

**Corrected Fields:**
- **License**: Changed from "Apache-2.0" to "gemma" (Gemma License)
- **Organization**: Changed from "Vidore" to "vidore" (lowercase)
- **HuggingFace ID**: Updated to "vidore/colpali-v1.2-hf" (HF transformers version)
- **Model Name**: Updated to "ColPali v1.2 HF"
- **Architecture Variant**: Changed from "paligemma-3b" to "paligemma-3b-mix-448"
- **Vocab Size**: Corrected from 257152 to 257216
- **File Structure**: Updated to sharded safetensors format:
  - `model-00001-of-00002.safetensors` (4.99 GB)
  - `model-00002-of-00002.safetensors` (862 MB)
- **Performance Metrics**: Replaced BEIR/MS-MARCO with ViDoRe benchmark scores:
  - `vidore_ndcg5`: 0.505 (average score on ViDoRe v2 benchmark)
  - `vidore_range`: [0.321, 0.585] (score range across datasets)

**Added Fields:**
- `image_size`: 448
- `patch_size`: 14
- `image_resolution`: [448, 448]
- `num_patches`: 1024

### 2. Added ColPali v1.3-hf Entry (lines 269-312)

**New Model Entry with Verified Data:**
- **ID**: "colpali-v1.3-hf"
- **HuggingFace ID**: "vidore/colpali-v1.3-hf"
- **Organization**: "vidore" (lowercase)
- **License**: "gemma" (Gemma License from PaliGemma backbone)
- **Architecture**: 
  - Type: "paligemma"
  - Variant: "paligemma-3b-mix-448"
  - Has projection: true
  - Projection dims: 128
  - Image size: 448
  - Patch size: 14
- **Specifications**:
  - Parameters: "3B"
  - Embedding dim: 128
  - Hidden dim: 2048
  - Context length: 512
  - Image resolution: [448, 448]
  - Num patches: 1024
  - Vocab size: 257216
- **Files**:
  - Tokenizer: "tokenizer.json"
  - Config: "config.json"
  - Weights: Sharded safetensors (2 files)
- **Capabilities**:
  - Languages: ["multilingual"]
  - Modalities: ["vision", "text"]
  - Multi-vector: true
  - Quantization: ["fp32", "fp16", "int8"]
- **Performance**:
  - ViDoRe NDCG@5: 0.546 (improved from v1.2's 0.505)
  - Score range: [0.499, 0.598]

## Data Sources

All metadata was extracted from official HuggingFace model pages:

1. **vidore/colpali-v1.2-hf**: https://huggingface.co/vidore/colpali-v1.2-hf
   - Config.json for specifications
   - Files tab for model structure
   
2. **vidore/colpali-v1.3-hf**: https://huggingface.co/vidore/colpali-v1.3-hf
   - Config.json for specifications
   - Files tab for model structure

3. **ViDoRe Benchmark v2**: https://huggingface.co/blog/manu/vidore-v2
   - Performance metrics (NDCG@5 scores)

## Technical Details Verified

### Architecture
- Base model: google/paligemma-3b-mix-448
- Vision backbone: SigLIP (so400m-patch14-384)
- Retrieval strategy: ColBERT-style multi-vector representations
- Training: LoRA adapters (r=32, alpha=32) on 127,460 query-page pairs

### File Structure
Both models use sharded safetensors format:
- 2 safetensors files totaling ~5.85 GB
- Model index file for layer mapping
- Complete tokenizer and config files

### Performance
ColPali v1.3 shows measurable improvement over v1.2:
- Average NDCG@5: 0.546 vs 0.505 (+0.041)
- More consistent performance across datasets
- Better zero-shot multilingual capabilities

## Quality Checklist

- [x] All data sourced from HuggingFace (no guesses or placeholders)
- [x] License correctly specified as "gemma"
- [x] Performance metrics accurate from ViDoRe v2 benchmark
- [x] JSON is valid (verified with json.tool)
- [x] File structure matches actual model repos
- [x] Organization name is lowercase "vidore"
- [x] Embedding dimensions verified (128 for ColPali)
- [x] Num patches calculated correctly: (448/14)^2 = 1024
- [x] Vocab size accurate: 257216
- [x] No mock data or placeholder implementations
- [x] No trailing commas in JSON

## Files Modified

- `/Users/tom/Developer/spaces/projects/hyperspatial/main/hypiler/models.json`
  - Lines 226-268: Updated ColPali v1.2 entry
  - Lines 269-312: Added ColPali v1.3-hf entry

## Next Steps

1. Consider adding other ColPali variants:
   - vidore/colpali-v1.3 (non-HF version)
   - vidore/colqwen2-v1.0 (newer Qwen2.5-based variant with NDCG@5 ~0.599)

2. Monitor for future ColPali releases on the ViDoRe leaderboard

3. Update performance metrics when ViDoRe v3 benchmark is released

## References

- ColPali Paper: https://arxiv.org/abs/2407.01449
- ViDoRe Benchmark: https://github.com/illuin-tech/vidore-benchmark
- ViDoRe v2 Blog: https://huggingface.co/blog/manu/vidore-v2
- PaliGemma Model: https://huggingface.co/google/paligemma-3b-mix-448
