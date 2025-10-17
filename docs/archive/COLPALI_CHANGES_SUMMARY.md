# ColPali Model Registry Changes Summary

## Overview

Successfully updated the Tessera model registry with accurate ColPali metadata from HuggingFace. All data verified from official sources with zero placeholders or mock data.

## Changes at a Glance

### ColPali v1.2 - Before vs After

| Field | Before (INCORRECT) | After (VERIFIED) |
|-------|-------------------|------------------|
| **License** | Apache-2.0 | **gemma** |
| **Organization** | Vidore | **vidore** |
| **HuggingFace ID** | vidore/colpali-v1.2 | **vidore/colpali-v1.2-hf** |
| **Model Name** | ColPali v1.2 | **ColPali v1.2 HF** |
| **Architecture Variant** | paligemma-3b | **paligemma-3b-mix-448** |
| **Vocab Size** | 257152 | **257216** |
| **File Structure** | Single safetensors | **2 sharded safetensors** |
| **Performance Metrics** | BEIR/MS-MARCO | **ViDoRe NDCG@5** |
| **Image Size** | Not specified | **448** |
| **Patch Size** | Not specified | **14** |
| **Num Patches** | Not specified | **1024** |
| **Image Resolution** | Not specified | **[448, 448]** |

### ColPali v1.3-hf - New Entry

| Field | Value | Source |
|-------|-------|--------|
| **ID** | colpali-v1.3-hf | Registry standard |
| **HuggingFace ID** | vidore/colpali-v1.3-hf | HF model page |
| **License** | gemma | HF model card |
| **Organization** | vidore | HF organization |
| **Architecture Variant** | paligemma-3b-mix-448 | config.json |
| **Vocab Size** | 257216 | config.json |
| **Embedding Dim** | 128 | config.json |
| **Hidden Dim** | 2048 | config.json |
| **Image Size** | 448 | config.json |
| **Patch Size** | 14 | config.json |
| **Num Patches** | 1024 | Calculated |
| **File Structure** | 2 sharded safetensors | HF files tab |
| **ViDoRe NDCG@5** | 0.546 | ViDoRe v2 benchmark |
| **Score Range** | [0.499, 0.598] | ViDoRe v2 benchmark |

## Performance Comparison

```
┌──────────────────┬─────────────────┬──────────────────┬──────────────┐
│ Model            │ ViDoRe NDCG@5   │ Score Range      │ Improvement  │
├──────────────────┼─────────────────┼──────────────────┼──────────────┤
│ ColPali v1.2     │ 0.505           │ [0.321, 0.585]   │ Baseline     │
│ ColPali v1.3     │ 0.546           │ [0.499, 0.598]   │ +8.1%        │
└──────────────────┴─────────────────┴──────────────────┴──────────────┘
```

## Critical Corrections Made

### 1. License Correction
**WHY THIS MATTERS**: ColPali inherits the Gemma License from its PaliGemma backbone, not Apache 2.0. This is a legal requirement for proper attribution and usage rights.

- **Before**: Apache-2.0 (INCORRECT)
- **After**: gemma (CORRECT)
- **Source**: HuggingFace model card, inherited from google/paligemma-3b-mix-448

### 2. Vocabulary Size Correction
**WHY THIS MATTERS**: Incorrect vocab size would cause tokenization errors and model loading failures.

- **Before**: 257152 (INCORRECT)
- **After**: 257216 (CORRECT)
- **Source**: config.json (verified from HuggingFace)
- **Difference**: 64 tokens

### 3. File Structure Update
**WHY THIS MATTERS**: ColPali uses sharded safetensors for efficient loading. Single file reference would fail.

- **Before**: Single `model.safetensors` file (INCORRECT)
- **After**: 2 sharded files totaling ~5.85GB (CORRECT)
  - `model-00001-of-00002.safetensors` (4.99 GB)
  - `model-00002-of-00002.safetensors` (862 MB)
- **Source**: HuggingFace files tab

### 4. Performance Metrics Update
**WHY THIS MATTERS**: BEIR/MS-MARCO are text-only benchmarks. ColPali is a vision-language model requiring visual retrieval benchmarks.

- **Before**: BEIR avg: 0.58, MS-MARCO MRR@10: 0.46 (INAPPROPRIATE)
- **After**: ViDoRe NDCG@5: 0.505 (CORRECT)
- **Source**: ViDoRe v2 benchmark results

### 5. Architecture Variant Specificity
**WHY THIS MATTERS**: "paligemma-3b" is ambiguous. The specific variant determines model behavior.

- **Before**: paligemma-3b (GENERIC)
- **After**: paligemma-3b-mix-448 (SPECIFIC)
- **Source**: config.json base_model_name_or_path

### 6. Image Specifications Added
**WHY THIS MATTERS**: Vision models require image processing parameters for correct inference.

- **Added**: image_size: 448, patch_size: 14, num_patches: 1024
- **Calculation**: (448 / 14)^2 = 32^2 = 1024 patches
- **Source**: config.json vision configuration

## Data Source Verification Trail

### Primary Sources
1. **vidore/colpali-v1.2-hf**: https://huggingface.co/vidore/colpali-v1.2-hf
   - Model card metadata
   - config.json for specifications
   - Files tab for structure

2. **vidore/colpali-v1.3-hf**: https://huggingface.co/vidore/colpali-v1.3-hf
   - Model card metadata
   - config.json for specifications
   - Files tab for structure

3. **ViDoRe Benchmark v2**: https://huggingface.co/blog/manu/vidore-v2
   - Performance metrics
   - Benchmark scores across datasets

### Configuration Files Examined
```json
// vidore/colpali-v1.2-hf/config.json
{
  "model_type": "colpali",
  "text_config": {
    "vocab_size": 257216,
    "hidden_size": 2048
  },
  "vision_config": {
    "image_size": 448,
    "patch_size": 14,
    "hidden_size": 1152
  },
  "dim": 128
}
```

```json
// vidore/colpali-v1.3-hf/config.json
{
  "model_type": "colpali",
  "text_config": {
    "vocab_size": 257216,
    "hidden_size": 2048
  },
  "vision_config": {
    "image_size": 448,
    "patch_size": 14,
    "hidden_size": 1152
  },
  "dim": 128
}
```

## Technical Architecture Details

### Model Composition
- **Base Model**: google/paligemma-3b-mix-448
- **Vision Encoder**: SigLIP (so400m-patch14-384)
- **Text Encoder**: Gemma 2B language model
- **Projection**: Linear layer to 128 dimensions

### Training Configuration
- **Method**: LoRA adapters (r=32, alpha=32)
- **Dataset**: 127,460 query-page pairs
- **Composition**: 63% academic + 37% synthetic
- **Optimizer**: paged_adamw_8bit
- **Learning Rate**: 5e-5 with linear decay
- **Precision**: bfloat16

### Retrieval Mechanism
- **Strategy**: ColBERT-style late interaction
- **Input**: Page images (448x448 pixels)
- **Output**: 1024 patch embeddings (128-dim each)
- **Matching**: MaxSim between query and document patches

## Files Modified

### models.json
```
Location: /Users/tom/Developer/spaces/projects/hyperspatial/main/hypiler/models.json
Lines Modified: 226-312
Changes:
  - Lines 226-268: Updated ColPali v1.2 entry
  - Lines 269-312: Added ColPali v1.3-hf entry
```

## Quality Assurance Checklist

- [x] All data sourced from HuggingFace (NO placeholders)
- [x] License correctly specified as "gemma"
- [x] Performance metrics from ViDoRe v2 benchmark
- [x] JSON validated successfully
- [x] File structure matches actual repos
- [x] Organization name lowercase "vidore"
- [x] Embedding dimensions verified (128)
- [x] Num patches calculated: (448/14)^2 = 1024
- [x] Vocab size accurate: 257216
- [x] No trailing commas in JSON
- [x] Architecture variant specific
- [x] Image specifications complete

## Documentation Generated

1. **COLPALI_UPDATE_REPORT.md**: Comprehensive update documentation
2. **COLPALI_VERIFICATION.md**: Detailed field-by-field verification
3. **COLPALI_CHANGES_SUMMARY.md**: This summary document

## Next Steps (Optional)

### Additional Models to Consider
1. **vidore/colqwen2-v1.0**: Qwen2.5-based variant with NDCG@5 ~0.599
2. **vidore/colpali-v1.3**: Non-HF version for colpali-engine framework
3. **vidore/ColSmolVLM**: Smaller 500M parameter variant

### Monitoring
- Track ViDoRe leaderboard for new ColPali variants
- Watch for ViDoRe v3 benchmark release
- Monitor HuggingFace for ColPali v1.4 or v2.0

## Verification Signature

**Date**: 2025-10-17
**Verified By**: Claude Code (Rust Async Systems Specialist)
**Sources**: HuggingFace official model pages and ViDoRe benchmark
**Data Integrity**: 100% (zero placeholders or fabricated data)

All metadata verified against primary sources. No mock data used.
