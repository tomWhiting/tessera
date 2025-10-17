# Chronos Bolt Quantile Implementation

## Overview

Fixed the Chronos Bolt implementation to properly handle 9 quantiles in the output, matching the pre-trained weights structure from HuggingFace (amazon/chronos-bolt-small).

## Problem

The original implementation expected `output_patch_embedding` to produce 64 dimensions (one per prediction step), but the pre-trained weights actually have 576 dimensions (64 prediction steps × 9 quantiles).

## Solution

### 1. Output Dimension Fix

**File**: `/Users/tom/Developer/spaces/projects/hyperspatial/main/hypiler/src/timeseries/models/chronos_bolt.rs`

**Changes in `ChronosBolt::new()` (lines 180-190)**:
```rust
// BEFORE: output_dim = prediction_length (64)
// AFTER:  output_dim = prediction_length * quantiles.len() (576)

let output_dim = config.prediction_length * config.quantiles.len();
let output_patch_embedding = ResidualMLP::new(
    config.d_model,         // 512
    config.d_model * 4,     // 2048
    output_dim,             // 576 (64 steps × 9 quantiles)
    vb.pp("output_patch_embedding"),
)?;
```

### 2. Output Reshaping

**Changes in `ChronosBolt::forward()` (lines 355-385)**:
```rust
// 1. Apply output_patch_embedding: [batch, d_model] -> [batch, 576]
let output_flat = self.output_patch_embedding.forward(&decoder_squeezed)?;

// 2. Reshape to separate prediction steps and quantiles
// [batch, 576] -> [batch, 64, 9]
let forecast_quantiles = output_flat
    .reshape((batch, self.config.prediction_length, num_quantiles))?;

// 3. Denormalize (broadcast scale over both dimensions)
let scale_reshaped = scale.unsqueeze(1)?;  // [batch, 1] -> [batch, 1, 1]
let denormalized = forecast_quantiles.broadcast_mul(&scale_reshaped)?;

// Returns: [batch, prediction_length, num_quantiles]
```

### 3. New API Methods

**Added `predict_quantiles()` method**:
```rust
pub fn predict_quantiles(&mut self, input: &Tensor) -> Result<Tensor>
```
- Returns full quantile predictions: `[batch, 64, 9]`
- Provides uncertainty quantification
- Allows extracting any quantile level

**Updated `forecast()` method**:
```rust
pub fn forecast(&mut self, input: &Tensor) -> Result<Tensor>
```
- Calls `predict_quantiles()` internally
- Extracts median (0.5 quantile, index 4)
- Returns point forecast: `[batch, 64]`
- Maintains backward compatibility

### 4. Documentation Updates

**Module header** (lines 6-23):
- Updated architecture description to reflect 576-dimensional output
- Added quantile prediction explanation
- Updated example code to show both methods

**Method documentation**:
- `forward()`: Now clearly states it returns quantile predictions
- `predict_quantiles()`: New method with comprehensive docs and examples
- `forecast()`: Updated to explain median extraction

### 5. Test Updates

**Added 3 new tests**:

1. `test_chronos_bolt_forward_pass_shape`:
   - Updated to expect `[batch, 64, 9]` output
   - Validates quantile structure

2. `test_chronos_bolt_forecast_median`:
   - Verifies `forecast()` returns `[batch, 64]`
   - Tests median extraction

3. `test_chronos_bolt_predict_quantiles`:
   - Validates `predict_quantiles()` API
   - Tests quantile indexing

### 6. Examples

**Updated existing example** (`chronos_bolt_test_weights.rs`):
- Added notes about 576-dimensional output
- Explained median vs. full quantiles

**New comprehensive example** (`chronos_bolt_quantiles.rs`):
- Demonstrates full quantile prediction workflow
- Shows uncertainty quantification
- Extracts specific percentiles (10th, 50th, 90th)
- Calculates prediction intervals
- Verifies point forecast matches median

## Configuration

The quantiles are defined in `ChronosBoltConfig`:

```rust
pub struct ChronosBoltConfig {
    // ...
    pub quantiles: Vec<f32>,  // [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
}
```

Default configuration (`chronos_bolt_small()` and `chronos_bolt_base()`):
- 9 quantiles: `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`
- Prediction length: 64 steps
- Output dimensions: 64 × 9 = 576

## API Usage

### Point Forecast (Median)
```rust
let mut model = ChronosBolt::from_pretrained("amazon/chronos-bolt-small", &device)?;
let input = Tensor::randn(0.0, 1.0, (1, 2048), &device)?;
let forecast = model.forecast(&input)?;  // [1, 64]
```

### Full Quantile Predictions
```rust
let quantiles = model.predict_quantiles(&input)?;  // [1, 64, 9]

// Extract specific quantiles
let q10 = quantiles.i((0, .., 0))?;  // 10th percentile
let q50 = quantiles.i((0, .., 4))?;  // 50th percentile (median)
let q90 = quantiles.i((0, .., 8))?;  // 90th percentile
```

### Uncertainty Quantification
```rust
// Calculate 80% prediction interval (10th to 90th percentile)
let quantiles = model.predict_quantiles(&input)?;
let lower = quantiles.i((.., .., 0))?;  // 10th percentile
let upper = quantiles.i((.., .., 8))?;  // 90th percentile
let interval_width = (upper - lower)?;
```

## Testing Results

All 6 tests pass:
```
test timeseries::models::chronos_bolt::tests::test_chronos_bolt_config_validation ... ok
test timeseries::models::chronos_bolt::tests::test_chronos_bolt_scaling_roundtrip ... ok
test timeseries::models::chronos_bolt::tests::test_residual_mlp_forward ... ok
test timeseries::models::chronos_bolt::tests::test_chronos_bolt_predict_quantiles ... ok
test timeseries::models::chronos_bolt::tests::test_chronos_bolt_forecast_median ... ok
test timeseries::models::chronos_bolt::tests::test_chronos_bolt_forward_pass_shape ... ok
```

## Compilation Status

- ✅ Library compiles without errors
- ✅ All examples compile successfully
- ✅ All tests pass
- ⚠️ Minor warnings (unused fields) - do not affect functionality

## Backward Compatibility

- ✅ `forecast()` method maintains same signature and output shape `[batch, 64]`
- ✅ Existing code using `forecast()` continues to work
- ✅ New `predict_quantiles()` method provides opt-in access to full quantiles
- ✅ Config structure unchanged (quantiles were always defined)

## Weight Compatibility

The implementation now correctly loads the pre-trained weights from HuggingFace:
- `output_patch_embedding.hidden_layer.weight`: `[2048, 512]` ✅
- `output_patch_embedding.output_layer.weight`: `[576, 2048]` ✅
- `output_patch_embedding.residual_layer.weight`: `[576, 512]` ✅

Output dimension: 576 = 64 prediction steps × 9 quantiles

## Summary

The Chronos Bolt implementation now correctly:
1. Handles 9 quantiles per prediction step (576 total output dimensions)
2. Provides both point forecasts (median) and full quantile predictions
3. Enables uncertainty quantification through prediction intervals
4. Maintains backward compatibility with existing code
5. Passes all tests and compiles successfully
