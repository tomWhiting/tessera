# Dense Embeddings Integration Tests - Phase 2.1

## Overview

Comprehensive integration test suite for `TesseraDense` API covering all major functionality including basic encoding, batch processing, similarity computation, Matryoshka support, factory patterns, and error handling.

## Test Statistics

- **Total Tests**: 28
- **Ignored Tests** (require model download): 22
- **Unit Tests** (no model required): 6
- **Test File**: `tests/dense_embeddings_test.rs`

## Test Coverage by Category

### 1. Basic Dense Encoding (2 tests)
- ✅ `test_dense_encode_single` - Single text encoding, dimension validation, non-zero check
- ✅ `test_dense_metadata_preservation` - Verify text and metadata preservation

### 2. Batch Processing (3 tests)
- ✅ `test_dense_batch_encoding` - Multiple texts in batch, dimension validation
- ✅ `test_dense_batch_vs_sequential_consistency` - Compare batch vs sequential encoding
- ✅ `test_dense_batch_order_preservation` - Verify input order is preserved

### 3. Similarity Computation (2 tests)
- ✅ `test_dense_similarity_semantic` - Similar vs dissimilar text comparison
- ✅ `test_dense_similarity_identical` - Identical text similarity ≈ 1.0

### 4. Normalization Validation (2 tests)
- ✅ `test_dense_normalization` - L2 normalization verification (magnitude ≈ 1.0)
- ✅ `test_dense_normalized_dot_equals_cosine` - Dot product equals cosine similarity

### 5. Pooling Strategy (1 test)
- ✅ `test_dense_pooling_strategy` - Mean pooling behavior validation

### 6. Matryoshka Support (2 tests)
- ✅ `test_matryoshka_dimension_truncation` - Multiple dimensions (768, 256, 64)
- ✅ `test_matryoshka_prefix_consistency` - Smaller dims are prefixes of larger

### 7. Factory Pattern (3 tests)
- ✅ `test_factory_dense_model` - Tessera::new() with dense model → Dense variant
- ✅ `test_factory_multivector_model` - Tessera::new() with ColBERT → MultiVector variant
- ✅ `test_factory_both_variants` - Pattern matching and using both variants

### 8. Builder Validation (5 tests)
- ✅ `test_builder_requires_model` - Error when model ID not provided
- ✅ `test_builder_invalid_model` - Error for nonexistent model
- ✅ `test_builder_wrong_model_type` - Error using ColBERT model with TesseraDense
- ✅ `test_builder_unsupported_dimension` - Error for invalid Matryoshka dimension
- ✅ `test_builder_dimension_on_fixed_model` - Error setting dimension on fixed model

### 9. Device Selection (3 tests)
- ✅ `test_device_auto_selection` - Auto device selection works
- ✅ `test_device_explicit_cpu` - Explicit CPU device selection
- ✅ `test_device_metal_on_macos` - Metal device on macOS (if available)

### 10. Error Handling (3 tests)
- ✅ `test_error_invalid_model_id` - Clear error for invalid model
- ✅ `test_error_messages_are_clear` - Helpful error messages with context
- ✅ `test_encode_empty_string` - Handles empty string gracefully

### 11. Additional Quality Tests (2 tests)
- ✅ `test_dense_model_info_methods` - model() and dimension() methods
- ✅ `test_dense_embedding_not_sparse` - Dense embeddings are >90% non-zero

## Models Used for Testing

### Primary Test Model
- **bge-base-en-v1.5**: Standard dense model, 768 dims, mean pooling, normalized

### Matryoshka Test Model
- **nomic-embed-v1.5**: Supports dimensions [64, 128, 256, 512, 768]

### Multi-Vector Comparison
- **colbert-v2**: Used for factory pattern tests to verify type distinction

## Test Patterns Followed

### From Phase 1 Tests
1. ✅ Descriptive test names with clear intent
2. ✅ `#[ignore]` for tests requiring model downloads
3. ✅ Use `assert!`, `assert_eq!` with helpful messages
4. ✅ Test both success and error cases
5. ✅ Clear assertion messages with context
6. ✅ Proper error handling with `if let Err(e)` pattern

### Consistency Checks
- Batch vs sequential encoding consistency
- Similar vs dissimilar text scoring
- Normalization verification
- Matryoshka prefix consistency

## Running the Tests

### Run all tests (requires model downloads)
```bash
cargo test --test dense_embeddings_test -- --ignored
```

### Run only unit tests (no downloads)
```bash
cargo test --test dense_embeddings_test
```

### Run specific test
```bash
cargo test --test dense_embeddings_test test_dense_encode_single -- --ignored
```

### List all tests
```bash
cargo test --test dense_embeddings_test -- --list
```

## Quality Checklist

- [x] At least 10 comprehensive tests (28 total)
- [x] Cover all major API surfaces
- [x] Test both success and failure cases
- [x] Clear, descriptive test names
- [x] Helpful assertion messages
- [x] Follow Phase 1 test patterns
- [x] Use appropriate `#[ignore]` tags
- [x] No TODOs or placeholders
- [x] Tests compile without warnings (except unused import false positive)
- [x] All non-ignored tests pass

## Implementation Notes

### Real Data Only
All tests use actual model implementations with no mock data or placeholders. Tests requiring model downloads are marked with `#[ignore]` for convenience during development.

### Error Handling Pattern
Tests use the `if let Err(e)` pattern to avoid `unwrap_err()` issues with non-Debug types:
```rust
if let Err(err) = result {
    assert!(err.to_string().contains("expected message"));
} else {
    panic!("Expected error");
}
```

### Matryoshka Testing
Tests verify that Matryoshka models:
1. Accept different target dimensions
2. Produce correct output dimensions
3. Maintain prefix consistency (small dims are prefix of large dims)
4. Reject invalid dimensions with clear errors

### Factory Pattern Testing
Tests verify the `Tessera` enum correctly:
1. Auto-detects model type from registry
2. Returns appropriate variant (Dense vs MultiVector)
3. Enables pattern matching to access variant-specific APIs

## Next Steps

1. Run ignored tests with actual models to verify functionality
2. Add performance benchmarks for batch encoding
3. Consider adding tests for edge cases (very long text, special characters)
4. Add integration tests with actual retrieval workflows
