# Dense Embeddings Test Examples

## Example Test Output

### Running All Tests (No Model Downloads)
```bash
$ cargo test --test dense_embeddings_test

running 28 tests
test test_builder_dimension_on_fixed_model ... ignored
test test_dense_batch_encoding ... ignored
test test_dense_batch_order_preservation ... ignored
test test_dense_batch_vs_sequential_consistency ... ignored
test test_dense_embedding_not_sparse ... ignored
test test_dense_encode_single ... ignored
test test_dense_metadata_preservation ... ignored
test test_dense_model_info_methods ... ignored
test test_dense_normalization ... ignored
test test_dense_normalized_dot_equals_cosine ... ignored
test test_dense_pooling_strategy ... ignored
test test_dense_similarity_identical ... ignored
test test_dense_similarity_semantic ... ignored
test test_device_auto_selection ... ignored
test test_device_explicit_cpu ... ignored
test test_device_metal_on_macos ... ignored
test test_encode_empty_string ... ignored
test test_factory_both_variants ... ignored
test test_factory_dense_model ... ignored
test test_factory_multivector_model ... ignored
test test_matryoshka_dimension_truncation ... ignored
test test_matryoshka_prefix_consistency ... ignored
test test_error_invalid_model_id ... ok
test test_builder_invalid_model ... ok
test test_builder_requires_model ... ok
test test_error_messages_are_clear ... ok
test test_builder_wrong_model_type ... ok
test test_builder_unsupported_dimension ... ok

test result: ok. 6 passed; 0 failed; 22 ignored; 0 measured; 0 filtered out
```

### Running Specific Test with Model Download
```bash
$ cargo test --test dense_embeddings_test test_dense_encode_single -- --ignored

running 1 test
test test_dense_encode_single ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 27 filtered out
```

## Test Categories by Complexity

### Level 1: Unit Tests (No Model Required)
Fast validation tests that don't require downloading models:
- `test_builder_requires_model` - Builder validation
- `test_builder_invalid_model` - Invalid model ID handling
- `test_builder_wrong_model_type` - Type mismatch detection
- `test_builder_unsupported_dimension` - Dimension validation
- `test_error_invalid_model_id` - Error message quality
- `test_error_messages_are_clear` - User-friendly errors

### Level 2: Basic Integration (Requires Model)
Core functionality tests with single model:
- `test_dense_encode_single` - Basic encoding
- `test_dense_similarity_semantic` - Similarity scoring
- `test_dense_normalization` - L2 norm validation
- `test_dense_model_info_methods` - API methods

### Level 3: Advanced Integration (Requires Models)
Complex workflows and edge cases:
- `test_matryoshka_dimension_truncation` - Multiple dimensions
- `test_matryoshka_prefix_consistency` - Dimension relationships
- `test_dense_batch_vs_sequential_consistency` - Batch processing
- `test_factory_both_variants` - Factory pattern with multiple models

## Example Test Code Snippets

### Basic Encoding Test
```rust
#[test]
#[ignore] // Requires model download
fn test_dense_encode_single() {
    let embedder = TesseraDense::new("bge-base-en-v1.5")
        .expect("Failed to create embedder");
    
    let text = "What is machine learning?";
    let embedding = embedder.encode(text)
        .expect("Failed to encode text");
    
    assert_eq!(embedding.dim(), 768, "Expected 768-dim embedding");
    
    let sum: f32 = embedding.embedding.iter().sum();
    assert!(sum.abs() > 0.01, "Embedding should not be all zeros");
}
```

### Error Handling Test
```rust
#[test]
fn test_builder_requires_model() {
    let result = TesseraDenseBuilder::new().build();

    assert!(result.is_err(), "Should error when model ID not provided");
    if let Err(err) = result {
        assert!(
            err.to_string().contains("Model ID must be specified"),
            "Error should mention missing model ID"
        );
    }
}
```

### Matryoshka Test
```rust
#[test]
#[ignore]
fn test_matryoshka_dimension_truncation() {
    let embedder_768 = TesseraDense::builder()
        .model("nomic-embed-v1.5")
        .dimension(768)
        .build()
        .expect("Failed to create 768-dim embedder");
    
    let embedder_256 = TesseraDense::builder()
        .model("nomic-embed-v1.5")
        .dimension(256)
        .build()
        .expect("Failed to create 256-dim embedder");
    
    let text = "What is machine learning?";
    
    let emb_768 = embedder_768.encode(text).unwrap();
    let emb_256 = embedder_256.encode(text).unwrap();
    
    assert_eq!(emb_768.dim(), 768);
    assert_eq!(emb_256.dim(), 256);
}
```

### Factory Pattern Test
```rust
#[test]
#[ignore]
fn test_factory_dense_model() {
    let embedder = Tessera::new("bge-base-en-v1.5")
        .expect("Failed to create embedder via factory");
    
    match embedder {
        Tessera::Dense(dense) => {
            let embedding = dense.encode("Test factory pattern")
                .expect("Failed to encode");
            assert_eq!(embedding.dim(), 768);
        }
        Tessera::MultiVector(_) => {
            panic!("Factory should have returned Dense variant");
        }
    }
}
```

## Test Execution Time Estimates

- **Unit tests** (6 tests): < 1 second
- **Basic integration** (10 tests): ~30-60 seconds (first run with download)
- **Advanced integration** (12 tests): ~2-3 minutes (multiple models)
- **Full suite** (28 tests): ~3-5 minutes (first run, < 1 min after cache)

## CI/CD Integration

### GitHub Actions Example
```yaml
- name: Run unit tests (no model downloads)
  run: cargo test --test dense_embeddings_test

- name: Run integration tests (with model downloads)
  run: cargo test --test dense_embeddings_test -- --ignored
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'
```

### Local Development Workflow
1. **During development**: Run unit tests only
   ```bash
   cargo test --test dense_embeddings_test
   ```

2. **Before commit**: Run basic integration tests
   ```bash
   cargo test --test dense_embeddings_test test_dense_encode -- --ignored
   ```

3. **Before PR**: Run full suite
   ```bash
   cargo test --test dense_embeddings_test -- --ignored
   ```

## Debugging Failed Tests

### If Test Fails During Encoding
1. Check model registry is correct (`models.json`)
2. Verify model files are downloaded
3. Check device compatibility (Metal/CUDA/CPU)
4. Review error message for context

### If Test Fails During Similarity
1. Verify normalization is enabled for model
2. Check embedding dimensions match
3. Review similarity computation logic
4. Compare with expected ranges (typically 0.0-1.0)

### If Test Fails During Batch Processing
1. Verify all texts are non-empty
2. Check batch size limitations
3. Review memory usage
4. Ensure order preservation

## Performance Optimization Tips

### For Faster Test Runs
1. Use smaller models for basic tests (`colbert-small` instead of `colbert-v2`)
2. Cache model downloads in CI/CD
3. Run unit tests in parallel
4. Skip ignored tests during development

### For More Thorough Testing
1. Test with multiple model architectures
2. Add stress tests with large batches
3. Test edge cases (empty strings, very long texts)
4. Add performance benchmarks
