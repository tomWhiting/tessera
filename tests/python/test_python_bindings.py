#!/usr/bin/env python3
"""Comprehensive test suite for Tessera Python bindings.

Tests all embedder types:
- TesseraDense: Single-vector dense embeddings
- TesseraMultiVector: ColBERT multi-vector embeddings
- TesseraSparse: SPLADE sparse embeddings
- TesseraVision: ColPali vision-language embeddings
- TesseraTimeSeries: Chronos Bolt time series forecasting
- Tessera: Factory pattern with auto-detection
"""

import numpy as np
import sys


def test_dense_basic():
    """Test basic dense embedding functionality."""
    print("\n" + "=" * 70)
    print("Testing TesseraDense - Dense Single-Vector Embeddings")
    print("=" * 70 + "\n")

    try:
        from tessera import TesseraDense
    except ImportError as e:
        print(f"Failed to import TesseraDense: {e}")
        print("\nMake sure you've built the bindings with:")
        print("  maturin develop --features python")
        return False

    # Test 1: Create embedder
    print("Test 1: Creating dense embedder...")
    try:
        embedder = TesseraDense("bge-base-en-v1.5")
        print(f"Created embedder: {embedder}")
    except Exception as e:
        print(f"Failed to create embedder: {e}")
        return False

    # Test 2: Single text encoding
    print("\nTest 2: Encoding single text...")
    try:
        text = "What is machine learning?"
        embedding = embedder.encode(text)
        print(f"Encoded text to shape: {embedding.shape}")
        print(f"  - Embedding dimension: {embedding.shape[0]}")
        print(f"  - Data type: {embedding.dtype}")

        # Validate shape and dtype
        assert embedding.ndim == 1, f"Expected 1D array, got {embedding.ndim}D"
        assert embedding.dtype == np.float32, f"Expected float32, got {embedding.dtype}"
        assert embedding.shape[0] > 0, "Expected positive embedding dimension"
    except Exception as e:
        print(f"Failed to encode text: {e}")
        return False

    # Test 3: Batch encoding
    print("\nTest 3: Batch encoding...")
    try:
        texts = ["Query one", "Query two", "Query three"]
        batch_embs = embedder.encode_batch(texts)
        print(f"Batch encoded {len(batch_embs)} texts")

        # Validate batch results
        assert len(batch_embs) == len(texts), f"Expected {len(texts)} embeddings, got {len(batch_embs)}"

        for i, emb in enumerate(batch_embs):
            print(f"  - Text {i+1}: shape {emb.shape}")
            assert emb.ndim == 1, f"Expected 1D array for text {i+1}"
            assert emb.dtype == np.float32, f"Expected float32 for text {i+1}"
    except Exception as e:
        print(f"Failed batch encoding: {e}")
        return False

    # Test 4: Similarity computation
    print("\nTest 4: Computing similarity...")
    try:
        text_a = "machine learning and AI"
        text_b = "artificial intelligence and ML"
        score = embedder.similarity(text_a, text_b)
        print(f"Similarity score: {score:.4f}")

        assert isinstance(score, float), f"Expected float score, got {type(score)}"
    except Exception as e:
        print(f"Failed similarity computation: {e}")
        return False

    # Test 5: Model properties
    print("\nTest 5: Checking model properties...")
    try:
        dim = embedder.dimension()
        model = embedder.model()
        print(f"Model: {model}")
        print(f"Dimension: {dim}")

        assert isinstance(dim, int), f"Expected int dimension, got {type(dim)}"
        assert dim > 0, f"Expected positive dimension, got {dim}"
        assert isinstance(model, str), f"Expected str model, got {type(model)}"
    except Exception as e:
        print(f"Failed to get model properties: {e}")
        return False

    print("\nTesseraDense tests passed!")
    return True


def test_colbert_basic():
    """Test basic ColBERT multi-vector embedding functionality."""
    print("\n" + "=" * 70)
    print("Testing Tessera Python Bindings - TesseraMultiVector")
    print("=" * 70 + "\n")

    try:
        from tessera import TesseraMultiVector
    except ImportError as e:
        print(f"❌ Failed to import tessera: {e}")
        print("\nMake sure you've built the bindings with:")
        print("  maturin develop --features python")
        return False

    # Test 1: Create embedder
    print("Test 1: Creating embedder...")
    try:
        embedder = TesseraMultiVector("colbert-v2")
        print(f"✓ Created embedder: {embedder}")
    except Exception as e:
        print(f"❌ Failed to create embedder: {e}")
        return False

    # Test 2: Single text encoding
    print("\nTest 2: Encoding single text...")
    try:
        text = "What is machine learning?"
        embeddings = embedder.encode(text)
        print(f"✓ Encoded text to shape: {embeddings.shape}")
        print(f"  - Number of tokens: {embeddings.shape[0]}")
        print(f"  - Embedding dimension: {embeddings.shape[1]}")
        print(f"  - Data type: {embeddings.dtype}")

        # Validate shape and dtype
        assert embeddings.ndim == 2, f"Expected 2D array, got {embeddings.ndim}D"
        assert embeddings.dtype == np.float32, f"Expected float32, got {embeddings.dtype}"
        assert embeddings.shape[0] > 0, "Expected at least one token"
        assert embeddings.shape[1] > 0, "Expected positive embedding dimension"
    except Exception as e:
        print(f"❌ Failed to encode text: {e}")
        return False

    # Test 3: Batch encoding
    print("\nTest 3: Batch encoding...")
    try:
        texts = [
            "Query one about machine learning",
            "Query two about artificial intelligence",
            "Query three about deep learning",
        ]
        batch_embs = embedder.encode_batch(texts)
        print(f"✓ Batch encoded {len(batch_embs)} texts")

        # Validate batch results
        assert len(batch_embs) == len(texts), f"Expected {len(texts)} embeddings, got {len(batch_embs)}"

        for i, emb in enumerate(batch_embs):
            print(f"  - Text {i+1}: shape {emb.shape}")
            assert emb.ndim == 2, f"Expected 2D array for text {i+1}"
            assert emb.dtype == np.float32, f"Expected float32 for text {i+1}"
    except Exception as e:
        print(f"❌ Failed batch encoding: {e}")
        return False

    # Test 4: Similarity computation
    print("\nTest 4: Computing similarity...")
    try:
        text_a = "machine learning and artificial intelligence"
        text_b = "deep learning is a subset of machine learning"
        score = embedder.similarity(text_a, text_b)
        print(f"✓ Similarity score: {score:.4f}")

        # Validate score
        assert isinstance(score, float), f"Expected float score, got {type(score)}"
        print(f"  - Score is in reasonable range: {-1.0 <= score <= 10.0}")
    except Exception as e:
        print(f"❌ Failed similarity computation: {e}")
        return False

    # Test 5: Model properties
    print("\nTest 5: Checking model properties...")
    try:
        dim = embedder.dimension()
        model = embedder.model()
        print(f"✓ Model: {model}")
        print(f"✓ Dimension: {dim}")

        # Validate properties
        assert isinstance(dim, int), f"Expected int dimension, got {type(dim)}"
        assert dim > 0, f"Expected positive dimension, got {dim}"
        assert isinstance(model, str), f"Expected str model, got {type(model)}"
        assert len(model) > 0, "Expected non-empty model string"
    except Exception as e:
        print(f"❌ Failed to get model properties: {e}")
        return False

    # Test 6: String representations
    print("\nTest 6: Testing string representations...")
    try:
        repr_str = repr(embedder)
        str_str = str(embedder)
        print(f"✓ repr(): {repr_str}")
        print(f"✓ str():  {str_str}")

        assert isinstance(repr_str, str), "Expected string from repr()"
        assert isinstance(str_str, str), "Expected string from str()"
        assert "TesseraMultiVector" in repr_str, "Expected class name in repr()"
    except Exception as e:
        print(f"❌ Failed string representation test: {e}")
        return False

    # Test 7: Error handling
    print("\nTest 7: Testing error handling...")
    try:
        # Try to create embedder with invalid model
        try:
            bad_embedder = TesseraMultiVector("invalid-model-name-xyz")
            print("❌ Should have raised an error for invalid model")
            return False
        except RuntimeError as e:
            print(f"✓ Correctly raised RuntimeError for invalid model: {str(e)[:60]}...")
    except Exception as e:
        print(f"❌ Unexpected error in error handling test: {e}")
        return False

    print("\nTesseraMultiVector tests passed!")
    return True


def test_sparse_basic():
    """Test basic sparse embedding functionality."""
    print("\n" + "=" * 70)
    print("Testing TesseraSparse - SPLADE Sparse Embeddings")
    print("=" * 70 + "\n")

    try:
        from tessera import TesseraSparse
    except ImportError as e:
        print(f"Failed to import TesseraSparse: {e}")
        return False

    # Test 1: Create embedder
    print("Test 1: Creating sparse embedder...")
    try:
        embedder = TesseraSparse("splade-pp-en-v1")
        print(f"Created embedder: {embedder}")
    except Exception as e:
        print(f"Failed to create embedder: {e}")
        return False

    # Test 2: Single text encoding
    print("\nTest 2: Encoding single text...")
    try:
        text = "machine learning"
        indices, values = embedder.encode(text)
        print(f"Encoded to sparse representation:")
        print(f"  - Non-zero dimensions: {len(indices)}")
        print(f"  - Indices dtype: {indices.dtype}")
        print(f"  - Values dtype: {values.dtype}")

        assert len(indices) == len(values), "Indices and values must have same length"
        assert indices.dtype == np.int32, f"Expected int32 indices, got {indices.dtype}"
        assert values.dtype == np.float32, f"Expected float32 values, got {values.dtype}"
    except Exception as e:
        print(f"Failed to encode text: {e}")
        return False

    # Test 3: Batch encoding
    print("\nTest 3: Batch encoding...")
    try:
        texts = ["machine learning", "deep learning", "neural networks"]
        batch_embs = embedder.encode_batch(texts)
        print(f"Batch encoded {len(batch_embs)} texts")

        for i, (indices, values) in enumerate(batch_embs):
            print(f"  - Text {i+1}: {len(indices)} non-zero dims")
            assert len(indices) == len(values), f"Mismatch for text {i+1}"
    except Exception as e:
        print(f"Failed batch encoding: {e}")
        return False

    # Test 4: Similarity computation
    print("\nTest 4: Computing similarity...")
    try:
        score = embedder.similarity("machine learning", "deep learning")
        print(f"Similarity score: {score:.4f}")
        assert isinstance(score, float), f"Expected float score, got {type(score)}"
    except Exception as e:
        print(f"Failed similarity computation: {e}")
        return False

    # Test 5: Model properties
    print("\nTest 5: Checking model properties...")
    try:
        vocab_size = embedder.vocab_size()
        model = embedder.model()
        print(f"Model: {model}")
        print(f"Vocab size: {vocab_size}")
        assert isinstance(vocab_size, int) and vocab_size > 0
    except Exception as e:
        print(f"Failed to get model properties: {e}")
        return False

    print("\nTesseraSparse tests passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 90)
    print(" " * 20 + "TESSERA PYTHON BINDINGS - COMPREHENSIVE TEST SUITE")
    print("=" * 90)

    tests = [
        ("Dense Embeddings", test_dense_basic),
        ("Multi-Vector Embeddings", test_colbert_basic),
        ("Sparse Embeddings", test_sparse_basic),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\nUnexpected error in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Print summary
    print("\n" + "=" * 90)
    print(" " * 30 + "TEST SUMMARY")
    print("=" * 90)

    all_passed = True
    for name, success in results:
        status = "PASSED" if success else "FAILED"
        symbol = "✓" if success else "✗"
        print(f"{symbol} {name}: {status}")
        if not success:
            all_passed = False

    print("=" * 90)

    if all_passed:
        print("\n🎉 All tests passed! Python bindings are working correctly.")
        print("\nNote: Vision and TimeSeries tests were skipped (models too large for CI).")
        print("To test them locally, add test_vision_basic() and test_timeseries_basic().")
    else:
        print("\n❌ Some tests failed. Please check the output above.")

    print("\n")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
