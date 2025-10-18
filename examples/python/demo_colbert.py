#!/usr/bin/env python3
"""Quick demo of Tessera Python bindings with ColBERT.

Shows basic usage: encoding, batch processing, and similarity search.
"""

import numpy as np


def main():
    print("\n" + "=" * 70)
    print("Tessera Python Bindings - ColBERT Demo")
    print("=" * 70 + "\n")

    # Import the bindings
    from tessera import TesseraMultiVector

    # Create ColBERT embedder
    print("📦 Loading ColBERT model...")
    embedder = TesseraMultiVector("colbert-v2")
    print(f"✓ Loaded: {embedder}\n")

    # Single text encoding
    print("🔤 Single text encoding:")
    text = "What is machine learning?"
    embeddings = embedder.encode(text)
    print(f"   Input: '{text}'")
    print(f"   Output shape: {embeddings.shape} (tokens × dimensions)")
    print(f"   Data type: {embeddings.dtype}\n")

    # Batch encoding
    print("📚 Batch encoding:")
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with many layers",
        "Natural language processing enables computers to understand text",
    ]
    batch_embeddings = embedder.encode_batch(documents)
    print(f"   Encoded {len(batch_embeddings)} documents")
    for i, emb in enumerate(batch_embeddings, 1):
        print(f"   Doc {i}: {emb.shape[0]} tokens × {emb.shape[1]} dims")
    print()

    # Semantic search with similarity scores
    print("🔍 Semantic similarity search:")
    query = "What is AI and machine learning?"
    print(f"   Query: '{query}'\n")

    # Compute similarities
    scores = []
    for doc in documents:
        score = embedder.similarity(query, doc)
        scores.append((doc, score))

    # Sort by score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)

    # Display results
    print("   Top matches:")
    for i, (doc, score) in enumerate(scores, 1):
        print(f"   {i}. Score: {score:.4f}")
        print(f"      \"{doc}\"\n")

    # Model properties
    print("ℹ️  Model info:")
    print(f"   Model ID: {embedder.model()}")
    print(f"   Embedding dimension: {embedder.dimension()}")

    print("\n" + "=" * 70)
    print("✨ Demo complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
