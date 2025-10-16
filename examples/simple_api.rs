//! Simple API demonstration
//!
//! Shows the ergonomic one-line API for encoding text with Tessera.

use tessera::{Result, Tessera};

fn main() -> Result<()> {
    println!("=== Tessera Simple API Demo ===\n");

    // Simple API - one line initialization
    println!("1. Simple API: Tessera::new()");
    let embedder = Tessera::new("colbert-v2")?;
    println!("   Created embedder for model: {}", embedder.model());
    println!("   Embedding dimension: {}\n", embedder.dimension());

    // Encode single text
    println!("2. Encoding single text");
    let text = "What is machine learning?";
    let embeddings = embedder.encode(text)?;
    println!("   Text: '{}'", text);
    println!("   Encoded to {} token vectors of {} dimensions\n", 
        embeddings.num_tokens, 
        embeddings.embedding_dim
    );

    // Encode batch
    println!("3. Batch encoding");
    let texts = [
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "Natural language processing analyzes text",
    ];
    let batch_embeddings = embedder.encode_batch(&texts)?;
    println!("   Encoded {} texts:", batch_embeddings.len());
    for (i, emb) in batch_embeddings.iter().enumerate() {
        println!("     [{}] {} tokens", i, emb.num_tokens);
    }
    println!();

    // Compute similarity
    println!("4. Computing similarity");
    let query = "What is machine learning?";
    let doc1 = "Machine learning is a subset of artificial intelligence";
    let doc2 = "Pizza is a type of Italian food";
    
    let sim1 = embedder.similarity(query, doc1)?;
    let sim2 = embedder.similarity(query, doc2)?;
    
    println!("   Query: '{}'", query);
    println!("   Doc 1: '{}' - Similarity: {:.4}", doc1, sim1);
    println!("   Doc 2: '{}' - Similarity: {:.4}", doc2, sim2);
    println!("   Relevance ranking: {} > {}\n", 
        if sim1 > sim2 { "Doc 1" } else { "Doc 2" },
        if sim1 > sim2 { "Doc 2" } else { "Doc 1" }
    );

    // Builder API
    println!("5. Builder API for advanced configuration");
    let embedder2 = Tessera::builder()
        .model("colbert-v2")
        .build()?;
    println!("   Created with builder pattern");
    println!("   Model: {}, Dimension: {}\n", embedder2.model(), embedder2.dimension());

    println!("=== Demo Complete ===");
    Ok(())
}
