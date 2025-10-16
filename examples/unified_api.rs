//! Unified API demonstration showing auto-detection of model types.
//!
//! Demonstrates the smart factory pattern that automatically creates
//! the appropriate embedder variant based on model type.

use tessera::{Result, Tessera};

fn main() -> Result<()> {
    println!("=== Tessera Unified API Demo ===\n");

    // Example 1: Auto-detect ColBERT model (creates MultiVector variant)
    println!("1. Auto-detecting ColBERT model");
    let colbert = Tessera::new("colbert-v2")?;
    println!("   Model: colbert-v2");
    
    match &colbert {
        Tessera::MultiVector(mv) => {
            println!("   ✓ Detected as MultiVector");
            println!("   Model: {}, Dimension: {}", mv.model(), mv.dimension());

            let embeddings = mv.encode("What is machine learning?")?;
            println!("   Encoded to {} token vectors\n", embeddings.num_tokens);
        }
        Tessera::Dense(_) => {
            println!("   ✗ Unexpected: Detected as Dense\n");
        }
        Tessera::Sparse(_) => {
            println!("   ✗ Unexpected: Detected as Sparse\n");
        }
    }

    // Example 2: Show error for unsupported model types
    println!("2. Error handling for unsupported types");
    match Tessera::new("splade-v3") {
        Ok(_) => println!("   Unexpected success!\n"),
        Err(e) => println!("   Expected error: {}\n", e),
    }

    // Example 3: Pattern matching to handle both types
    println!("3. Pattern matching for type-specific operations");
    let models = vec!["colbert-v2"];
    
    for model_id in models {
        let embedder = Tessera::new(model_id)?;
        
        match embedder {
            Tessera::MultiVector(mv) => {
                println!("   MultiVector model: {}", mv.model());
                println!("     - Supports quantization: Yes");
                println!("     - Output: Token-level embeddings");
                println!("     - Similarity: MaxSim");
            }
            Tessera::Dense(d) => {
                println!("   Dense model: {}", d.model());
                println!("     - Supports quantization: No");
                println!("     - Output: Single pooled vector");
                println!("     - Similarity: Cosine");
            }
            Tessera::Sparse(s) => {
                println!("   Sparse model: {}", s.model());
                println!("     - Supports quantization: No");
                println!("     - Output: Sparse vocabulary vector");
                println!("     - Similarity: Dot product");
            }
        }
    }
    println!();

    // Example 4: Direct type usage when you know the model type
    println!("4. Direct type usage (when model type is known)");
    
    use tessera::TesseraMultiVector;
    let mv_embedder = TesseraMultiVector::new("colbert-v2")?;
    println!("   TesseraMultiVector: {} dims", mv_embedder.dimension());
    
    println!();

    println!("=== Demo Complete ===");
    Ok(())
}
