//! Builder API demonstration
//!
//! Shows advanced configuration using the builder pattern.

use candle_core::Device;
use tessera::{Result, TesseraMultiVector};

fn main() -> Result<()> {
    println!("=== Tessera Builder API Demo ===\n");

    // Example 1: Simple builder usage
    println!("1. Basic builder - model only (auto-select device)");
    let embedder1 = TesseraMultiVector::builder()
        .model("colbert-v2")
        .build()?;
    println!("   Model: {}", embedder1.model());
    println!("   Dimension: {}\n", embedder1.dimension());

    // Example 2: Specify device explicitly
    println!("2. Builder with explicit CPU device");
    let embedder2 = TesseraMultiVector::builder()
        .model("colbert-v2")
        .device(Device::Cpu)
        .build()?;
    println!("   Model: {}", embedder2.model());
    println!("   Dimension: {}\n", embedder2.dimension());

    // Example 3: Matryoshka dimensions (if supported)
    println!("3. Builder with Matryoshka dimension");
    println!("   Note: jina-colbert-v2 supports Matryoshka dimensions");
    println!("   Supported dimensions: [64, 96, 128, 192, 256, 384, 512, 768]");

    let embedder3 = TesseraMultiVector::builder()
        .model("jina-colbert-v2")
        .dimension(128)  // Use 128 instead of default 768
        .build()?;
    println!("   Model: {}", embedder3.model());
    println!("   Dimension: {} (truncated from 768)\n", embedder3.dimension());

    // Encode with the Matryoshka model
    let text = "Testing Matryoshka dimension truncation";
    let emb = embedder3.encode(text)?;
    println!("   Encoded '{}' to {} tokens x {} dims\n", 
        text, emb.num_tokens, emb.embedding_dim);

    // Example 4: Error handling - invalid model
    println!("4. Error handling - model not found");
    match TesseraMultiVector::builder().model("nonexistent-model").build() {
        Ok(_) => println!("   Unexpected success!"),
        Err(e) => println!("   Expected error: {}\n", e),
    }

    // Example 5: Error handling - unsupported dimension
    println!("5. Error handling - unsupported dimension");
    match TesseraMultiVector::builder()
        .model("colbert-v2")  // Fixed dimension: 128
        .dimension(256)       // Not supported!
        .build()
    {
        Ok(_) => println!("   Unexpected success!"),
        Err(e) => println!("   Expected error: {}\n", e),
    }

    // Example 6: Comparison - simple vs builder API
    println!("6. API comparison");

    // Simple API
    let simple = TesseraMultiVector::new("colbert-v2")?;
    println!("   Simple: TesseraMultiVector::new(\"colbert-v2\")");
    println!("   -> Model: {}, Dim: {}", simple.model(), simple.dimension());

    // Builder API (equivalent)
    let builder = TesseraMultiVector::builder()
        .model("colbert-v2")
        .build()?;
    println!("   Builder: TesseraMultiVector::builder().model(\"colbert-v2\").build()");
    println!("   -> Model: {}, Dim: {}\n", builder.model(), builder.dimension());

    println!("=== Demo Complete ===");
    Ok(())
}
