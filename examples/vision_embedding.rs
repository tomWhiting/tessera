//! Example demonstrating VisionEmbedding type usage
//!
//! Shows how to create and work with vision embeddings for ColPali-style
//! vision-language models.

use tessera::core::VisionEmbedding;

fn main() -> anyhow::Result<()> {
    // Create a simple vision embedding with 1024 patches (32x32 grid)
    // for a typical ColPali 448x448 image
    let num_patches = 1024;
    let embedding_dim = 128;

    // Create mock embeddings (in real usage, these would come from a model)
    let embeddings: Vec<Vec<f32>> = (0..num_patches)
        .map(|i| vec![(i as f32).sin(); embedding_dim])
        .collect();

    let vision_emb = VisionEmbedding::new(
        embeddings,
        num_patches,
        embedding_dim,
        Some("document_page_1.pdf".to_string()),
    );

    // Test accessors
    println!("VisionEmbedding created successfully!");
    println!("  num_patches: {}", vision_emb.num_patches());
    println!("  embedding_dim: {}", vision_emb.embedding_dim());
    println!("  shape: {:?}", vision_emb.shape());
    println!("  source: {:?}", vision_emb.source());

    // Show that clone works
    let cloned = vision_emb.clone();
    println!("\nCloned embedding also works:");
    println!("  cloned shape: {:?}", cloned.shape());
    println!("  cloned source: {:?}", cloned.source());

    // Test without source
    let vision_emb_no_source = VisionEmbedding::new(
        (0..256).map(|i| vec![(i as f32).cos(); 128]).collect(),
        256,
        128,
        None,
    );
    println!("\nVisionEmbedding without source:");
    println!("  shape: {:?}", vision_emb_no_source.shape());
    println!("  source: {:?}", vision_emb_no_source.source());

    Ok(())
}
