//! Demonstrate the model registry system.
//!
//! Shows how to:
//! - Access model metadata from the registry
//! - Query models by various criteria
//! - Use the registry to create model configurations

use tessera::model_registry::{
    get_model, models_by_language, models_by_max_embedding_dim, models_by_organization,
    models_by_type, models_with_matryoshka, ModelType, COLBERT_SMALL, COLBERT_V2,
    JINA_COLBERT_V2, MODEL_REGISTRY,
};
use tessera::ModelConfig;

fn main() {
    println!("Model Registry Demo");
    println!("==================\n");

    // Access constants directly
    println!("1. Direct constant access:");
    println!("   {}: {} dims, {}K context, {}",
        COLBERT_V2.name,
        COLBERT_V2.embedding_dim,
        COLBERT_V2.context_length / 1000,
        COLBERT_V2.organization
    );
    println!("   {}: {} dims, {}K context, {}",
        COLBERT_SMALL.name,
        COLBERT_SMALL.embedding_dim,
        COLBERT_SMALL.context_length / 1000,
        COLBERT_SMALL.organization
    );
    println!("   {}: {} dims, {}K context, {} languages",
        JINA_COLBERT_V2.name,
        JINA_COLBERT_V2.embedding_dim,
        JINA_COLBERT_V2.context_length / 1000,
        JINA_COLBERT_V2.languages.len()
    );

    // List all models
    println!("\n2. All available models ({} total):", MODEL_REGISTRY.len());
    for model in MODEL_REGISTRY {
        println!("   - {} ({}): {} dims, {} params, {}",
            model.name,
            model.id,
            model.embedding_dim,
            model.parameters,
            model.license
        );
    }

    // Get model by ID
    println!("\n3. Get model by ID:");
    if let Some(model) = get_model("colbert-small") {
        println!("   Found: {}", model.name);
        println!("   HuggingFace: {}", model.huggingface_id);
        println!("   Description: {}", model.description);
    }

    // Query by type
    println!("\n4. Query by type (ColBERT models):");
    let colbert_models = models_by_type(ModelType::Colbert);
    for model in colbert_models {
        println!("   - {}: {} dims", model.name, model.embedding_dim);
    }

    // Query by organization
    println!("\n5. Query by organization (Jina AI):");
    let jina_models = models_by_organization("Jina AI");
    for model in jina_models {
        println!("   - {}: {} dims, {} languages",
            model.name,
            model.embedding_dim,
            model.languages.len()
        );
    }

    // Query by language
    println!("\n6. Query by language (English):");
    let english_models = models_by_language("en");
    println!("   {} models support English", english_models.len());

    // Query by embedding dimension
    println!("\n7. Query by max embedding dimension (<=128):");
    let compact_models = models_by_max_embedding_dim(128);
    for model in compact_models {
        println!("   - {}: {} dims", model.name, model.embedding_dim);
    }

    // Query Matryoshka models
    println!("\n8. Matryoshka-enabled models:");
    let matryoshka_models = models_with_matryoshka();
    for model in matryoshka_models {
        let dims = model.embedding_dim.supported_dimensions();
        print!("   - {}: [", model.name);
        for (i, dim) in dims.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{}", dim);
        }
        println!("] dims");
    }

    // Create config from registry
    println!("\n9. Create ModelConfig from registry:");
    match ModelConfig::from_registry("jina-colbert-v2") {
        Ok(config) => {
            println!("   Model: {}", config.model_name);
            println!("   Embedding dim: {}", config.embedding_dim);
            println!("   Max seq length: {}", config.max_seq_length);
        }
        Err(e) => println!("   Error: {}", e),
    }

    // Model comparison
    println!("\n10. Model comparison:");
    println!("    Model                           Dims   Context  Params   BEIR   MRR@10");
    println!("    ----                            ----   -------  ------   ----   ------");
    for model in MODEL_REGISTRY {
        println!("    {:30} {:>5}  {:>7}  {:>6}   {:.2}   {:.2}",
            model.name,
            model.embedding_dim,
            format!("{}K", model.context_length / 1000),
            model.parameters,
            model.beir_avg,
            model.ms_marco_mrr10
        );
    }

    println!("\n11. Architecture details:");
    for model in MODEL_REGISTRY {
        let is_matryoshka = matches!(model.embedding_dim, tessera::model_registry::EmbeddingDimension::Matryoshka { .. });
        println!("   {}: {}-{}, projection={}, matryoshka={}",
            model.name,
            model.architecture_type,
            model.architecture_variant,
            model.has_projection,
            is_matryoshka
        );
    }

    println!("\nRegistry demo complete!");
}
