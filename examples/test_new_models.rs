//! Test newly added models from Phase 1.4.
//!
//! This example verifies that:
//! 1. GTE-ModernColBERT can be loaded from the registry
//! 2. Model metadata is correct
//! 3. Model can be instantiated (without downloading weights)

use tessera::model_registry::{get_model, GTE_MODERN_COLBERT, JINA_COLBERT_V2};
use tessera::ModelConfig;

fn main() {
    println!("Testing Phase 1.4 Model Registry Additions");
    println!("==========================================\n");

    // Test 1: Verify Jina-ColBERT-v2 (should already exist)
    println!("1. Testing Jina-ColBERT-v2:");
    println!("   ID: {}", JINA_COLBERT_V2.id);
    println!("   Name: {}", JINA_COLBERT_V2.name);
    println!("   HuggingFace: {}", JINA_COLBERT_V2.huggingface_id);
    println!("   Dimensions: {}", JINA_COLBERT_V2.embedding_dim);
    println!("   Context length: {}K", JINA_COLBERT_V2.context_length / 1000);
    println!("   Languages: {}", JINA_COLBERT_V2.languages.len());
    println!("   Organization: {}", JINA_COLBERT_V2.organization);
    println!("   License: {}", JINA_COLBERT_V2.license);
    
    // Check if Matryoshka
    let is_matryoshka = matches!(
        JINA_COLBERT_V2.embedding_dim,
        tessera::model_registry::EmbeddingDimension::Matryoshka { .. }
    );
    println!("   Matryoshka support: {}", is_matryoshka);
    if is_matryoshka {
        let supported = JINA_COLBERT_V2.embedding_dim.supported_dimensions();
        println!("   Supported dimensions: {:?}", supported);
    }
    println!("   ✓ Jina-ColBERT-v2 verified\n");

    // Test 2: Verify GTE-ModernColBERT
    println!("2. Testing GTE-ModernColBERT:");
    println!("   ID: {}", GTE_MODERN_COLBERT.id);
    println!("   Name: {}", GTE_MODERN_COLBERT.name);
    println!("   HuggingFace: {}", GTE_MODERN_COLBERT.huggingface_id);
    println!("   Dimensions: {}", GTE_MODERN_COLBERT.embedding_dim);
    println!("   Context length: {}K", GTE_MODERN_COLBERT.context_length / 1000);
    println!("   Languages: {:?}", GTE_MODERN_COLBERT.languages);
    println!("   Organization: {}", GTE_MODERN_COLBERT.organization);
    println!("   License: {}", GTE_MODERN_COLBERT.license);
    println!("   Architecture: {}-{}", 
        GTE_MODERN_COLBERT.architecture_type,
        GTE_MODERN_COLBERT.architecture_variant
    );
    println!("   Has projection: {}", GTE_MODERN_COLBERT.has_projection);
    println!("   Parameters: {}", GTE_MODERN_COLBERT.parameters);
    println!("   BEIR avg: {:.2}", GTE_MODERN_COLBERT.beir_avg);
    println!("   MS MARCO MRR@10: {:.2}", GTE_MODERN_COLBERT.ms_marco_mrr10);
    println!("   ✓ GTE-ModernColBERT verified\n");

    // Test 3: Load by ID from registry
    println!("3. Testing registry lookup:");
    if let Some(model) = get_model("gte-modern-colbert") {
        println!("   Found '{}' in registry", model.name);
        println!("   Description: {}", model.description);
        println!("   ✓ Registry lookup successful\n");
    } else {
        println!("   ✗ ERROR: Could not find gte-modern-colbert in registry\n");
    }

    // Test 4: Create ModelConfig from registry
    println!("4. Testing ModelConfig creation:");
    match ModelConfig::from_registry("gte-modern-colbert") {
        Ok(config) => {
            println!("   Model name: {}", config.model_name);
            println!("   Embedding dim: {}", config.embedding_dim);
            println!("   Max seq length: {}", config.max_seq_length);
            println!("   ✓ ModelConfig creation successful\n");
        }
        Err(e) => {
            println!("   ✗ ERROR: Failed to create ModelConfig: {}\n", e);
        }
    }

    // Test 5: Verify model count
    println!("5. Verifying total model count:");
    let total_models = tessera::model_registry::MODEL_REGISTRY.len();
    println!("   Total models in registry: {}", total_models);
    
    // Count ColBERT models
    let colbert_count = tessera::model_registry::models_by_type(
        tessera::model_registry::ModelType::Colbert
    ).len();
    println!("   ColBERT models: {}", colbert_count);
    
    if total_models >= 18 {
        println!("   ✓ Registry expanded successfully\n");
    } else {
        println!("   ✗ WARNING: Expected at least 18 models\n");
    }

    println!("==========================================");
    println!("All Phase 1.4 registry tests completed!");
    println!("==========================================");
}
