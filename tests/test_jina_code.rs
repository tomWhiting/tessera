//! Test for Jina code embedding models

use std::error::Error as StdError;
use tessera::TesseraDense;

/// Helper to print full error chain
fn print_error_chain<E: StdError>(e: &E) {
    eprintln!("\n=== ERROR CHAIN ===");
    let mut source: Option<&dyn StdError> = e.source();
    let mut level = 0;
    eprintln!("Level {}: {}", level, e);
    while let Some(s) = source {
        level += 1;
        eprintln!("Level {}: {}", level, s);
        source = s.source();
    }
    eprintln!("=== END ERROR CHAIN ===\n");
}

#[test]
#[ignore] // Requires downloading model - run with --ignored
fn test_jina_code_v2_base() {
    println!("Loading jina-embeddings-v2-base-code (JinaBERT Code variant)...");

    match TesseraDense::new("jina-embeddings-v2-base-code") {
        Ok(embedder) => {
            println!("Model loaded successfully!");

            // Try encoding some code
            let code = r#"fn main() { println!("Hello, world!"); }"#;
            match embedder.encode(code) {
                Ok(emb) => {
                    println!("Encoded! Dimension: {}", emb.dim());
                    assert_eq!(emb.dim(), 768, "Expected 768 dimensions");
                }
                Err(e) => panic!("Failed to encode: {}", e),
            }
        }
        Err(e) => {
            print_error_chain(&e);
            panic!("Failed to load model");
        }
    }
}

#[test]
#[ignore] // Requires downloading model - run with --ignored
fn test_jina_code_embeddings_05b() {
    println!("Loading jina-code-embeddings-0.5b (Qwen2 based)...");

    match TesseraDense::new("jina-code-embeddings-0.5b") {
        Ok(embedder) => {
            println!("Model loaded successfully!");

            // Try encoding some code
            let code = r#"fn main() { println!("Hello, world!"); }"#;
            match embedder.encode(code) {
                Ok(emb) => {
                    println!("Encoded! Dimension: {}", emb.dim());
                    assert_eq!(emb.dim(), 896, "Expected 896 dimensions for 0.5B model");
                }
                Err(e) => panic!("Failed to encode: {}", e),
            }
        }
        Err(e) => {
            print_error_chain(&e);
            panic!("Failed to load model");
        }
    }
}

#[test]
#[ignore] // Requires downloading model - run with --ignored
fn test_jina_code_embeddings_15b() {
    println!("Loading jina-code-embeddings-1.5b (Qwen2 based)...");

    match TesseraDense::new("jina-code-embeddings-1.5b") {
        Ok(embedder) => {
            println!("Model loaded successfully!");

            // Try encoding some code
            let code = r#"fn main() { println!("Hello, world!"); }"#;
            match embedder.encode(code) {
                Ok(emb) => {
                    println!("Encoded! Dimension: {}", emb.dim());
                    assert_eq!(emb.dim(), 1536, "Expected 1536 dimensions for 1.5B model");
                }
                Err(e) => panic!("Failed to encode: {}", e),
            }
        }
        Err(e) => {
            print_error_chain(&e);
            panic!("Failed to load model");
        }
    }
}
