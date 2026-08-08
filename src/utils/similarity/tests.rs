use super::*;
use ndarray::{array, Array2, Axis};

fn embeddings(num_tokens: usize, embedding_dim: usize, seed: usize) -> TokenEmbeddings {
    let values = Array2::from_shape_fn((num_tokens, embedding_dim), |(token, dimension)| {
        let value = (token * 37 + dimension * 17 + seed * 13) % 101;
        (value as f32 - 50.0) / 17.0
    });

    TokenEmbeddings::from_parts_unchecked(values, String::new(), num_tokens, embedding_dim)
}

fn full_matrix_max_sim(query: &TokenEmbeddings, document: &TokenEmbeddings) -> f32 {
    let similarity_matrix = query.matrix().dot(&document.matrix().t());
    let max_sims = similarity_matrix.map_axis(Axis(1), |row| {
        row.fold(f32::NEG_INFINITY, |acc, &score| acc.max(score))
    });
    max_sims.sum()
}

fn assert_scores_match(actual: f32, expected: f32, shape: (usize, usize, usize)) {
    let tolerance = 1e-5 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "score mismatch for shape {shape:?}: tiled={actual}, full={expected}, tolerance={tolerance}"
    );
}

#[test]
fn test_cosine_similarity_identical() {
    let a = array![1.0, 2.0, 3.0];
    let b = array![1.0, 2.0, 3.0];
    let sim = cosine_similarity(&a, &b).unwrap();
    assert!((sim - 1.0).abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let a = array![1.0, 0.0, 0.0];
    let b = array![0.0, 1.0, 0.0];
    let sim = cosine_similarity(&a, &b).unwrap();
    assert!((sim - 0.0).abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_opposite() {
    let a = array![1.0, 0.0];
    let b = array![-1.0, 0.0];
    let sim = cosine_similarity(&a, &b).unwrap();
    assert!((sim - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_dimension_mismatch() {
    let a = array![1.0, 2.0];
    let b = array![1.0, 2.0, 3.0];
    assert!(cosine_similarity(&a, &b).is_err());
}

#[test]
fn vector_metrics_reject_empty_and_non_finite_inputs() {
    let empty = Array1::zeros(0);
    let finite = array![1.0];
    let nan = array![f32::NAN];

    assert!(cosine_similarity(&empty, &empty).is_err());
    assert!(dot_product(&nan, &finite).is_err());
    assert!(euclidean_distance(&finite, &nan).is_err());
}

#[test]
fn vector_metrics_reject_non_finite_results() {
    let huge = array![f32::MAX, f32::MAX];

    assert!(dot_product(&huge, &huge).is_err());
    assert!(euclidean_distance(&huge, &array![-f32::MAX, -f32::MAX]).is_err());
}

#[test]
fn test_dot_product() {
    let a = array![1.0, 2.0, 3.0];
    let b = array![4.0, 5.0, 6.0];
    let dot = dot_product(&a, &b).unwrap();
    // 1*4 + 2*5 + 3*6 = 32
    assert!((dot - 32.0).abs() < 1e-6);
}

#[test]
fn test_dot_product_dimension_mismatch() {
    let a = array![1.0, 2.0];
    let b = array![1.0, 2.0, 3.0];
    assert!(dot_product(&a, &b).is_err());
}

#[test]
fn test_euclidean_distance() {
    let a = array![0.0, 0.0];
    let b = array![3.0, 4.0];
    let dist = euclidean_distance(&a, &b).unwrap();
    // √(3² + 4²) = 5
    assert!((dist - 5.0).abs() < 1e-6);
}

#[test]
fn test_euclidean_distance_identical() {
    let a = array![1.0, 2.0, 3.0];
    let b = array![1.0, 2.0, 3.0];
    let dist = euclidean_distance(&a, &b).unwrap();
    assert!((dist - 0.0).abs() < 1e-6);
}

#[test]
fn test_euclidean_distance_dimension_mismatch() {
    let a = array![1.0, 2.0];
    let b = array![1.0, 2.0, 3.0];
    assert!(euclidean_distance(&a, &b).is_err());
}

#[test]
fn test_max_sim_simple() {
    // Create simple query embeddings (2 tokens, 3 dimensions each)
    let query_emb = array![
        [1.0, 0.0, 0.0], // Token 1
        [0.0, 1.0, 0.0], // Token 2
    ];
    let query = TokenEmbeddings::new(query_emb, "query text".to_string()).unwrap();

    // Create simple document embeddings (3 tokens, 3 dimensions each)
    let doc_emb = array![
        [1.0, 0.0, 0.0], // Token 1 (matches query token 1)
        [0.0, 0.5, 0.0], // Token 2 (partial match to query token 2)
        [0.0, 1.0, 0.0], // Token 3 (matches query token 2)
    ];
    let document = TokenEmbeddings::new(doc_emb, "document text".to_string()).unwrap();

    let score = max_sim(&query, &document).unwrap();

    // Query token 1 max similarity: max(1.0, 0.0, 0.0) = 1.0
    // Query token 2 max similarity: max(0.0, 0.5, 1.0) = 1.0
    // Total: 1.0 + 1.0 = 2.0
    assert!((score - 2.0).abs() < 1e-6);
}

#[test]
fn test_max_sim_tiled_matches_full_reference_across_shapes() {
    let block = MAX_SIM_DOCUMENT_BLOCK_SIZE;
    let shapes = [
        (1, 1, 1),
        (3, 1, 5),
        (1, 3, 5),
        (7, block - 1, 11),
        (7, block, 11),
        (7, block + 1, 11),
        (3, 2 * block - 1, 13),
        (3, 2 * block, 13),
        (3, 2 * block + 1, 13),
    ];

    for (case, shape @ (query_tokens, document_tokens, dimension)) in shapes.into_iter().enumerate()
    {
        let query = embeddings(query_tokens, dimension, case + 1);
        let document = embeddings(document_tokens, dimension, case + 17);
        let expected = full_matrix_max_sim(&query, &document);
        let actual = max_sim(&query, &document).unwrap();
        assert_scores_match(actual, expected, shape);
    }
}

#[test]
fn test_max_sim_rejects_empty_inputs() {
    let shapes = [(0, 3, 5), (3, 0, 5), (0, 0, 5)];

    for (case, (query_tokens, document_tokens, dimension)) in shapes.into_iter().enumerate() {
        let query = embeddings(query_tokens, dimension, case + 1);
        let document = embeddings(document_tokens, dimension, case + 7);
        assert!(max_sim(&query, &document).is_err());
    }
}

#[test]
fn test_max_sim_rejects_malformed_metadata_and_values() {
    let valid = embeddings(1, 2, 1);
    let malformed_tokens =
        TokenEmbeddings::from_parts_unchecked(valid.matrix().clone(), String::new(), 2, 2);
    assert!(max_sim(&malformed_tokens, &valid).is_err());

    let malformed_dim =
        TokenEmbeddings::from_parts_unchecked(valid.matrix().clone(), String::new(), 1, 3);
    assert!(max_sim(&valid, &malformed_dim).is_err());

    let non_finite =
        TokenEmbeddings::from_parts_unchecked(array![[f32::NAN, 1.0]], String::new(), 1, 2);
    assert!(max_sim(&non_finite, &valid).is_err());
}

#[test]
fn test_max_sim_rejects_non_finite_result() {
    let query = TokenEmbeddings::new(array![[f32::MAX]], String::new()).unwrap();
    let document = TokenEmbeddings::new(array![[f32::MAX]], String::new()).unwrap();

    assert!(max_sim(&query, &document).is_err());
}

#[test]
fn test_max_sim_dimension_mismatch() {
    let query_emb = array![[1.0, 0.0]];
    let query = TokenEmbeddings::new(query_emb, "query".to_string()).unwrap();

    let doc_emb = array![[1.0, 0.0, 0.0]];
    let document = TokenEmbeddings::new(doc_emb, "document".to_string()).unwrap();

    let error = max_sim(&query, &document).unwrap_err();
    assert_eq!(
        error.to_string(),
        "Query and document embedding dimensions must match (query: 2, document: 3)"
    );
}
