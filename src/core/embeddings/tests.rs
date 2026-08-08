use super::*;
use ndarray::{array, Array1, Array2};

impl TokenEmbeddings {
    /// Construct intentionally inconsistent metadata for validation tests.
    pub const fn from_parts_unchecked(
        embeddings: Array2<f32>,
        text: String,
        num_tokens: usize,
        embedding_dim: usize,
    ) -> Self {
        Self {
            embeddings,
            text,
            num_tokens,
            embedding_dim,
        }
    }
}

#[test]
fn token_embeddings_require_nonempty_finite_matrix() {
    assert!(TokenEmbeddings::new(Array2::zeros((0, 4)), String::new()).is_err());
    assert!(TokenEmbeddings::new(Array2::zeros((2, 0)), String::new()).is_err());
    assert!(TokenEmbeddings::new(array![[f32::NAN]], String::new()).is_err());

    let embedding = TokenEmbeddings::new(array![[1.0, 2.0]], "text".to_string()).unwrap();
    assert_eq!(embedding.shape(), (1, 2));
    assert_eq!(embedding.num_tokens(), 1);
    assert_eq!(embedding.embedding_dim(), 2);
    assert_eq!(embedding.text(), "text");
    assert_eq!(embedding.matrix(), &array![[1.0, 2.0]]);
    assert_eq!(embedding.into_matrix(), array![[1.0, 2.0]]);
}

#[test]
fn dense_embeddings_require_nonempty_finite_vector() {
    assert!(DenseEmbedding::new(Array1::zeros(0), String::new()).is_err());
    assert!(DenseEmbedding::new(array![f32::INFINITY], String::new()).is_err());

    let embedding = DenseEmbedding::new(array![1.0, 2.0], "text".to_string()).unwrap();
    assert_eq!(embedding.dim(), 2);
    assert_eq!(embedding.text(), "text");
    assert_eq!(embedding.values(), &array![1.0, 2.0]);
    assert_eq!(embedding.into_values(), array![1.0, 2.0]);
}

#[test]
fn sparse_embeddings_require_valid_canonical_entries() {
    assert!(SparseEmbedding::new(Vec::new(), 0, String::new()).is_err());
    assert!(SparseEmbedding::new(vec![(4, 1.0)], 4, String::new()).is_err());
    assert!(SparseEmbedding::new(vec![(1, f32::NAN)], 4, String::new()).is_err());
    assert!(SparseEmbedding::new(vec![(1, 0.0)], 4, String::new()).is_err());
    assert!(SparseEmbedding::new(vec![(2, 1.0), (1, 2.0)], 4, String::new()).is_err());
    assert!(SparseEmbedding::new(vec![(1, 1.0), (1, 2.0)], 4, String::new()).is_err());

    let embedding = SparseEmbedding::new(vec![(1, 1.0), (3, 2.0)], 4, "text".to_string()).unwrap();
    assert_eq!(embedding.nnz(), 2);
    assert_eq!(embedding.entries(), &[(1, 1.0), (3, 2.0)]);
    assert_eq!(embedding.vocab_size(), 4);
    assert_eq!(embedding.text(), "text");
    assert!((embedding.sparsity() - 0.5).abs() < f32::EPSILON);
}
