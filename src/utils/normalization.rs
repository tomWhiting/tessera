//! Vector normalization utilities.
//!
//! Provides normalization functions for embedding vectors:
//!
//! - L2 normalization: Scale vector to unit length
//! - L2 norm computation: Calculate vector magnitude
//!
//! Normalization is important for many similarity metrics (especially cosine
//! similarity) and can improve numerical stability in deep learning models.

use ndarray::Array1;

/// Compute L2 norm (Euclidean length) of a vector.
///
/// The L2 norm is the square root of the sum of squared elements,
/// representing the vector's magnitude or length.
///
/// Formula: `||v||₂ = √(Σvᵢ²)`
///
/// # Arguments
/// * `vec` - Input vector
///
/// # Returns
/// L2 norm (non-negative scalar)
///
/// # Example
/// ```
/// use ndarray::array;
/// use tessera::utils::l2_norm;
///
/// let v = array![3.0, 4.0];
/// let norm = l2_norm(&v);
/// assert!((norm - 5.0).abs() < 1e-6);  // √(3² + 4²) = 5
/// ```
#[must_use] pub fn l2_norm(vec: &Array1<f32>) -> f32 {
    vec.dot(vec).sqrt()
}

/// L2 normalize a vector (scale to unit length).
///
/// Divides the vector by its L2 norm, producing a unit vector that points
/// in the same direction but has length 1. This is essential for cosine
/// similarity and many other distance metrics.
///
/// Formula: `v_normalized = v / ||v||₂`
///
/// # Arguments
/// * `vec` - Input vector
///
/// # Returns
/// Normalized vector (unit length)
///
/// # Note
/// If the input vector has zero norm, returns the original vector unchanged
/// to avoid division by zero.
///
/// # Example
/// ```
/// use ndarray::array;
/// use tessera::utils::{l2_normalize, l2_norm};
///
/// let v = array![3.0, 4.0];
/// let normalized = l2_normalize(&v);
///
/// // Verify unit length
/// let norm = l2_norm(&normalized);
/// assert!((norm - 1.0).abs() < 1e-6);
///
/// // Check direction preserved
/// assert!((normalized[0] - 0.6).abs() < 1e-6);  // 3/5
/// assert!((normalized[1] - 0.8).abs() < 1e-6);  // 4/5
/// ```
#[must_use] pub fn l2_normalize(vec: &Array1<f32>) -> Array1<f32> {
    let norm = l2_norm(vec);
    if norm > 0.0 {
        vec / norm
    } else {
        vec.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_l2_norm() {
        let v = array![3.0, 4.0];
        let norm = l2_norm(&v);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_norm_unit_vector() {
        let v = array![1.0, 0.0, 0.0];
        let norm = l2_norm(&v);
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_norm_zero_vector() {
        let v = array![0.0, 0.0, 0.0];
        let norm = l2_norm(&v);
        assert!((norm - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize() {
        let v = array![3.0, 4.0];
        let normalized = l2_normalize(&v);

        // Check unit length
        let norm = l2_norm(&normalized);
        assert!((norm - 1.0).abs() < 1e-6);

        // Check direction
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_already_normalized() {
        let v = array![1.0, 0.0, 0.0];
        let normalized = l2_normalize(&v);

        // Should remain unchanged
        assert!((normalized[0] - 1.0).abs() < 1e-6);
        assert!((normalized[1] - 0.0).abs() < 1e-6);
        assert!((normalized[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let v = array![0.0, 0.0, 0.0];
        let normalized = l2_normalize(&v);

        // Should return zero vector (no normalization possible)
        assert!((normalized[0] - 0.0).abs() < 1e-6);
        assert!((normalized[1] - 0.0).abs() < 1e-6);
        assert!((normalized[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_negative_values() {
        let v = array![-3.0, -4.0];
        let normalized = l2_normalize(&v);

        // Check unit length
        let norm = l2_norm(&normalized);
        assert!((norm - 1.0).abs() < 1e-6);

        // Check direction (should preserve sign)
        assert!((normalized[0] - (-0.6)).abs() < 1e-6);
        assert!((normalized[1] - (-0.8)).abs() < 1e-6);
    }
}
