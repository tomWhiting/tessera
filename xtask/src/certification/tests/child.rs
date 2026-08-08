use super::{cosine, min_max, sparse_cosine, sparse_dot};

#[test]
fn compact_similarity_helpers_are_stable() {
    assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < f32::EPSILON);
    assert!((cosine(&[1.0, 0.0], &[0.0, 1.0])).abs() < f32::EPSILON);
    assert_eq!(cosine(&[1.0], &[1.0, 0.0]), f32::NEG_INFINITY);

    let left = [(1, 2.0), (4, 3.0)];
    let right = [(0, 8.0), (4, 5.0)];
    assert!((sparse_dot(&left, &right) - 15.0).abs() < f32::EPSILON);
    assert!(sparse_cosine(&left, &left) > 0.9999);
}

#[test]
fn min_max_reports_observed_range() {
    assert_eq!(min_max(&[2.0, -1.0, 4.0]), (-1.0, 4.0));
}
