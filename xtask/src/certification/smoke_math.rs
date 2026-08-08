pub(super) fn norm(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

pub(super) fn cosine(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() {
        return f32::NEG_INFINITY;
    }
    cosine_iter(left.iter(), right.iter())
}

pub(super) fn cosine_iter<'a>(
    left: impl Iterator<Item = &'a f32>,
    right: impl Iterator<Item = &'a f32>,
) -> f32 {
    let (mut dot, mut left_norm, mut right_norm) = (0.0_f32, 0.0_f32, 0.0_f32);
    for (left, right) in left.zip(right) {
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    dot / (left_norm.sqrt() * right_norm.sqrt()).max(f32::MIN_POSITIVE)
}

pub(super) fn sparse_dot(left: &[(usize, f32)], right: &[(usize, f32)]) -> f32 {
    let (mut left_index, mut right_index, mut score) = (0, 0, 0.0_f32);
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].0.cmp(&right[right_index].0) {
            std::cmp::Ordering::Less => left_index += 1,
            std::cmp::Ordering::Greater => right_index += 1,
            std::cmp::Ordering::Equal => {
                score += left[left_index].1 * right[right_index].1;
                left_index += 1;
                right_index += 1;
            }
        }
    }
    score
}

pub(super) fn sparse_cosine(left: &[(usize, f32)], right: &[(usize, f32)]) -> f32 {
    let dot = sparse_dot(left, right);
    let left_norm = left
        .iter()
        .map(|(_, value)| value * value)
        .sum::<f32>()
        .sqrt();
    let right_norm = right
        .iter()
        .map(|(_, value)| value * value)
        .sum::<f32>()
        .sqrt();
    dot / (left_norm * right_norm).max(f32::MIN_POSITIVE)
}

pub(super) fn min_max(values: &[f32]) -> (f32, f32) {
    values.iter().copied().fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(minimum, maximum), value| (minimum.min(value), maximum.max(value)),
    )
}
