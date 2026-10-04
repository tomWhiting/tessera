//! MurmurHash3 x86 32-bit, as Python's `mmh3.hash` computes it with seed 0.

const C1: u32 = 0xcc9e_2d51;
const C2: u32 = 0x1b87_3593;

/// Signed 32-bit MurmurHash3 of the UTF-8 bytes of `text`, seed 0.
pub fn mmh3_hash(text: &str) -> i32 {
    let bytes = text.as_bytes();
    let mut hash: u32 = 0;
    let mut blocks = bytes.chunks_exact(4);
    for block in &mut blocks {
        let mut k = u32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        k = k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        hash ^= k;
        hash = hash
            .rotate_left(13)
            .wrapping_mul(5)
            .wrapping_add(0xe654_6b64);
    }
    let tail = blocks.remainder();
    if !tail.is_empty() {
        let mut k = 0_u32;
        for (shift, &byte) in tail.iter().enumerate() {
            k |= u32::from(byte) << (8 * shift);
        }
        k = k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        hash ^= k;
    }
    // The length is mixed in modulo 2^32, as the reference implementation does.
    hash ^= u32::try_from(bytes.len() & 0xffff_ffff).unwrap_or(u32::MAX);
    hash ^= hash >> 16;
    hash = hash.wrapping_mul(0x85eb_ca6b);
    hash ^= hash >> 13;
    hash = hash.wrapping_mul(0xc2b2_ae35);
    hash ^= hash >> 16;
    i32::from_ne_bytes(hash.to_ne_bytes())
}

#[cfg(test)]
mod tests {
    use super::mmh3_hash;

    #[test]
    fn matches_python_mmh3_hash_values() {
        // Values printed by Python mmh3 5.3.1: mmh3.hash(text).
        let cases = [
            ("", 0),
            ("a", 1_009_084_850),
            ("hello", 613_153_351),
            ("tessera", -1_691_024_159),
            ("haematit", -1_604_771_326),
            ("€12", -2_070_641_781),
            ("The quick brown fox jumps over the lazy dog", 776_992_547),
        ];
        for (text, expected) in cases {
            assert_eq!(mmh3_hash(text), expected, "{text:?}");
        }
    }
}
