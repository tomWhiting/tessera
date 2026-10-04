use std::collections::{BTreeMap, HashMap};

use super::murmur::mmh3_hash;
use super::projection::{PROJECTION_INPUT, PROJECTION_OUTPUT};
use super::resolve::resolve_words;
use super::{MinicoilError, MinicoilTables, ProjectionRows};

/// Whether the text is stored (a document) or searched with (a question).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// FastEmbed `embed`: values carry the BM25 term-frequency weight.
    Document,
    /// FastEmbed `query_embed`: every weight is 1.
    Question,
}

/// FastEmbed 0.8.1's miniCOIL constants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinicoilConstants {
    /// BM25 saturation `k`.
    pub k: f64,
    /// BM25 length normalisation `b`.
    pub b: f64,
    /// BM25 average document length.
    pub avg_len: f64,
    /// Bucket width used to place unknown-word indices after the vocabulary.
    pub gap: u32,
    /// Longest stem, in characters, kept for an unknown word.
    pub token_max_length: usize,
}

impl Default for MinicoilConstants {
    fn default() -> Self {
        Self {
            k: 1.2,
            b: 0.75,
            avg_len: 150.0,
            gap: 32_000,
            token_max_length: 40,
        }
    }
}

/// A miniCOIL sparse vector: indices ascending, values in the same order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MinicoilSparse {
    /// Sparse indices, ascending.
    pub indices: Vec<u32>,
    /// Signed values, one per index. IDF is not applied.
    pub values: Vec<f32>,
}

const SPECIAL_TOKENS: [&str; 5] = ["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"];
const INT32_MAX: u64 = (1 << 31) - 1;

#[derive(Debug, Clone)]
struct WordEntry {
    word: String,
    /// Vocabulary id, or `None` for a word scored by BM25 alone.
    vocab_id: Option<u32>,
    count: usize,
    embedding: Vec<f64>,
}

/// Insertion-ordered map with Python `dict` assignment semantics.
#[derive(Default)]
struct OrderedWords {
    entries: Vec<WordEntry>,
    positions: HashMap<String, usize>,
}

impl OrderedWords {
    fn assign(&mut self, entry: WordEntry) {
        if let Some(&position) = self.positions.get(&entry.word) {
            self.entries[position] = entry;
        } else {
            self.positions
                .insert(entry.word.clone(), self.entries.len());
            self.entries.push(entry);
        }
    }

    fn get_mut(&mut self, word: &str) -> Option<&mut WordEntry> {
        let position = *self.positions.get(word)?;
        self.entries.get_mut(position)
    }
}

/// Converts one text's tokens and encoder outputs into its miniCOIL sparse vector.
///
/// `tokens` are the tokenizer's token strings for every unmasked position, special
/// tokens included; `token_vectors` holds 512 values per token in the same order.
/// `rows` must hold the projection matrix of every vocabulary id the words
/// resolve to (see [`super::resolve::resolve_words`]).
///
/// # Errors
///
/// Returns [`MinicoilError::TokenVectors`] when the vector count does not match
/// the tokens and [`MinicoilError::MissingRow`] when a resolved id has no row.
pub fn sparse_vector(
    tokens: &[&str],
    token_vectors: &[f32],
    role: Role,
    tables: &MinicoilTables,
    rows: &ProjectionRows,
    constants: &MinicoilConstants,
) -> Result<MinicoilSparse, MinicoilError> {
    let expected = tokens.len() * PROJECTION_INPUT;
    if token_vectors.len() != expected {
        return Err(MinicoilError::TokenVectors {
            tokens: tokens.len(),
            values: token_vectors.len(),
            expected,
        });
    }
    let words = resolve_words(tokens, tables);

    // Per-token vocabulary ids, word counts per id, and unknown-word counts in
    // first-seen order (VocabResolver.resolve_tokens).
    let mut token_ids = vec![0_u32; tokens.len()];
    let mut counts: HashMap<u32, usize> = HashMap::new();
    let mut unknown: Vec<(String, usize)> = Vec::new();
    for word in &words {
        for &position in &word.token_positions {
            token_ids[position] = word.vocab_id;
        }
        if word.vocab_id == 0 {
            match unknown.iter_mut().find(|(seen, _)| *seen == word.word) {
                Some((_, count)) => *count += 1,
                None => unknown.push((word.word.clone(), 1)),
            }
        } else {
            *counts.entry(word.vocab_id).or_default() += 1;
        }
    }

    // Known words in ascending vocabulary id, then unknown words (MiniCOIL._post_process_onnx_output).
    let mut sentence = OrderedWords::default();
    for (vocab_id, mean) in mean_vectors(&token_ids, token_vectors) {
        if vocab_id == 0 {
            continue;
        }
        let matrix = rows
            .get(vocab_id)
            .ok_or(MinicoilError::MissingRow { vocab_id })?;
        let word = tables.words.get(&vocab_id).cloned().unwrap_or_default();
        sentence.assign(WordEntry {
            word,
            vocab_id: Some(vocab_id),
            count: counts.get(&vocab_id).copied().unwrap_or_default(),
            embedding: project(&mean, matrix),
        });
    }
    for (word, count) in unknown {
        sentence.assign(WordEntry {
            word,
            vocab_id: None,
            count,
            embedding: vec![1.0],
        });
    }

    let cleaned = clean_words(sentence, tables, constants);
    Ok(to_sparse(&cleaned, role, tables.vocab_size, constants))
}

/// Mean token vector per vocabulary id, ids ascending (`Encoder.avg_by_vocab_ids`).
///
/// Sums accumulate in float32 in token order; the division is done in float64
/// and stored as float32, as NumPy's in-place `float32 /= int32` does.
fn mean_vectors(token_ids: &[u32], token_vectors: &[f32]) -> BTreeMap<u32, Vec<f32>> {
    let mut sums: BTreeMap<u32, (Vec<f32>, u32)> = BTreeMap::new();
    for (&vocab_id, vector) in token_ids
        .iter()
        .zip(token_vectors.chunks_exact(PROJECTION_INPUT))
    {
        let (sum, count) = sums
            .entry(vocab_id)
            .or_insert_with(|| (vec![0.0; PROJECTION_INPUT], 0));
        for (total, &value) in sum.iter_mut().zip(vector) {
            *total += value;
        }
        *count += 1;
    }
    sums.into_iter()
        .map(|(vocab_id, (sum, count))| {
            let divisor = f64::from(count);
            let mean = sum
                .iter()
                .map(|&total| to_f32(f64::from(total) / divisor))
                .collect();
            (vocab_id, mean)
        })
        .collect()
}

/// Projects a mean vector through one 512 × 4 matrix, then tanh, in float32.
fn project(mean: &[f32], matrix: &[f32]) -> Vec<f64> {
    let mut output = [0.0_f32; PROJECTION_OUTPUT];
    for (&input, weights) in mean.iter().zip(matrix.chunks_exact(PROJECTION_OUTPUT)) {
        for (total, &weight) in output.iter_mut().zip(weights) {
            *total = input.mul_add(weight, *total);
        }
    }
    output.iter().map(|value| f64::from(value.tanh())).collect()
}

/// `SparseVectorConverter.clean_words`: unknown words are split on non-word
/// characters, stemmed and merged; stop words and special tokens are dropped.
///
/// FastEmbed also drops whole words that are a single Unicode punctuation
/// character. That check is not repeated here: such a word has no word
/// characters, so the cleaning below leaves nothing of it either.
fn clean_words(
    sentence: OrderedWords,
    tables: &MinicoilTables,
    constants: &MinicoilConstants,
) -> OrderedWords {
    let unwanted = |word: &str| SPECIAL_TOKENS.contains(&word) || tables.stopwords.contains(word);
    let mut cleaned = OrderedWords::default();
    for entry in sentence.entries {
        if entry.vocab_id.is_some() {
            cleaned.assign(entry);
            continue;
        }
        if unwanted(&entry.word) {
            continue;
        }
        let spaced = entry
            .word
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '_' || c.is_whitespace() {
                    c
                } else {
                    ' '
                }
            })
            .collect::<String>();
        for subword in spaced.split_whitespace() {
            let stem = tables.stem(subword);
            if stem.chars().count() > constants.token_max_length || unwanted(&stem) {
                continue;
            }
            if let Some(existing) = cleaned.get_mut(&stem) {
                existing.count += entry.count;
            } else {
                cleaned.assign(WordEntry {
                    word: stem,
                    ..entry.clone()
                });
            }
        }
    }
    cleaned
}

/// `SparseVectorConverter.embedding_to_vector` and `embedding_to_vector_query`.
fn to_sparse(
    cleaned: &OrderedWords,
    role: Role,
    vocab_size: u32,
    constants: &MinicoilConstants,
) -> MinicoilSparse {
    let gap = u64::from(constants.gap);
    let shift = (u64::from(vocab_size) * PROJECTION_OUTPUT as u64 / gap + 2) * gap;
    let sentence_len = cleaned
        .entries
        .iter()
        .map(|entry| entry.count)
        .sum::<usize>();

    let mut pairs: Vec<(u32, f32)> = Vec::new();
    for entry in &cleaned.entries {
        let weight = match role {
            Role::Document => bm25_tf(entry.count, sentence_len, constants),
            Role::Question => 1.0,
        };
        match entry.vocab_id {
            Some(vocab_id) => {
                for (offset, value) in (0_u32..).zip(normalize(&entry.embedding)) {
                    pairs.push((vocab_id * 4 + offset, to_f32(value * weight)));
                }
            }
            None => pairs.push((unknown_index(&entry.word, shift), to_f32(weight))),
        }
    }
    pairs.sort_by_key(|&(index, _)| index);
    MinicoilSparse {
        indices: pairs.iter().map(|&(index, _)| index).collect(),
        values: pairs.iter().map(|&(_, value)| value).collect(),
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "Python converts these counts to float exactly as written; counts are far below 2^52"
)]
fn bm25_tf(occurrences: usize, sentence_len: usize, constants: &MinicoilConstants) -> f64 {
    let occurrences = occurrences as f64;
    let length = sentence_len as f64;
    // Unfused, in FastEmbed's order: res = n * (k + 1); res /= n + k * (1 - b + b * len / avg_len).
    let length_weight =
        constants.k * (1.0 - constants.b + constants.b * length / constants.avg_len);
    occurrences * (constants.k + 1.0) / (occurrences + length_weight)
}

/// Rounds a float64 to float32, as NumPy does when it stores into a float32 array.
#[expect(
    clippy::cast_possible_truncation,
    reason = "rounding to float32 is the operation FastEmbed performs here"
)]
const fn to_f32(value: f64) -> f32 {
    value as f32
}

fn normalize(vector: &[f64]) -> Vec<f64> {
    let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm < 1e-8 {
        return vector.to_vec();
    }
    vector.iter().map(|value| value / norm).collect()
}

/// `SparseVectorConverter.unkn_word_token_id`: shift + |mmh3(word)| mod (2^31 - 1 - shift).
fn unknown_index(word: &str, shift: u64) -> u32 {
    let hash = i64::from(mmh3_hash(word)).unsigned_abs();
    let index = shift + hash % (INT32_MAX - shift);
    u32::try_from(index).unwrap_or(u32::MAX)
}
