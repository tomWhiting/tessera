use std::collections::{HashMap, HashSet};
use std::path::Path;

use rust_stemmers::{Algorithm, Stemmer};
use serde::Deserialize;

use super::MinicoilError;

/// The miniCOIL vocabulary: word strings by id (ids start at 1) and the stem mapping.
#[derive(Debug, Clone, Default)]
pub struct Vocabulary {
    /// Word string for each vocabulary id.
    pub words: HashMap<u32, String>,
    /// Stem (or word form) to the vocabulary word it stands for.
    pub stem_mapping: HashMap<String, String>,
    /// Number of vocabulary ids including the unknown id 0.
    pub vocab_size: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VocabularyFile {
    vocab: Vec<String>,
    stem_mapping: HashMap<String, String>,
}

/// Reads `minicoil.triplet.model.vocab`: `{"vocab": [...], "stem_mapping": {...}}`.
///
/// # Errors
///
/// Returns [`MinicoilError::Io`] when the file cannot be read and
/// [`MinicoilError::Vocabulary`] when it is not that JSON shape, the vocabulary
/// does not fit 32-bit ids, or a stem maps to a word outside the vocabulary.
pub fn load_vocabulary(path: &Path) -> Result<Vocabulary, MinicoilError> {
    let bytes = std::fs::read(path).map_err(|source| MinicoilError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let malformed = |detail: String| MinicoilError::Vocabulary {
        path: path.to_path_buf(),
        detail,
    };
    let file: VocabularyFile =
        serde_json::from_slice(&bytes).map_err(|error| malformed(error.to_string()))?;
    let vocab_size = u32::try_from(file.vocab.len() + 1)
        .map_err(|_| malformed("vocabulary does not fit 32-bit ids".to_string()))?;
    let words = (1..vocab_size).zip(file.vocab).collect::<HashMap<_, _>>();
    let known = words.values().collect::<HashSet<_>>();
    if let Some((stem, word)) = file
        .stem_mapping
        .iter()
        .find(|(_, word)| !known.contains(word))
    {
        return Err(malformed(format!(
            "stem {stem:?} maps to {word:?}, which is not in the vocabulary"
        )));
    }
    Ok(Vocabulary {
        words,
        stem_mapping: file.stem_mapping,
        vocab_size,
    })
}

/// Reads `stopwords.txt`, one word per line, as Python's `str.splitlines` does.
///
/// # Errors
///
/// Returns [`MinicoilError::Io`] when the file cannot be read or is not UTF-8.
pub fn load_stopwords(path: &Path) -> Result<HashSet<String>, MinicoilError> {
    let text = std::fs::read_to_string(path).map_err(|source| MinicoilError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(text.lines().map(str::to_string).collect())
}

/// Everything word resolution needs: vocabulary, stem mapping, stop words and the stemmer.
pub struct MinicoilTables {
    pub(super) words: HashMap<u32, String>,
    pub(super) ids: HashMap<String, u32>,
    pub(super) stem_mapping: HashMap<String, String>,
    pub(super) stopwords: HashSet<String>,
    pub(super) vocab_size: u32,
    pub(super) stemmer: Stemmer,
}

impl MinicoilTables {
    /// Builds the tables with FastEmbed's Snowball English stemmer.
    #[must_use]
    pub fn new(vocabulary: Vocabulary, stopwords: HashSet<String>) -> Self {
        let ids = vocabulary
            .words
            .iter()
            .map(|(&id, word)| (word.clone(), id))
            .collect();
        Self {
            words: vocabulary.words,
            ids,
            stem_mapping: vocabulary.stem_mapping,
            stopwords,
            vocab_size: vocabulary.vocab_size,
            stemmer: Stemmer::create(Algorithm::English),
        }
    }

    /// Number of vocabulary ids including the unknown id 0.
    #[must_use]
    pub const fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    pub(super) fn stem(&self, word: &str) -> String {
        self.stemmer.stem(word).into_owned()
    }
}
