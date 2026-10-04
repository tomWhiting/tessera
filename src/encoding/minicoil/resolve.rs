use super::MinicoilTables;

/// How a rebuilt word found its vocabulary id, in FastEmbed's branch order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resolution {
    /// The word is a stop word; id 0.
    StopWord,
    /// The word is itself a vocabulary word.
    Exact,
    /// The word is a key of the stem mapping.
    StemMapping,
    /// The word's Snowball English stem is a key of the stem mapping.
    Stemmed,
    /// None of the above; id 0.
    Unknown,
}

/// One word rebuilt from WordPiece tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedWord {
    /// The word as joined from its pieces.
    pub word: String,
    /// Positions of its tokens in the token list.
    pub token_positions: Vec<usize>,
    /// Which branch resolved it.
    pub resolution: Resolution,
    /// Its vocabulary id, 0 when it has none.
    pub vocab_id: u32,
}

const CONTINUATION: &str = "##";

/// Joins `##` continuation pieces into words and resolves each word.
///
/// Mirrors `VocabResolver._reconstruct_bpe` and `VocabResolver.resolve_tokens`:
/// every token, special tokens included, belongs to exactly one word.
#[must_use]
pub fn resolve_words(tokens: &[&str], tables: &MinicoilTables) -> Vec<ResolvedWord> {
    let mut words: Vec<(String, Vec<usize>)> = Vec::new();
    for (position, token) in tokens.iter().enumerate() {
        match (token.strip_prefix(CONTINUATION), words.last_mut()) {
            (Some(piece), Some((word, positions))) => {
                word.push_str(piece);
                positions.push(position);
            }
            (Some(piece), None) => words.push((piece.to_string(), vec![position])),
            (None, _) => words.push(((*token).to_string(), vec![position])),
        }
    }
    words
        .into_iter()
        .map(|(word, token_positions)| {
            let (resolution, vocab_id) = resolve(&word, tables);
            ResolvedWord {
                word,
                token_positions,
                resolution,
                vocab_id,
            }
        })
        .collect()
}

fn resolve(word: &str, tables: &MinicoilTables) -> (Resolution, u32) {
    if tables.stopwords.contains(word) {
        return (Resolution::StopWord, 0);
    }
    if let Some(&id) = tables.ids.get(word) {
        return (Resolution::Exact, id);
    }
    if let Some(id) = mapped_id(word, tables) {
        return (Resolution::StemMapping, id);
    }
    mapped_id(&tables.stem(word), tables)
        .map_or((Resolution::Unknown, 0), |id| (Resolution::Stemmed, id))
}

fn mapped_id(key: &str, tables: &MinicoilTables) -> Option<u32> {
    tables
        .stem_mapping
        .get(key)
        .and_then(|word| tables.ids.get(word))
        .copied()
}
