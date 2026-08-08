use std::{collections::HashSet, ops::Range};

use anyhow::{Context, Result};

use crate::core::Tokenizer;
use crate::runtime::{ContextWindowConfig, ResourcePolicy};

#[cfg(test)]
mod tests;

const DEFAULT_QUERY_MAX_LENGTH: usize = 32;
const DEFAULT_DOCUMENT_MAX_LENGTH: usize = 180;
const MIN_ROLE_LENGTH: usize = 3;
const MIN_DOCUMENT_WINDOW_LENGTH: usize = MIN_ROLE_LENGTH + 1;
const ASCII_PUNCTUATION: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

/// Bounded sequence lengths for reference ColBERT preprocessing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColbertConfig {
    query_max_length: usize,
    document_max_length: usize,
}

impl ColbertConfig {
    /// Resolves explicit lengths or conservative ColBERT defaults.
    pub(crate) fn resolve(
        query_max_length: Option<usize>,
        document_max_length: Option<usize>,
        policy: &ResourcePolicy,
    ) -> Result<Self> {
        let policy_max = policy.max_sequence_tokens();
        let query = query_max_length.unwrap_or_else(|| DEFAULT_QUERY_MAX_LENGTH.min(policy_max));
        let document =
            document_max_length.unwrap_or_else(|| DEFAULT_DOCUMENT_MAX_LENGTH.min(policy_max));

        validate_length("query", query, policy_max)?;
        validate_length("document", document, policy_max)?;

        Ok(Self {
            query_max_length: query,
            document_max_length: document,
        })
    }

    pub(super) const fn query_max_length(self) -> usize {
        self.query_max_length
    }

    pub(super) const fn document_max_length(self) -> usize {
        self.document_max_length
    }
}

fn validate_length(role: &str, length: usize, policy_max: usize) -> Result<()> {
    anyhow::ensure!(
        length >= MIN_ROLE_LENGTH,
        "ColBERT {role} maximum length must be at least {MIN_ROLE_LENGTH} tokens"
    );
    anyhow::ensure!(
        length <= policy_max,
        "ColBERT {role} maximum length {length} exceeds resource policy limit {policy_max}"
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum InputRole {
    Generic,
    Query,
    Document,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct PreparedInput {
    pub(super) token_ids: Vec<u32>,
    pub(super) attention_mask: Vec<u32>,
    pub(super) output_mask: Vec<u32>,
}

impl PreparedInput {
    fn pad_to(&mut self, length: usize, pad_token_id: u32) {
        self.token_ids.resize(length, pad_token_id);
        self.attention_mask.resize(length, 0);
        self.output_mask.resize(length, 0);
    }
}

#[derive(Debug)]
struct ArtifactIds {
    cls: u32,
    sep: u32,
    mask: u32,
    pad: u32,
    query_marker: u32,
    document_marker: u32,
    punctuation: HashSet<u32>,
}

/// Tokenizer-artifact-derived state required by ColBERT preprocessing.
pub(super) struct ColbertPreprocessor {
    config: ColbertConfig,
    ids: ArtifactIds,
}

impl ColbertPreprocessor {
    pub(super) fn from_tokenizer(tokenizer: &Tokenizer, config: ColbertConfig) -> Result<Self> {
        let cls = required_id(tokenizer, "[CLS]", "classification")?;
        let sep = required_id(tokenizer, "[SEP]", "separator")?;
        let mask = required_id(tokenizer, "[MASK]", "query augmentation")?;
        let pad = required_id(tokenizer, "[PAD]", "padding")?;
        let query_marker = required_id(tokenizer, "[unused0]", "[Q] marker")?;
        let document_marker = required_id(tokenizer, "[unused1]", "[D] marker")?;
        let unknown = required_id(tokenizer, "[UNK]", "unknown-token detection")?;
        let punctuation = punctuation_ids(tokenizer, unknown)?;

        Ok(Self {
            config,
            ids: ArtifactIds {
                cls,
                sep,
                mask,
                pad,
                query_marker,
                document_marker,
                punctuation,
            },
        })
    }

    pub(super) fn prepare_batch(
        &self,
        tokenizer: &Tokenizer,
        texts: &[&str],
        role: InputRole,
    ) -> Result<Vec<PreparedInput>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let policy = tokenizer.resource_policy();
        policy
            .validate_batch(texts.len(), 0)
            .map_err(anyhow::Error::new)?;

        let mut prepared = texts
            .iter()
            .map(|text| self.prepare(tokenizer, text, role))
            .collect::<Result<Vec<_>>>()?;
        let padded_length = prepared
            .iter()
            .map(|input| input.token_ids.len())
            .max()
            .unwrap_or(0);

        policy
            .validate_sequence(padded_length)
            .map_err(anyhow::Error::new)?;
        policy
            .validate_batch(texts.len(), padded_length)
            .map_err(anyhow::Error::new)?;
        for input in &mut prepared {
            input.pad_to(padded_length, self.ids.pad);
        }
        Ok(prepared)
    }

    pub(super) fn prepare(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        role: InputRole,
    ) -> Result<PreparedInput> {
        match role {
            InputRole::Generic => {
                let (token_ids, attention_mask) = tokenizer.encode(text, true)?;
                Ok(PreparedInput {
                    output_mask: attention_mask.clone(),
                    token_ids,
                    attention_mask,
                })
            }
            InputRole::Query | InputRole::Document => {
                let (token_ids, _) = tokenizer.encode_for_bounded_truncation(text, true)?;
                let prepared = prepare_role_tokens(&token_ids, role, self.config, &self.ids)?;
                let policy = tokenizer.resource_policy();
                policy
                    .validate_sequence(prepared.token_ids.len())
                    .map_err(anyhow::Error::new)?;
                policy
                    .validate_batch(1, prepared.token_ids.len())
                    .map_err(anyhow::Error::new)?;
                Ok(prepared)
            }
        }
    }

    /// Converts the public final-input budget into the tokenizer planner's
    /// budget. The planner accounts for `[CLS]` and `[SEP]`; ColBERT inserts
    /// the additional `[D]` row after planning.
    pub(super) fn document_window_plan_config(
        &self,
        config: ContextWindowConfig,
    ) -> Result<ContextWindowConfig> {
        document_window_plan_config(config, self.config)
    }

    pub(super) fn prepare_document_window(
        &self,
        token_ids: &[u32],
        attention_mask: &[u32],
        owned_local: Range<usize>,
    ) -> Result<PreparedInput> {
        prepare_document_window_tokens(
            token_ids,
            attention_mask,
            owned_local,
            self.config,
            &self.ids,
        )
    }
}

fn required_id(tokenizer: &Tokenizer, token: &str, purpose: &str) -> Result<u32> {
    tokenizer.token_to_id(token).with_context(|| {
        format!("Tokenizer artifact has no {token} token required for ColBERT {purpose}")
    })
}

fn punctuation_ids(tokenizer: &Tokenizer, unknown: u32) -> Result<HashSet<u32>> {
    ASCII_PUNCTUATION
        .chars()
        .map(|symbol| {
            let text = symbol.to_string();
            let (ids, _) = tokenizer.encode_for_bounded_truncation(&text, false)?;
            anyhow::ensure!(
                ids.len() == 1 && ids[0] != unknown,
                "Tokenizer artifact cannot represent ColBERT punctuation {symbol:?} as one known token"
            );
            Ok(ids[0])
        })
        .collect()
}

fn prepare_role_tokens(
    token_ids: &[u32],
    role: InputRole,
    config: ColbertConfig,
    ids: &ArtifactIds,
) -> Result<PreparedInput> {
    anyhow::ensure!(
        token_ids.len() >= 2
            && token_ids.first() == Some(&ids.cls)
            && token_ids.last() == Some(&ids.sep),
        "ColBERT tokenizer output must be framed by artifact [CLS] and [SEP] tokens"
    );

    let (marker, max_length) = match role {
        InputRole::Query => (ids.query_marker, config.query_max_length()),
        InputRole::Document => (ids.document_marker, config.document_max_length()),
        InputRole::Generic => anyhow::bail!("generic input has no ColBERT role marker"),
    };
    let content_capacity = max_length - MIN_ROLE_LENGTH;
    let content = token_ids[1..token_ids.len() - 1]
        .iter()
        .take(content_capacity)
        .copied();
    let mut role_ids = Vec::with_capacity(max_length);
    role_ids.extend([ids.cls, marker]);
    role_ids.extend(content);
    role_ids.push(ids.sep);

    // ColBERT query augmentation keeps every appended [MASK] position live in
    // both self-attention and MaxSim output selection.
    if role == InputRole::Query {
        role_ids.resize(max_length, ids.mask);
    }
    let attention_mask = vec![1; role_ids.len()];
    // Document punctuation still participates in contextualization but its
    // projected rows are excluded from the stored late-interaction vectors.
    let output_mask = if role == InputRole::Document {
        role_ids
            .iter()
            .map(|id| u32::from(!ids.punctuation.contains(id)))
            .collect()
    } else {
        attention_mask.clone()
    };

    Ok(PreparedInput {
        token_ids: role_ids,
        attention_mask,
        output_mask,
    })
}

fn document_window_plan_config(
    config: ContextWindowConfig,
    colbert: ColbertConfig,
) -> Result<ContextWindowConfig> {
    let window_tokens = config.window_tokens();
    anyhow::ensure!(
        window_tokens >= MIN_DOCUMENT_WINDOW_LENGTH,
        "ColBERT document windows must allow at least {MIN_DOCUMENT_WINDOW_LENGTH} tokens for [CLS], [D], content, and [SEP]"
    );
    anyhow::ensure!(
        window_tokens <= colbert.document_max_length(),
        "ColBERT document window length {window_tokens} exceeds configured document maximum {}",
        colbert.document_max_length()
    );
    Ok(ContextWindowConfig::new(
        window_tokens - 1,
        config.overlap_tokens(),
    ))
}

fn prepare_document_window_tokens(
    token_ids: &[u32],
    attention_mask: &[u32],
    owned_local: Range<usize>,
    config: ColbertConfig,
    ids: &ArtifactIds,
) -> Result<PreparedInput> {
    anyhow::ensure!(
        token_ids.len() >= 2
            && token_ids.first() == Some(&ids.cls)
            && token_ids.last() == Some(&ids.sep),
        "ColBERT document window must be framed by artifact [CLS] and [SEP] tokens"
    );
    anyhow::ensure!(
        attention_mask.len() == token_ids.len(),
        "ColBERT document window token and attention-mask lengths differ"
    );
    anyhow::ensure!(
        token_ids.len() < config.document_max_length(),
        "ColBERT document window exceeds configured document maximum {} after [D] insertion",
        config.document_max_length()
    );

    let content_length = token_ids.len() - 2;
    anyhow::ensure!(
        owned_local.start <= owned_local.end && owned_local.end <= content_length,
        "ColBERT document ownership range {:?} exceeds {content_length} window content tokens",
        owned_local
    );

    let mut role_ids = Vec::with_capacity(token_ids.len() + 1);
    role_ids.extend([ids.cls, ids.document_marker]);
    role_ids.extend_from_slice(&token_ids[1..]);

    let mut role_attention = Vec::with_capacity(attention_mask.len() + 1);
    role_attention.extend([attention_mask[0], 1]);
    role_attention.extend_from_slice(&attention_mask[1..]);

    // The planner's ownership range is relative to content only. `[CLS]` and
    // the inserted `[D]` occupy the first two model-output rows.
    let owned_rows = (owned_local.start + 2)..(owned_local.end + 2);
    let mut output_mask = vec![0; role_ids.len()];
    output_mask[0] = role_attention[0];
    output_mask[1] = role_attention[1];
    let separator = role_ids.len() - 1;
    output_mask[separator] = role_attention[separator];
    for row in owned_rows {
        output_mask[row] =
            u32::from(role_attention[row] == 1 && !ids.punctuation.contains(&role_ids[row]));
    }

    Ok(PreparedInput {
        token_ids: role_ids,
        attention_mask: role_attention,
        output_mask,
    })
}
