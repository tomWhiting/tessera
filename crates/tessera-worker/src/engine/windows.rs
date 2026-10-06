//! Protocol 2 (EMB-E2): a document longer than the model reads answers whole windows.

use haem_frames::embedding::{
    encode_vector, input_refusal, Embed, FailedCode, Input, ItemCode, Limits, Outcome, Window,
    Windows,
};
use tessera::{
    ContextWindowConfig, DenseWindowEmbedding, EmbeddingOutcome, EmbeddingRefusal, Role,
    TesseraError, WindowEmbeddingOutcome,
};

use super::{contiguous, wire_number, Engine};
use crate::failure::Failure;

/// What one item gets, decided by tokenizing alone, before any inference.
enum Plan {
    /// The byte classifier refuses it (`None`) or the model reads its tokens whole.
    Whole(Option<usize>),
    /// This many windows cover it.
    Windows(usize),
    /// Refused by name with the text's content tokens; never embedded.
    Refused(ItemCode, usize),
}

fn size(value: u64, name: &str) -> Result<usize, Failure> {
    usize::try_from(value).map_err(|_| Failure::limits(format!("{name} exceeds usize")))
}

fn invalid(message: &str) -> Failure {
    Failure::new(FailedCode::EmbedOutputInvalid, message)
}

/// A measuring error names a content-free item as protocol 1 does; anything else is inference.
fn measure_failure(error: &TesseraError, id: &str) -> Failure {
    let no_content = match error {
        TesseraError::EncodingError { source, .. } => source.chain().any(|cause| {
            matches!(
                cause.downcast_ref::<EmbeddingRefusal>(),
                Some(EmbeddingRefusal::NoContentTokens)
            )
        }),
        _ => false,
    };
    if no_content {
        Failure::limits(format!("embed_input_no_content_tokens item {id:?}"))
    } else {
        Failure::inference(error)
    }
}

impl Engine {
    /// Answers a document Embed under the Start's `overlap_tokens` and `max_windows`.
    ///
    /// Every item is planned before any inference, so a text over the cap or one
    /// a window cannot hold uncut is refused by name and never embedded.
    pub(super) fn encode_windows(
        &self,
        request: &Embed,
        limits: &Limits,
        asked: Windows,
    ) -> Result<Vec<Outcome>, Failure> {
        let overlap = size(asked.overlap_tokens, "overlap_tokens")?;
        let cap = size(asked.max_windows, "max_windows")?;
        let fed = size(limits.tokens, "tokens")?;
        let plans = request
            .items
            .iter()
            .map(|input| self.plan(input, limits.input_bytes, overlap, cap, fed))
            .collect::<Result<Vec<_>, _>>()?;
        let texts = |wanted: fn(&Plan) -> bool| -> Vec<&str> {
            request
                .items
                .iter()
                .zip(&plans)
                .filter(|(_, plan)| wanted(plan))
                .map(|(input, _)| input.text.as_str())
                .collect()
        };
        let whole = texts(|plan| matches!(plan, Plan::Whole(_)));
        let long = texts(|plan| matches!(plan, Plan::Windows(_)));
        let mut whole = if whole.is_empty() {
            Vec::new()
        } else {
            self.model
                .encode_batch_outcomes(&whole, Some(Role::Document))
                .map_err(|error| Failure::inference(&error))?
        }
        .into_iter();
        let config = ContextWindowConfig::new(self.max_tokens, overlap);
        let mut long = if long.is_empty() {
            Vec::new()
        } else {
            self.model
                .encode_batch_windows(&long, Some(Role::Document), Some(config))
                .map_err(|error| Failure::inference(&error))?
        }
        .into_iter();
        let missing = || invalid("embedding outcome count does not match request");
        request
            .items
            .iter()
            .zip(plans)
            .map(|(input, plan)| match plan {
                Plan::Whole(tokens) => match (whole.next().ok_or_else(missing)?, tokens) {
                    (
                        EmbeddingOutcome::Refused(EmbeddingRefusal::TextLongerThanModel { .. }),
                        Some(tokens),
                    ) => self.refused(input, ItemCode::TextLongerThanModel, tokens),
                    (outcome, _) => Self::outcome(input, outcome, limits.input_bytes),
                },
                Plan::Windows(count) => {
                    self.windowed(input, long.next().ok_or_else(missing)?, count, fed)
                }
                Plan::Refused(code, tokens) => self.refused(input, code, tokens),
            })
            .collect()
    }

    fn plan(
        &self,
        input: &Input,
        input_bytes: u64,
        overlap: usize,
        cap: usize,
        fed: usize,
    ) -> Result<Plan, Failure> {
        if input_refusal(&input.text, input_bytes).is_some() {
            return Ok(Plan::Whole(None));
        }
        let extent = self
            .model
            .window_extent(&input.text, Some(Role::Document), self.max_tokens)
            .map_err(|error| measure_failure(&error, &input.id))?;
        if extent.tokens_total <= extent.content_capacity {
            return Ok(Plan::Whole(Some(extent.tokens_total)));
        }
        Ok(match extent.windows(overlap) {
            Some(count) if count > cap => {
                Plan::Refused(ItemCode::WindowsOverCap, extent.tokens_total)
            }
            // Never cut: a window the model reads whole must also fit the tokens it is fed.
            Some(count) if self.max_tokens <= fed => Plan::Windows(count),
            _ => Plan::Refused(ItemCode::TextLongerThanModel, extent.tokens_total),
        })
    }

    fn windowed(
        &self,
        input: &Input,
        outcome: WindowEmbeddingOutcome,
        count: usize,
        fed: usize,
    ) -> Result<Outcome, Failure> {
        let WindowEmbeddingOutcome::Embedded(embedded) = outcome else {
            return Err(invalid("a planned windowed input was refused"));
        };
        if embedded.windows().len() != count {
            return Err(invalid("window count does not match its plan"));
        }
        // A window over what the model is fed would have been cut: it answers no vector.
        if embedded
            .windows()
            .iter()
            .any(|window| window.tokens() > fed.min(self.max_tokens))
        {
            return self.refused(
                input,
                ItemCode::TextLongerThanModel,
                embedded.tokens_total(),
            );
        }
        let windows = embedded
            .windows()
            .iter()
            .map(window)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Outcome::Windows {
            id: input.id.clone(),
            windows,
            tokens_total: wire_number(embedded.tokens_total(), "tokens_total")?,
        })
    }

    fn refused(&self, input: &Input, code: ItemCode, tokens: usize) -> Result<Outcome, Failure> {
        Ok(Outcome::Refused {
            id: input.id.clone(),
            code,
            tokens_total: Some(wire_number(tokens, "tokens_total")?),
            tokens_limit: Some(wire_number(self.max_tokens, "max_tokens")?),
        })
    }
}

fn window(window: &DenseWindowEmbedding) -> Result<Window, Failure> {
    Ok(Window {
        start: wire_number(window.byte_start(), "start")?,
        end: wire_number(window.byte_end(), "end")?,
        vector: encode_vector(contiguous(window.values().as_slice())?)?,
        tokens_read: wire_number(window.content_tokens(), "tokens_read")?,
    })
}
