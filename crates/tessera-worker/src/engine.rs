use std::path::Path;

use haem_frames::embedding::{
    check_model_limits_windowed, check_ready, check_vectors_windowed, encode_vector, input_refusal,
    Distance, Embed, FailedCode, Input, ItemCode, Kind, Message, Model, Outcome, Ready, Start,
    Vectors,
};
use tessera::{
    Device, EmbeddingOutcome, EmbeddingRefusal, InstalledModel, ModelConfig, Role, TesseraDense,
    TesseraDenseBuilder,
};

use crate::failure::Failure;
use crate::policy::Budget;

mod windows;

pub struct Engine {
    model: TesseraDense,
    max_tokens: usize,
    ready: Message,
}

fn wire_number(value: usize, name: &str) -> Result<u64, Failure> {
    u64::try_from(value).map_err(|_| Failure::limits(format!("{name} exceeds u64")))
}

/// A vector's values as one slice, as the codec takes them.
fn contiguous(values: Option<&[f32]>) -> Result<&[f32], Failure> {
    values.ok_or_else(|| {
        Failure::new(
            FailedCode::EmbedOutputInvalid,
            "vector storage is not contiguous",
        )
    })
}

impl Engine {
    pub fn load(
        start: &Start,
        budget: &Budget,
        resources: &haem_worker::setup::Resources,
    ) -> Result<Self, Failure> {
        let directory = Path::new(&start.model_dir);
        let id = InstalledModel::registry_id(directory)
            .map_err(|error| Failure::installed(&error, directory))?;
        let entry = tessera::models::registry::get_model(&id).ok_or_else(|| {
            Failure::new(
                FailedCode::EmbedModelMismatch,
                format!("model {id} is not registered"),
            )
        })?;
        if !entry.is_runnable()
            || !matches!(
                entry.model_type,
                tessera::models::registry::ModelType::Dense
            )
        {
            return Err(Failure::new(
                FailedCode::EmbedModelMismatch,
                format!("model {id} has no runnable dense adapter"),
            ));
        }
        let config = ModelConfig::from_registry(&id).map_err(|error| {
            Failure::new(
                FailedCode::EmbedModelMismatch,
                format!("{}: {error}", directory.display()),
            )
        })?;
        let policy = budget.policy(config.embedding_dim, entry.parameters)?;
        let model = TesseraDenseBuilder::new()
            .model(&id)
            .model_dir(directory)
            .device(Device::Cpu)
            .resource_policy(policy)
            .dtype(tessera::ModelDType::F32)
            .build()
            .map_err(|error| Failure::model_load(&error, directory))?;
        model
            .encode_batch_outcomes(&[], Some(Role::Query))
            .map_err(|error| Failure::model_load(&error, directory))?;
        let identity = model.identity();
        let max_tokens = identity.max_tokens;
        let manifest_sha256 = identity.manifest_sha256.as_ref().ok_or_else(|| {
            Failure::new(
                FailedCode::EmbedModelMismatch,
                "installed model has no manifest digest",
            )
        })?;
        let distance = match identity.distance.as_str() {
            "cosine" => Distance::Cosine,
            "dot" => Distance::Dot,
            "euclidean" => Distance::Euclidean,
            other => {
                return Err(Failure::new(
                    FailedCode::EmbedModelMismatch,
                    format!("unsupported model distance {other}"),
                ))
            }
        };
        let ready = Ready {
            protocol: start.protocol,
            worker: env!("TESSERA_WORKER_BUILD").to_string(),
            core_limit: resources.core_limit,
            descriptors_closed: resources.descriptors_closed,
            environment: resources.environment,
            model: Model {
                name: identity.name.clone(),
                revision: identity.revision.clone(),
                manifest_sha256: manifest_sha256.clone(),
                dimensions: wire_number(identity.dimensions, "dimensions")?,
                max_tokens: wire_number(identity.max_tokens, "max_tokens")?,
                special_tokens: wire_number(identity.special_tokens, "special_tokens")?,
                prefix_tokens: wire_number(identity.prefix_tokens, "prefix_tokens")?,
                normalised: identity.normalised,
                distance,
            },
        };
        check_ready(&ready)?;
        check_model_limits_windowed(&start.limits, &ready.model, start.windows)?;
        Ok(Self {
            model,
            max_tokens,
            ready: Message::Ready(ready),
        })
    }

    pub const fn ready_message(&self) -> &Message {
        &self.ready
    }

    /// Answers an Embed; a document Embed under protocol 2 may answer windows.
    pub fn encode(&self, request: &Embed, start: &Start) -> Result<Vectors, Failure> {
        let items = match start.windows.filter(|_| request.kind == Kind::Document) {
            Some(asked) => self.encode_windows(request, &start.limits, asked)?,
            None => self.encode_whole(request, start.limits.input_bytes)?,
        };
        let vectors = Vectors { items };
        let Message::Ready(ready) = &self.ready else {
            return Err(Failure::new(
                FailedCode::EmbedOutputInvalid,
                "worker model identity is unavailable",
            ));
        };
        check_vectors_windowed(&vectors, request, ready, &start.limits, start.windows)?;
        Ok(vectors)
    }

    fn encode_whole(&self, request: &Embed, input_bytes: u64) -> Result<Vec<Outcome>, Failure> {
        let role = match request.kind {
            Kind::Document => Role::Document,
            Kind::Query => Role::Query,
        };
        let texts: Vec<&str> = request
            .items
            .iter()
            .map(|item| item.text.as_str())
            .collect();
        let outcomes = self
            .model
            .encode_batch_outcomes(&texts, Some(role))
            .map_err(|error| Failure::inference(&error))?;
        if outcomes.len() != request.items.len() {
            return Err(Failure::new(
                FailedCode::EmbedOutputInvalid,
                "embedding outcome count does not match request",
            ));
        }
        if let Some(failure) = overlength_failure(request, &outcomes) {
            return Err(failure);
        }
        request
            .items
            .iter()
            .zip(outcomes)
            .map(|(input, outcome)| Self::outcome(input, outcome, input_bytes))
            .collect()
    }

    fn outcome(
        input: &Input,
        outcome: EmbeddingOutcome,
        input_bytes: u64,
    ) -> Result<Outcome, Failure> {
        let expected = input_refusal(&input.text, input_bytes);
        Ok(match outcome {
            EmbeddingOutcome::Embedded(embedding) => {
                if expected.is_some() {
                    return Err(Failure::new(
                        FailedCode::EmbedOutputInvalid,
                        "refused input produced a vector",
                    ));
                }
                Outcome::Vector {
                    id: input.id.clone(),
                    vector: encode_vector(contiguous(embedding.values().as_slice())?)?,
                    tokens_read: wire_number(embedding.tokens_total(), "tokens_read")?,
                    tokens_total: wire_number(embedding.tokens_total(), "tokens_total")?,
                }
            }
            EmbeddingOutcome::Refused(refusal) => {
                let code = match refusal {
                    EmbeddingRefusal::Empty => ItemCode::EmbedInputEmpty,
                    EmbeddingRefusal::TooLarge { .. } => ItemCode::EmbedInputTooLarge,
                    EmbeddingRefusal::NoContentTokens => {
                        return Err(Failure::limits(format!(
                            "embed_input_no_content_tokens item {:?}",
                            input.id
                        )));
                    }
                    EmbeddingRefusal::TextLongerThanModel {
                        tokens_total,
                        tokens_limit,
                    } => {
                        return Err(Failure::overlength(&[(
                            input.id.as_str(),
                            tokens_total,
                            tokens_limit,
                        )]));
                    }
                };
                if expected != Some(code) {
                    return Err(Failure::new(
                        FailedCode::EmbedOutputInvalid,
                        "input refusal does not match the shared classifier",
                    ));
                }
                Outcome::Refused {
                    id: input.id.clone(),
                    code,
                    tokens_total: None,
                    tokens_limit: None,
                }
            }
        })
    }
}

fn overlength_failure(request: &Embed, outcomes: &[EmbeddingOutcome]) -> Option<Failure> {
    let overlength: Vec<_> = request
        .items
        .iter()
        .zip(outcomes)
        .filter_map(|(input, outcome)| match outcome {
            EmbeddingOutcome::Refused(EmbeddingRefusal::TextLongerThanModel {
                tokens_total,
                tokens_limit,
            }) => Some((input.id.as_str(), *tokens_total, *tokens_limit)),
            _ => None,
        })
        .collect();
    if overlength.is_empty() {
        None
    } else {
        Some(Failure::overlength(&overlength))
    }
}
