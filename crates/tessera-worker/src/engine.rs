use std::path::Path;

use haem_frames::embedding::{
    check_model_limits, check_ready, check_vectors, encode_vector, input_refusal, Distance, Embed,
    FailedCode, ItemCode, Kind, Message, Model, Outcome, Ready, Start, Vectors,
};
use tessera::{
    CutEmbeddingOutcome, Device, EmbeddingRefusal, InstalledModel, ModelConfig, Role, TesseraDense,
    TesseraDenseBuilder,
};

use crate::failure::Failure;
use crate::policy::Budget;

pub struct Engine {
    model: TesseraDense,
    ready: Message,
}

fn wire_number(value: usize, name: &str) -> Result<u64, Failure> {
    u64::try_from(value).map_err(|_| Failure::limits(format!("{name} exceeds u64")))
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
            .encode_batch_cut(&[], Some(Role::Query))
            .map_err(|error| Failure::model_load(&error, directory))?;
        let identity = model.identity();
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
            protocol: haem_frames::embedding::PROTOCOL,
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
        check_model_limits(&start.limits, &ready.model)?;
        Ok(Self {
            model,
            ready: Message::Ready(ready),
        })
    }

    pub const fn ready_message(&self) -> &Message {
        &self.ready
    }

    pub fn encode(&self, request: &Embed, start: &Start) -> Result<Vectors, Failure> {
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
            .encode_batch_cut(&texts, Some(role))
            .map_err(|error| Failure::inference(&error))?;
        if outcomes.len() != request.items.len() {
            return Err(Failure::new(
                FailedCode::EmbedOutputInvalid,
                "embedding outcome count does not match request",
            ));
        }
        let mut items = Vec::with_capacity(outcomes.len());
        for (input, outcome) in request.items.iter().zip(outcomes) {
            let expected = input_refusal(&input.text, start.limits.input_bytes);
            items.push(match outcome {
                CutEmbeddingOutcome::Embedded(embedding) => {
                    if expected.is_some() {
                        return Err(Failure::new(
                            FailedCode::EmbedOutputInvalid,
                            "refused input produced a vector",
                        ));
                    }
                    let values = embedding.values().as_slice().ok_or_else(|| {
                        Failure::new(
                            FailedCode::EmbedOutputInvalid,
                            "vector storage is not contiguous",
                        )
                    })?;
                    Outcome::Vector {
                        id: input.id.clone(),
                        vector: encode_vector(values)?,
                        tokens_read: wire_number(embedding.tokens_read(), "tokens_read")?,
                        tokens_total: wire_number(embedding.tokens_total(), "tokens_total")?,
                    }
                }
                CutEmbeddingOutcome::Refused(refusal) => {
                    let code = match refusal {
                        EmbeddingRefusal::Empty => ItemCode::EmbedInputEmpty,
                        EmbeddingRefusal::TooLarge { .. } => ItemCode::EmbedInputTooLarge,
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
                    }
                }
            });
        }
        let vectors = Vectors { items };
        let Message::Ready(ready) = &self.ready else {
            return Err(Failure::new(
                FailedCode::EmbedOutputInvalid,
                "worker model identity is unavailable",
            ));
        };
        check_vectors(&vectors, request, ready, &start.limits)?;
        Ok(vectors)
    }
}
