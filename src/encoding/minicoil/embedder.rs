use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use candle_core::Device;

use super::assets::Assets;
use super::projection::ProjectionFile;
use super::resolve::resolve_words;
use super::tables::{load_stopwords, load_vocabulary};
use super::vector::{sparse_vector, MinicoilConstants, MinicoilSparse, Role};
use super::{MinicoilError, MinicoilTables, ProjectionRows};
use crate::encoding::dense::CandleDenseEncoder;
use crate::models::loader::ModelFileResolver;
use crate::models::{registry, ModelConfig, ModelInfo};
use crate::runtime::{ModelDType, ResourcePolicy};

/// Text encoder returning signed sparse values and unsigned term indices.
pub struct MinicoilEmbedder {
    encoder: CandleDenseEncoder,
    tables: MinicoilTables,
    projection: Mutex<ProjectionFile>,
}

impl MinicoilEmbedder {
    /// Loads the registered encoder and word tables on CPU.
    ///
    /// # Errors
    /// Refuses another model id, missing or mismatched pinned artifacts,
    /// invalid tables, or a model that cannot be loaded.
    pub fn new(model_id: &str) -> Result<Self> {
        Self::load(model_id, None)
    }

    /// Loads an installed encoder and local word-table directory without fetching.
    ///
    /// # Errors
    /// Refuses invalid installations, artifact sizes or digests, or model loading errors.
    pub fn from_model_dirs(encoder_dir: &Path, tables_dir: &Path) -> Result<Self> {
        Self::load("minicoil-v1", Some((encoder_dir, tables_dir)))
    }

    fn load(model_id: &str, directories: Option<(&Path, &Path)>) -> Result<Self> {
        anyhow::ensure!(
            model_id == "minicoil-v1",
            "Unknown miniCOIL model {model_id}"
        );
        let model = registry::get_model(model_id).context("Unknown miniCOIL registry model")?;
        crate::api::builder::ensure_runnable_model(model)?;
        let assets = Assets::registry()?;
        let encoder_model =
            registry::get_model(&assets.encoder_model).context("Unknown miniCOIL encoder model")?;
        anyhow::ensure!(
            encoder_model.huggingface_id == assets.encoder_repository
                && encoder_model.revision == Some(assets.encoder_revision.as_str()),
            "miniCOIL encoder registry pin differs from its declared weights"
        );
        let encoder_files = match directories {
            Some((directory, _)) => ModelFileResolver::installed(encoder_model, directory)?,
            None => ModelFileResolver::new(encoder_model)?,
        };
        assets
            .encoder_weights
            .verify(&encoder_files.get(&assets.encoder_weights.path)?)?;
        assets
            .file("tokenizer.json")?
            .verify(&encoder_files.get("tokenizer.json")?)?;
        assets
            .file("config.json")?
            .verify(&encoder_files.get("config.json")?)?;
        let table_files = if directories.is_none() {
            Some(ModelFileResolver::new(model)?)
        } else {
            None
        };
        let table_path = |name: &str| -> Result<std::path::PathBuf> {
            let path = match directories {
                Some((_, directory)) => directory.join(name),
                None => table_files
                    .as_ref()
                    .context("Missing miniCOIL file resolver")?
                    .get(name)?,
            };
            assets.file(name)?.verify(&path)?;
            Ok(path)
        };
        let tables = MinicoilTables::new(
            load_vocabulary(&table_path("minicoil.triplet.model.vocab")?)?,
            load_stopwords(&table_path("stopwords.txt")?)?,
        );
        let projection = ProjectionFile::open(&table_path("minicoil.triplet.model.npy")?)?;
        anyhow::ensure!(
            projection.vocab_rows()
                == usize::try_from(tables.vocab_size())
                    .context("miniCOIL vocabulary size does not fit this target")?,
            "miniCOIL projection and vocabulary row counts differ"
        );
        let policy =
            ResourcePolicy::new(model.context_length, 1, model.context_length, 200_000_000)
                .with_max_activation_bytes(500_000_000);
        let mut config = ModelConfig::from_registry(&assets.encoder_model)?;
        config.max_seq_length = model.context_length;
        let (encoder, _) = CandleDenseEncoder::new_with_dtype_and_resource_policy_from_dir(
            config,
            Device::Cpu,
            ModelDType::F32,
            policy,
            directories.map(|(directory, _)| directory),
        )?;
        Ok(Self {
            encoder,
            tables,
            projection: Mutex::new(projection),
        })
    }

    /// Converts one admitted text using document or question term weighting.
    ///
    /// # Errors
    /// Refuses inputs above the registered limits, invalid token outputs,
    /// a poisoned projection lock, or failed projection reads.
    pub fn encode(&self, text: &str, role: Role) -> Result<MinicoilSparse> {
        let (tokens, vectors) = self.encoder.encode_unpooled(text)?;
        let expected = tokens
            .len()
            .checked_mul(512)
            .context("miniCOIL token count overflow")?;
        if vectors.len() != expected {
            return Err(MinicoilError::TokenVectors {
                tokens: tokens.len(),
                values: vectors.len(),
                expected,
            }
            .into());
        }
        let borrowed = tokens.iter().map(String::as_str).collect::<Vec<_>>();
        let ids = resolve_words(&borrowed, &self.tables)
            .into_iter()
            .map(|word| word.vocab_id)
            .filter(|id| *id != 0)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let rows = if ids.is_empty() {
            ProjectionRows::default()
        } else {
            self.projection
                .lock()
                .map_err(|_| anyhow::anyhow!("miniCOIL projection lock poisoned"))?
                .read_rows(&ids)?
        };
        sparse_from_tokens(&tokens, &vectors, role, &self.tables, &rows).map_err(Into::into)
    }
}

/// Checks immutable asset declarations before artifact resolution.
///
/// # Errors
/// Refuses mismatched repository revisions or missing consumed artifacts.
pub fn validate_registry_assets(model: &ModelInfo) -> Result<()> {
    anyhow::ensure!(
        model.huggingface_id == "Qdrant/minicoil-v1"
            && model.revision == Some("4a7b05822a7a246d25778508593fff58fe574dfe"),
        "miniCOIL table repository or revision differs from the admitted pin"
    );
    let assets = Assets::registry()?;
    let encoder =
        registry::get_model(&assets.encoder_model).context("Unknown miniCOIL encoder model")?;
    anyhow::ensure!(
        encoder.huggingface_id == assets.encoder_repository
            && encoder.revision == Some(assets.encoder_revision.as_str())
            && encoder.safetensors_file == Some(assets.encoder_weights.path.as_str()),
        "miniCOIL encoder registry pin differs from its declared assets"
    );
    for name in [
        "config.json",
        "tokenizer.json",
        "minicoil.triplet.model.npy",
        "minicoil.triplet.model.vocab",
        "stopwords.txt",
    ] {
        assets.file(name)?;
    }
    Ok(())
}

pub(super) fn sparse_from_tokens(
    tokens: &[String],
    vectors: &[f32],
    role: Role,
    tables: &MinicoilTables,
    rows: &ProjectionRows,
) -> Result<MinicoilSparse, MinicoilError> {
    let tokens = tokens.iter().map(String::as_str).collect::<Vec<_>>();
    sparse_vector(
        &tokens,
        vectors,
        role,
        tables,
        rows,
        &MinicoilConstants::default(),
    )
}
