//! Registry-pinned model artifact resolution for `HuggingFace` Hub.

#[cfg(feature = "fetch")]
use std::ffi::OsStr;
use std::path::PathBuf;

use anyhow::Result;
#[cfg(feature = "fetch")]
use anyhow::{bail, Context};
#[cfg(feature = "fetch")]
use hf_hub::api::sync::{ApiBuilder, ApiRepo};
#[cfg(feature = "fetch")]
use hf_hub::{Cache, CacheRepo, Repo, RepoType};

#[cfg(feature = "fetch")]
use super::registry;
use super::registry::ModelInfo;

mod installed;
pub use installed::{InstalledModel, InstalledModelError};

#[cfg(test)]
mod tests;

enum ArtifactSource {
    #[cfg(feature = "fetch")]
    Online(ApiRepo),
    #[cfg(feature = "fetch")]
    Offline(CacheRepo),
    Installed(InstalledModel),
}

/// Resolves every artifact for one model from its immutable registry revision.
pub(crate) struct ModelFileResolver {
    model: &'static ModelInfo,
    source: ArtifactSource,
}

impl ModelFileResolver {
    /// Creates a resolver directly from immutable generated registry metadata.
    pub(crate) fn new(model: &'static ModelInfo) -> Result<Self> {
        registry_revision(model)?;
        #[cfg(not(feature = "fetch"))]
        {
            Err(crate::error::TesseraError::FetchingNotBuiltIn {
                model_id: model.id.to_string(),
            }
            .into())
        }
        #[cfg(feature = "fetch")]
        {
            let repo = validated_repo(model)?;

            let source = if offline_enabled()? {
                ArtifactSource::Offline(Cache::from_env().repo(repo))
            } else {
                let api = ApiBuilder::from_env()
                    .build()
                    .context("Failed to initialize HuggingFace Hub API")?;
                ArtifactSource::Online(api.repo(repo))
            };

            Ok(Self { model, source })
        }
    }

    pub(crate) fn installed(
        model: &'static ModelInfo,
        directory: &std::path::Path,
    ) -> Result<Self> {
        let installed = InstalledModel::open_for_model(model, directory)?;
        Ok(Self {
            model,
            source: ArtifactSource::Installed(installed),
        })
    }

    /// Creates a resolver using the registry pin itself.
    #[cfg(feature = "fetch")]
    fn from_registry(model_name: &str) -> Result<Self> {
        let model = registry::get_model_by_hf_id(model_name).ok_or_else(|| {
            anyhow::anyhow!("Model '{model_name}' is not registered for artifact loading")
        })?;
        Self::new(model)
    }

    /// Gets one artifact, downloading only when online and absent from the cache.
    pub(crate) fn get(&self, filename: &str) -> Result<PathBuf> {
        #[cfg(feature = "fetch")]
        let revision = self.model.revision.ok_or_else(|| {
            anyhow::anyhow!(
                "Model '{}' lost its required immutable revision",
                self.model.id
            )
        })?;
        match &self.source {
            #[cfg(feature = "fetch")]
            ArtifactSource::Online(repo) => repo.get(filename).with_context(|| {
                format!(
                    "Failed to resolve {filename} from {} at revision {}",
                    self.model.huggingface_id, revision
                )
            }),
            #[cfg(feature = "fetch")]
            ArtifactSource::Offline(repo) => repo.get(filename).ok_or_else(|| {
                anyhow::anyhow!(
                    "TESSERA_OFFLINE=1 and pinned artifact {filename} is absent from the cache for {} at revision {}",
                    self.model.huggingface_id,
                    revision
                )
            }),
            ArtifactSource::Installed(installed) => Ok(installed.get(filename)?),
        }
    }

    /// Returns the immutable metadata that defines this resolver's artifacts.
    pub(crate) const fn model(&self) -> &'static ModelInfo {
        self.model
    }

    /// Resolves the exact weight artifact declared by the immutable registry.
    pub(crate) fn weights(&self) -> Result<PathBuf> {
        self.model.safetensors_file.map_or_else(
            || self.get(self.model.pytorch_file),
            |filename| self.get(filename),
        )
    }
}

fn registry_revision(model: &ModelInfo) -> Result<&str> {
    model.revision.ok_or_else(|| {
        anyhow::anyhow!(
            "Model '{}' has no pinned HuggingFace revision and cannot load artifacts",
            model.id
        )
    })
}

#[cfg(feature = "fetch")]
fn validated_repo(model: &ModelInfo) -> Result<Repo> {
    let pinned_revision = registry_revision(model)?;
    let repo = Repo::with_revision(
        model.huggingface_id.to_string(),
        RepoType::Model,
        pinned_revision.to_string(),
    );
    Ok(repo)
}

#[cfg(feature = "fetch")]
fn offline_enabled() -> Result<bool> {
    parse_offline_flag(std::env::var_os("TESSERA_OFFLINE").as_deref())
}

#[cfg(feature = "fetch")]
fn parse_offline_flag(value: Option<&OsStr>) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(value) if value == OsStr::new("1") => Ok(true),
        Some(value) => bail!(
            "TESSERA_OFFLINE must be unset or exactly '1', got {}",
            value.display()
        ),
    }
}

/// Downloads or resolves a registered model artifact at its pinned revision.
///
/// With `TESSERA_OFFLINE=1`, this performs cache lookup only and fails when the
/// exact pinned artifact is absent.
#[cfg(feature = "fetch")]
pub fn download_model_file(model_name: &str, filename: &str) -> Result<PathBuf> {
    ModelFileResolver::from_registry(model_name)?.get(filename)
}

/// Downloads or resolves `config.json` for a registered model.
#[cfg(feature = "fetch")]
pub fn download_config(model_name: &str) -> Result<PathBuf> {
    let files = ModelFileResolver::from_registry(model_name)?;
    files.get(files.model().config_file)
}

/// Downloads or resolves `tokenizer.json` for a registered model.
#[cfg(feature = "fetch")]
pub fn download_tokenizer(model_name: &str) -> Result<PathBuf> {
    let files = ModelFileResolver::from_registry(model_name)?;
    files.get(files.model().tokenizer_file)
}
