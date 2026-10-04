use std::io::Read;
use std::path::{Component, Path};

use anyhow::{Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Artifact {
    pub(super) path: String,
    pub(super) size_bytes: u64,
    pub(super) sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Assets {
    pub(super) encoder_model: String,
    pub(super) encoder_repository: String,
    pub(super) encoder_revision: String,
    pub(super) encoder_weights: Artifact,
    pub(super) files: Vec<Artifact>,
}

impl Assets {
    pub(super) fn registry() -> Result<Self> {
        let mut registry: serde_json::Value =
            serde_json::from_str(include_str!("../../../models.json"))
                .context("Reading sparse asset registry")?;
        let models = registry
            .pointer_mut("/model_categories/sparse/models")
            .and_then(serde_json::Value::as_array_mut)
            .context("Sparse asset registry has no models")?;
        let entry = models
            .iter_mut()
            .find(|entry| {
                entry.get("id").and_then(serde_json::Value::as_str) == Some("minicoil-v1")
            })
            .context("Sparse asset registry has no minicoil-v1")?;
        let assets: Self = serde_json::from_value(
            entry
                .get_mut("minicoil_assets")
                .context("Missing miniCOIL asset metadata")?
                .take(),
        )
        .context("Invalid miniCOIL asset metadata")?;
        for artifact in assets
            .files
            .iter()
            .chain(std::iter::once(&assets.encoder_weights))
        {
            anyhow::ensure!(
                !artifact.path.is_empty()
                    && Path::new(&artifact.path)
                        .components()
                        .all(|part| matches!(part, Component::Normal(_))),
                "Invalid miniCOIL artifact path"
            );
            anyhow::ensure!(
                artifact.size_bytes > 0
                    && artifact.sha256.len() == 64
                    && artifact
                        .sha256
                        .bytes()
                        .all(|value| value.is_ascii_digit() || (b'a'..=b'f').contains(&value)),
                "Invalid miniCOIL artifact size or digest"
            );
        }
        Ok(assets)
    }

    pub(super) fn file(&self, name: &str) -> Result<&Artifact> {
        self.files
            .iter()
            .find(|artifact| artifact.path == name)
            .with_context(|| format!("Missing miniCOIL artifact {name}"))
    }
}

impl Artifact {
    pub(super) fn verify(&self, path: &Path) -> Result<()> {
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("Opening miniCOIL artifact {}", self.path))?;
        anyhow::ensure!(
            file.metadata()?.len() == self.size_bytes,
            "miniCOIL artifact {} has the wrong size",
            self.path
        );
        let mut digest = Sha256::new();
        let mut buffer = [0_u8; 65_536];
        loop {
            let count = file
                .read(&mut buffer)
                .with_context(|| format!("Reading miniCOIL artifact {}", self.path))?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
        anyhow::ensure!(
            format!("{:x}", digest.finalize()) == self.sha256,
            "miniCOIL artifact {} SHA-256 mismatch",
            self.path
        );
        Ok(())
    }
}
