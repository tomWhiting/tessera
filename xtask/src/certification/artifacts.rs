use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::Command;

use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Cache, Repo, RepoType};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::spec::{CertResult, LoadedSpec};

const HASH_BUFFER_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct VerifiedArtifact {
    pub(crate) path: String,
    pub(crate) size_bytes: u64,
    pub(crate) sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CacheState {
    Missing,
    Present,
    SizeMismatch,
}

impl CacheState {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Missing => "missing",
            Self::Present => "present",
            Self::SizeMismatch => "size-mismatch",
        }
    }
}

pub(crate) fn cache_root(repository: &Path) -> PathBuf {
    repository.join(".tessera/cert-cache")
}

pub(crate) fn configure_cache(repository: &Path) -> CertResult<PathBuf> {
    let root = cache_root(repository);
    fs::create_dir_all(&root)?;
    std::env::set_var("HF_HOME", &root);
    Ok(root)
}

pub(crate) fn fetch(repository: &Path, loaded: &LoadedSpec) -> CertResult<Vec<VerifiedArtifact>> {
    let root = configure_cache(repository)?;
    let profile = loaded.spec.profile("smoke")?;
    let available = available_disk_bytes(&root).ok_or_else(|| {
        "cannot enforce the certification cache free-disk requirement on this platform".to_string()
    })?;
    let required_free = required_fetch_free_bytes(
        loaded.spec.expected_artifact_bytes()?,
        profile.process.min_free_disk_bytes,
    )?;
    if available < required_free {
        return Err(format!(
            "certification cache has {available} free bytes, but fetching model '{}' while retaining its reserve requires at least {required_free}",
            loaded.spec.model.id
        )
        .into());
    }

    let api = ApiBuilder::from_env().with_progress(true).build()?;
    let repository_handle = api.repo(pinned_repo(loaded));
    let mut verified = Vec::with_capacity(loaded.spec.artifacts.len());
    for expected in &loaded.spec.artifacts {
        let path = repository_handle.get(&expected.path)?;
        verified.push(verify_one(
            &path,
            &expected.path,
            expected.size_bytes,
            &expected.sha256,
        )?);
    }
    Ok(verified)
}

fn required_fetch_free_bytes(artifact_bytes: u64, retained_free_bytes: u64) -> CertResult<u64> {
    artifact_bytes
        .checked_add(retained_free_bytes)
        .ok_or_else(|| "certification disk requirement overflow".into())
}

pub(crate) fn verify_cached(
    repository: &Path,
    loaded: &LoadedSpec,
) -> CertResult<Vec<VerifiedArtifact>> {
    configure_cache(repository)?;
    let cache = Cache::from_env().repo(pinned_repo(loaded));
    loaded
        .spec
        .artifacts
        .iter()
        .map(|expected| {
            let path = cache.get(&expected.path).ok_or_else(|| {
                format!(
                    "model '{}' is missing cached artifact '{}'",
                    loaded.spec.model.id, expected.path
                )
            })?;
            verify_one(&path, &expected.path, expected.size_bytes, &expected.sha256)
        })
        .collect()
}

pub(crate) fn cache_state(repository: &Path, loaded: &LoadedSpec) -> CertResult<CacheState> {
    let root = cache_root(repository);
    if !root.exists() {
        return Ok(CacheState::Missing);
    }
    std::env::set_var("HF_HOME", root);
    let cache = Cache::from_env().repo(pinned_repo(loaded));
    let mut state = CacheState::Present;
    for expected in &loaded.spec.artifacts {
        let Some(path) = cache.get(&expected.path) else {
            return Ok(CacheState::Missing);
        };
        if fs::metadata(path)?.len() != expected.size_bytes {
            state = CacheState::SizeMismatch;
        }
    }
    Ok(state)
}

pub(crate) fn purge(repository: &Path, loaded: &LoadedSpec) -> CertResult<bool> {
    let root = configure_cache(repository)?;
    let hub = root.join("hub");
    let folder = pinned_repo(loaded).folder_name();
    let target = hub.join(folder);
    if target.parent() != Some(hub.as_path()) || !target.starts_with(&hub) {
        return Err("refusing to purge a path outside the dedicated certification cache".into());
    }
    if !target.exists() {
        return Ok(false);
    }
    fs::remove_dir_all(target)?;
    Ok(true)
}

fn pinned_repo(loaded: &LoadedSpec) -> Repo {
    Repo::with_revision(
        loaded.spec.model.repository.clone(),
        RepoType::Model,
        loaded.spec.model.revision.clone(),
    )
}

fn verify_one(
    local_path: &Path,
    artifact_path: &str,
    expected_size: u64,
    expected_sha256: &str,
) -> CertResult<VerifiedArtifact> {
    let actual_size = fs::metadata(local_path)?.len();
    if actual_size != expected_size {
        return Err(format!(
            "artifact '{artifact_path}' is {actual_size} bytes; expected {expected_size}"
        )
        .into());
    }
    let actual_sha256 = sha256_file(local_path)?;
    if actual_sha256 != expected_sha256 {
        return Err(format!(
            "artifact '{artifact_path}' SHA-256 mismatch: got {actual_sha256}, expected {expected_sha256}"
        )
        .into());
    }
    Ok(VerifiedArtifact {
        path: artifact_path.to_string(),
        size_bytes: actual_size,
        sha256: actual_sha256,
    })
}

fn sha256_file(path: &Path) -> CertResult<String> {
    let mut reader = BufReader::with_capacity(HASH_BUFFER_BYTES, File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; HASH_BUFFER_BYTES];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(unix)]
fn available_disk_bytes(path: &Path) -> Option<u64> {
    let output = Command::new("df").args(["-Pk"]).arg(path).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let available_kib = text
        .lines()
        .last()?
        .split_whitespace()
        .nth(3)?
        .parse::<u64>()
        .ok()?;
    available_kib.checked_mul(1024)
}

#[cfg(not(unix))]
fn available_disk_bytes(_path: &Path) -> Option<u64> {
    None
}

#[cfg(test)]
#[path = "tests/artifacts.rs"]
mod tests;
