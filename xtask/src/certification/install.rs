use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};

use serde::Serialize;
use tessera::InstalledModel;

use super::artifacts::{self, VerifiedArtifact};
use super::spec::{CertResult, LoadedSpec, Representation};

#[derive(Serialize)]
struct Manifest<'a> {
    schema_version: u32,
    model: ManifestModel<'a>,
    artifacts: &'a [VerifiedArtifact],
}

#[derive(Serialize)]
struct ManifestModel<'a> {
    id: &'a str,
    repository: &'a str,
    revision: &'a str,
}

pub(crate) fn run(repository: &Path, model_id: &str, directory: &Path) -> CertResult<()> {
    let loaded = super::spec::load_model(repository, model_id)?;
    let digest = install(repository, &loaded, directory)?;
    println!(
        "installed '{}' in {}; manifest sha256 {digest}",
        model_id,
        directory.display()
    );
    Ok(())
}

fn install(repository: &Path, loaded: &LoadedSpec, directory: &Path) -> CertResult<String> {
    require_dense(loaded.spec.model.representation)?;
    for artifact in &loaded.spec.artifacts {
        let mut components = Path::new(&artifact.path).components();
        if !matches!(components.next(), Some(Component::Normal(_)))
            || components.next().is_some()
            || artifact.path == "manifest.json"
        {
            return Err(format!(
                "installed artifact '{}' must be a plain filename distinct from manifest.json",
                artifact.path
            )
            .into());
        }
    }
    let verified = artifacts::verify_cached(repository, loaded)?;
    let created_directory = prepare_directory(directory)?;
    let mut created_files = Vec::new();
    match copy_and_manifest(repository, loaded, directory, &verified, &mut created_files) {
        Ok(digest) => Ok(digest),
        Err(error) => match cleanup_owned(directory, &created_files, created_directory) {
            Ok(()) => Err(error),
            Err(cleanup_error) => {
                Err(format!("{error}; install cleanup failed: {cleanup_error}").into())
            }
        },
    }
}

pub(crate) fn require_dense(representation: Representation) -> CertResult<()> {
    if representation != Representation::Dense {
        return Err("--model-dir and cert install support dense models only".into());
    }
    Ok(())
}

fn prepare_directory(directory: &Path) -> CertResult<bool> {
    match fs::symlink_metadata(directory) {
        Ok(metadata) => {
            if !metadata.is_dir() || metadata.file_type().is_symlink() {
                return Err(format!(
                    "install destination '{}' must be a directory, not a symlink or file",
                    directory.display()
                )
                .into());
            }
            if fs::read_dir(directory)?.next().transpose()?.is_some() {
                return Err(
                    format!("install destination '{}' is not empty", directory.display()).into(),
                );
            }
            Ok(false)
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            fs::create_dir(directory)?;
            Ok(true)
        }
        Err(error) => Err(error.into()),
    }
}

fn copy_and_manifest(
    repository: &Path,
    loaded: &LoadedSpec,
    directory: &Path,
    verified: &[VerifiedArtifact],
    created_files: &mut Vec<PathBuf>,
) -> CertResult<String> {
    for artifact in verified {
        let source = artifacts::cached_artifact_path(repository, loaded, &artifact.path)?;
        let destination = directory.join(&artifact.path);
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&destination)?;
        created_files.push(destination);
        let mut input = fs::File::open(source)?;
        io::copy(&mut input, &mut output)?;
        output.sync_all()?;
    }
    artifacts::verify_directory(directory, loaded)?;
    let manifest = Manifest {
        schema_version: 1,
        model: ManifestModel {
            id: &loaded.spec.model.id,
            repository: &loaded.spec.model.repository,
            revision: &loaded.spec.model.revision,
        },
        artifacts: verified,
    };
    let destination = directory.join("manifest.json");
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&destination)?;
    created_files.push(destination);
    output.write_all(&serde_json::to_vec_pretty(&manifest)?)?;
    output.sync_all()?;
    let installed = InstalledModel::open(&loaded.spec.model.id, directory)?;
    Ok(installed.manifest_sha256().to_string())
}

fn cleanup_owned(directory: &Path, files: &[PathBuf], created_directory: bool) -> CertResult<()> {
    let mut failures = Vec::new();
    for path in files.iter().rev() {
        if let Err(error) = fs::remove_file(path) {
            failures.push(format!("{}: {error}", path.display()));
        }
    }
    if created_directory {
        if let Err(error) = fs::remove_dir(directory) {
            failures.push(format!("{}: {error}", directory.display()));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; ").into())
    }
}

#[cfg(test)]
#[path = "tests/install.rs"]
mod tests;
