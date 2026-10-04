use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

use super::{compare_vectors, record_path, write_record, ChildMeasurement, MeasurementRecord};
use crate::certification::{evidence, install, process, spec};
use spec::{CertResult, ProcessLimits};

pub(crate) fn run(
    repository: &Path,
    model: &str,
    profile_name: &str,
    model_dir: Option<&Path>,
) -> CertResult<()> {
    let loaded = spec::load_model(repository, model)?;
    install::require_dense(loaded.spec.model.representation)?;
    let profile = loaded.spec.profile(profile_name)?;
    let directory = model_dir.map(fs::canonicalize).transpose()?;
    let invocation = evidence::now_unix_ms()?;
    let path = record_path(repository, model, invocation);
    let (source_commit, source_dirty) = evidence::source_state(repository)?;
    fs::create_dir_all(path.parent().ok_or("measurement path has no parent")?)?;
    let scratch = path.with_extension("scratch");
    fs::create_dir(&scratch)?;
    let input = ChildInput {
        repository,
        model,
        profile: profile_name,
        model_dir: directory.as_deref(),
        scratch: &scratch,
        process: &profile.process,
    };
    let result = (|| {
        let batch = launch(&input, profile.process.cpu_threads, true)?;
        let one = launch(&input, 1, false)?;
        let configured = launch(&input, profile.process.cpu_threads, false)?;
        if batch.installed_manifest_sha256 != one.installed_manifest_sha256
            || batch.installed_manifest_sha256 != configured.installed_manifest_sha256
            || directory.is_some() != batch.installed_manifest_sha256.is_some()
        {
            return Err(
                "measurement_manifest_mismatch: children did not load the same installed manifest"
                    .into(),
            );
        }
        let after = evidence::source_state(repository)?;
        if after != (source_commit.clone(), source_dirty) {
            return Err(
                "measurement_source_changed: commit or dirty flag changed during measurement"
                    .into(),
            );
        }
        let record = MeasurementRecord {
            schema_version: 1,
            kind: "dense_measurement".into(),
            source_commit,
            source_dirty,
            model_id: model.to_string(),
            model_revision: loaded.spec.model.revision.clone(),
            profile: profile_name.to_string(),
            spec_sha256: loaded.sha256,
            installed_manifest_sha256: batch.installed_manifest_sha256,
            configured_threads: profile.process.cpu_threads,
            comparison_threads: 1,
            batch: batch.batch,
            threads: compare_vectors(&one.vector, &configured.vector)?,
        };
        write_record(&path, &record)?;
        println!("measurement: {}", path.display());
        println!("{}", serde_json::to_string_pretty(&record)?);
        Ok(())
    })();
    match fs::remove_dir_all(&scratch) {
        Ok(()) => result,
        Err(error) => Err(format!(
            "measurement_cleanup_failed: {error}; measurement result: {result:?}"
        )
        .into()),
    }
}

struct ChildInput<'a> {
    repository: &'a Path,
    model: &'a str,
    profile: &'a str,
    model_dir: Option<&'a Path>,
    scratch: &'a Path,
    process: &'a ProcessLimits,
}

fn launch(input: &ChildInput<'_>, threads: usize, batch: bool) -> CertResult<ChildMeasurement> {
    let outcome = input
        .scratch
        .join(format!("threads-{threads}-batch-{batch}.child"));
    let mut command = Command::new(std::env::current_exe()?);
    command
        .args([
            "cert",
            "__measure-one",
            "--model",
            input.model,
            "--profile",
            input.profile,
            "--threads",
        ])
        .arg(threads.to_string())
        .arg("--outcome")
        .arg(&outcome)
        .current_dir(input.repository)
        .env("RAYON_NUM_THREADS", threads.to_string())
        .env("CANDLE_NUM_THREADS", threads.to_string());
    if batch {
        command.arg("--batch");
    }
    process::configure_source(&mut command, input.repository, input.model_dir);
    let result = (|| {
        let mut child = command.spawn()?;
        let monitor = process::monitor_child(
            &mut child,
            Duration::from_secs(input.process.timeout_seconds),
            input.process.max_peak_rss_bytes,
        )?;
        if let Some(error) = monitor.launcher_error {
            return Err(format!("measurement_child_limit: {error}").into());
        }
        if !monitor.status.success() {
            return Err(format!(
                "measurement_child_failed: threads={threads}, batch={batch}, exit={}",
                monitor.status
            )
            .into());
        }
        Ok(serde_json::from_slice(&fs::read(&outcome)?)?)
    })();
    match fs::remove_file(&outcome) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(
                format!("measurement_cleanup_failed: {error}; child result: {result:?}").into(),
            )
        }
    }
    result
}
