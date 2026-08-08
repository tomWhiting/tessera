use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus};
use std::thread;
use std::time::{Duration, Instant};

use super::artifacts;
use super::evidence::{self, ChildOutcome, PeakRssEvidence, RecordInput};
use super::spec::{self, CertResult, LoadedSpec};

const RSS_SAMPLE_INTERVAL_MS: u64 = 50;

#[derive(Debug, Clone)]
pub(crate) struct RunOptions {
    pub(crate) profile: String,
    pub(crate) repeat: usize,
}

pub(crate) fn run_model(repository: &Path, model_id: &str, options: &RunOptions) -> CertResult<()> {
    if options.repeat == 0 {
        return Err("--repeat must be greater than zero".into());
    }
    let loaded = spec::load_model(repository, model_id)?;
    loaded.spec.profile(&options.profile)?;
    let mut failures = Vec::new();
    for repetition in 1..=options.repeat {
        if let Err(error) = launch_one(repository, &loaded, options, repetition) {
            failures.push(format!("run {repetition}: {error}"));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "model '{}' certification failed: {}",
            loaded.spec.model.id,
            failures.join("; ")
        )
        .into())
    }
}

pub(crate) fn run_all(repository: &Path, options: &RunOptions) -> CertResult<()> {
    let loaded = spec::load_all(repository)?;
    let mut failures = Vec::new();
    for document in loaded {
        println!("certifying '{}' serially", document.spec.model.id);
        if let Err(error) = run_model(repository, &document.spec.model.id, options) {
            failures.push(error.to_string());
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} model certifications failed:\n- {}",
            failures.len(),
            failures.join("\n- ")
        )
        .into())
    }
}

fn launch_one(
    repository: &Path,
    loaded: &LoadedSpec,
    options: &RunOptions,
    repetition: usize,
) -> CertResult<()> {
    let profile = loaded.spec.profile(&options.profile)?;
    let started_unix_ms = evidence::now_unix_ms()?;
    let outcome_path = outcome_path(
        repository,
        &loaded.spec.model.id,
        repetition,
        started_unix_ms,
    );
    if let Some(parent) = outcome_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let executable = std::env::current_exe()?;
    let mut child = Command::new(executable)
        .args([
            "cert",
            "__one",
            "--model",
            &loaded.spec.model.id,
            "--profile",
            &options.profile,
            "--outcome",
        ])
        .arg(&outcome_path)
        .current_dir(repository)
        .env("HF_HOME", artifacts::cache_root(repository))
        .env("TESSERA_OFFLINE", "1")
        .env("RAYON_NUM_THREADS", profile.process.cpu_threads.to_string())
        .env(
            "CANDLE_NUM_THREADS",
            profile.process.cpu_threads.to_string(),
        )
        .spawn()?;
    let child_pid = child.id();
    let monitor = monitor_child(
        &mut child,
        Duration::from_secs(profile.process.timeout_seconds),
        profile.process.max_peak_rss_bytes,
    )?;
    let completed_unix_ms = evidence::now_unix_ms()?;
    let mut outcome = read_outcome(&outcome_path, &monitor)
        .unwrap_or_else(|error| failed_outcome(format!("child outcome was unreadable: {error}")));
    let _ = fs::remove_file(&outcome_path);
    apply_launcher_result(
        &mut outcome,
        monitor.launcher_error.as_deref(),
        monitor.status.success(),
        &monitor.status.to_string(),
    );
    let record = evidence::build_record(
        repository,
        RecordInput {
            loaded,
            profile: &options.profile,
            repetition,
            child_pid,
            started_unix_ms,
            completed_unix_ms,
            peak_rss: PeakRssEvidence {
                enforced: monitor.rss_enforced,
                bytes: monitor.peak_rss_bytes,
                limit_bytes: profile.process.max_peak_rss_bytes,
                method: monitor.rss_method,
                sample_interval_ms: RSS_SAMPLE_INTERVAL_MS,
            },
            outcome,
        },
    )?;
    let path = evidence::evidence_path(
        repository,
        &loaded.spec.model.id,
        repetition,
        completed_unix_ms,
    );
    evidence::write_record(&path, &record)?;
    println!("evidence: {}", path.display());
    if record.status == "passed" {
        Ok(())
    } else {
        Err(record
            .error
            .unwrap_or_else(|| "child certification failed".to_string())
            .into())
    }
}

struct MonitorResult {
    status: ExitStatus,
    peak_rss_bytes: Option<u64>,
    rss_enforced: bool,
    rss_method: String,
    launcher_error: Option<String>,
}

fn monitor_child(
    child: &mut Child,
    timeout: Duration,
    max_peak_rss_bytes: u64,
) -> CertResult<MonitorResult> {
    let started = Instant::now();
    let mut peak_rss_bytes = None::<u64>;
    let mut rss_samples = 0_usize;
    let mut launcher_error = None;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(MonitorResult {
                status,
                peak_rss_bytes,
                rss_enforced: rss_samples > 0,
                rss_method: rss_method(rss_samples),
                launcher_error,
            });
        }
        if let Some(rss_bytes) = process_rss_bytes(child.id()) {
            rss_samples += 1;
            peak_rss_bytes = Some(peak_rss_bytes.map_or(rss_bytes, |peak| peak.max(rss_bytes)));
            if rss_bytes > max_peak_rss_bytes {
                launcher_error = Some(format!(
                    "sampled RSS {rss_bytes} exceeded limit {max_peak_rss_bytes}"
                ));
                child.kill()?;
                let status = child.wait()?;
                return Ok(MonitorResult {
                    status,
                    peak_rss_bytes,
                    rss_enforced: true,
                    rss_method: rss_method(rss_samples),
                    launcher_error,
                });
            }
        }
        if started.elapsed() >= timeout {
            launcher_error = Some(format!(
                "child exceeded timeout of {} seconds",
                timeout.as_secs()
            ));
            child.kill()?;
            let status = child.wait()?;
            return Ok(MonitorResult {
                status,
                peak_rss_bytes,
                rss_enforced: rss_samples > 0,
                rss_method: rss_method(rss_samples),
                launcher_error,
            });
        }
        thread::sleep(Duration::from_millis(RSS_SAMPLE_INTERVAL_MS));
    }
}

fn read_outcome(path: &Path, monitor: &MonitorResult) -> CertResult<ChildOutcome> {
    if path.exists() {
        return Ok(serde_json::from_slice(&fs::read(path)?)?);
    }
    Ok(failed_outcome(
        monitor.launcher_error.clone().unwrap_or_else(|| {
            format!(
                "child exited with {} before writing an outcome",
                monitor.status
            )
        }),
    ))
}

fn apply_launcher_result(
    outcome: &mut ChildOutcome,
    launcher_error: Option<&str>,
    exit_success: bool,
    exit_status: &str,
) {
    if let Some(launcher_error) = launcher_error {
        let detail = outcome
            .error
            .take()
            .filter(|child_error| child_error != launcher_error)
            .map_or_else(
                || launcher_error.to_string(),
                |child_error| format!("{launcher_error}; child reported: {child_error}"),
            );
        outcome.status = "failed".to_string();
        outcome.error = Some(detail);
    } else if !exit_success && outcome.status == "passed" {
        outcome.status = "failed".to_string();
        outcome.error = Some(format!("child exited with {exit_status}"));
    }
}

fn failed_outcome(error: String) -> ChildOutcome {
    ChildOutcome {
        status: "failed".to_string(),
        error: Some(error),
        verified_artifacts: Vec::new(),
        observation: None,
    }
}

fn outcome_path(
    repository: &Path,
    model_id: &str,
    repetition: usize,
    timestamp_ms: u128,
) -> PathBuf {
    repository
        .join(".tessera/cert-evidence")
        .join(model_id)
        .join(format!(".{timestamp_ms}-run-{repetition}.outcome"))
}

#[cfg(unix)]
fn process_rss_bytes(process_id: u32) -> Option<u64> {
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &process_id.to_string()])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let kibibytes = String::from_utf8(output.stdout)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()?;
    kibibytes.checked_mul(1024)
}

#[cfg(not(unix))]
fn process_rss_bytes(_process_id: u32) -> Option<u64> {
    None
}

fn rss_method(samples: usize) -> String {
    if samples == 0 {
        "unavailable".to_string()
    } else {
        "sampled-process-rss-watchdog".to_string()
    }
}

#[cfg(test)]
#[path = "tests/process.rs"]
mod tests;
