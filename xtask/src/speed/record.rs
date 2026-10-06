use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

use serde::Serialize;

use super::cli::{Route, SpeedResult};
use super::fixtures::{digest, MODEL, REVISION};

#[derive(Default, Serialize)]
pub(super) struct Counts {
    pub items: usize,
    pub vectors: usize,
    pub windows: usize,
    pub source_content_tokens: usize,
    pub forward_calls_derived: usize,
    pub real_tokens: usize,
    pub physical_token_rows_derived: usize,
    pub real_squared_lengths: usize,
    pub physical_squared_lengths_derived: usize,
    pub admission_token_rows_derived: usize,
    pub admission_squared_lengths_derived: usize,
    pub raw_output_bytes: usize,
}

pub(super) fn counts(lengths: &[usize], route: Route) -> Counts {
    let mut result = Counts {
        items: lengths.len(),
        vectors: lengths.len(),
        real_tokens: lengths.iter().sum(),
        source_content_tokens: lengths.iter().map(|n| n.saturating_sub(2)).sum(),
        real_squared_lengths: lengths.iter().map(|n| n * n).sum(),
        raw_output_bytes: lengths.len() * 768 * 4,
        ..Counts::default()
    };
    for chunk in lengths.chunks(4) {
        let longest = chunk.iter().max().copied().unwrap_or(0);
        result.admission_token_rows_derived += chunk.len() * longest;
        result.admission_squared_lengths_derived += chunk.len() * longest * longest;
        match route {
            Route::Outcomes => {
                result.forward_calls_derived += chunk.len();
                result.physical_token_rows_derived += chunk.iter().sum::<usize>();
                result.physical_squared_lengths_derived +=
                    chunk.iter().map(|n| n * n).sum::<usize>();
            }
            Route::Batch => {
                result.forward_calls_derived += 1;
                result.physical_token_rows_derived += chunk.len() * longest;
                result.physical_squared_lengths_derived += chunk.len() * longest * longest;
            }
        }
    }
    result
}

#[derive(Serialize)]
pub(super) struct Context {
    source_commit: String,
    source_tree: String,
    source_status: String,
    binary_sha256: String,
    build_features_from_launcher: String,
    model: &'static str,
    revision: &'static str,
    device: &'static str,
    dtype: &'static str,
    host_os: &'static str,
    host_arch: &'static str,
    cpu: String,
    os_version: String,
    pub fixture_manifest_sha256: String,
    pub installed_manifest_sha256: String,
    pub threads: usize,
    pub effective_rayon_threads: usize,
    pub effective_candle_threads: usize,
    pub load_average_start: String,
}

fn command(program: &str, args: &[&str], cwd: &Path) -> SpeedResult<String> {
    let output = Command::new(program).args(args).current_dir(cwd).output()?;
    if !output.status.success() {
        return Err(format!("speed_metadata_command: {program} exited {}", output.status).into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

pub(super) fn context(
    repository: &Path,
    fixtures: String,
    manifest: String,
    threads: usize,
) -> SpeedResult<Context> {
    let configuration = tessera::configure_cpu_threads(threads)?;
    let source_status = command(
        "git",
        &["status", "--porcelain=v1", "--untracked-files=no"],
        repository,
    )?;
    if !source_status.is_empty() {
        return Err("speed_dirty_source_tree".into());
    }
    Ok(Context {
        source_commit: command("git", &["rev-parse", "HEAD"], repository)?,
        source_tree: command("git", &["rev-parse", "HEAD^{tree}"], repository)?,
        source_status,
        binary_sha256: digest(&std::fs::read(std::env::current_exe()?)?),
        build_features_from_launcher: std::env::var("TESSERA_SPEED_FEATURES")?,
        model: MODEL,
        revision: REVISION,
        device: "cpu",
        dtype: "f32",
        host_os: std::env::consts::OS,
        host_arch: std::env::consts::ARCH,
        cpu: command("sysctl", &["-n", "machdep.cpu.brand_string"], repository)?,
        os_version: command("sw_vers", &["-productVersion"], repository)?,
        fixture_manifest_sha256: fixtures,
        installed_manifest_sha256: manifest,
        threads,
        effective_rayon_threads: configuration.rayon_threads().get(),
        effective_candle_threads: configuration.candle_threads().get(),
        load_average_start: command("sysctl", &["-n", "vm.loadavg"], repository)?,
    })
}

pub(super) struct Recorder {
    file: File,
    context: Context,
}

#[derive(Serialize)]
struct Observation<'a> {
    schema_version: u32,
    context: &'a Context,
    journey: &'a str,
    route: &'a str,
    dataset: &'a str,
    observation: usize,
    elapsed_ns: u128,
    elapsed_seconds: f64,
    counts: &'a Counts,
    output_sha256: String,
    sampled_rss_bytes_after: u64,
    status: &'static str,
    timing_boundary: &'static str,
}

impl Recorder {
    pub fn new(path: &Path, context: Context) -> SpeedResult<Self> {
        Ok(Self {
            file: OpenOptions::new().write(true).create_new(true).open(path)?,
            context,
        })
    }
    pub fn write(
        &mut self,
        identity: (&str, &str, &str, usize),
        elapsed: Duration,
        counts: &Counts,
        hash: String,
    ) -> SpeedResult<()> {
        let rss = command(
            "ps",
            &["-o", "rss=", "-p", &std::process::id().to_string()],
            Path::new("."),
        )?
        .parse::<u64>()?
        .checked_mul(1024)
        .ok_or("speed_rss_overflow")?;
        let observation = Observation { schema_version: 1, context: &self.context,
            journey: identity.0, route: identity.1, dataset: identity.2, observation: identity.3,
            elapsed_ns: elapsed.as_nanos(), elapsed_seconds: elapsed.as_secs_f64(), counts,
            output_sha256: hash, sampled_rss_bytes_after: rss, status: "passed",
            timing_boundary: "public call through dimension/count/finiteness checks and output hash sink; excludes JSONL/RSS sampling" };
        serde_json::to_writer(&mut self.file, &observation)?;
        self.file.write_all(b"\n")?;
        self.file.flush()?;
        Ok(())
    }
    pub fn finish(self) -> SpeedResult<()> {
        self.file.sync_all()?;
        Ok(())
    }
}
