use std::error::Error;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};

use super::artifacts;
use super::process::{self, RunOptions};
use super::readiness;
use super::spec;

#[derive(Debug, Parser)]
#[command(
    name = "cert",
    about = "Local, resource-bounded model certification",
    disable_help_subcommand = true
)]
struct CertCli {
    #[command(subcommand)]
    command: CertCommand,
}

#[derive(Debug, Subcommand)]
enum CertCommand {
    /// List checked certification specifications and cache presence.
    List,
    /// Download one pinned model into the dedicated cache and verify every byte.
    Fetch {
        /// Registry model ID.
        #[arg(long)]
        model: String,
    },
    /// Run one model in a fresh, monitored, offline CPU child process.
    Run {
        /// Registry model ID.
        #[arg(long)]
        model: String,
        /// Execution device. The first certification tranche is CPU-only.
        #[arg(long, value_enum, default_value_t = CertDevice::Cpu)]
        device: CertDevice,
        /// Named profile from the checked specification.
        #[arg(long, default_value = "smoke")]
        profile: String,
        /// Number of fresh child processes to run serially.
        #[arg(long, default_value_t = 1)]
        repeat: usize,
    },
    /// Run every checked specification serially, one child process at a time.
    RunAll {
        /// Execution device. The first certification tranche is CPU-only.
        #[arg(long, value_enum, default_value_t = CertDevice::Cpu)]
        device: CertDevice,
        /// Named profile from each checked specification.
        #[arg(long, default_value = "smoke")]
        profile: String,
        /// Number of fresh child processes per model.
        #[arg(long, default_value_t = 1)]
        repeat: usize,
    },
    /// Report whether one model has enough immutable evidence for promotion.
    Readiness {
        /// Registry model ID.
        #[arg(long)]
        model: String,
        /// Emit the report as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Remove one model from the dedicated, re-downloadable certification cache.
    Purge {
        /// Registry model ID.
        #[arg(long)]
        model: String,
    },
    /// Internal one-model child entrypoint; invoke through `cert run`.
    #[command(name = "__one", hide = true)]
    One {
        #[arg(long)]
        model: String,
        #[arg(long)]
        profile: String,
        #[arg(long)]
        outcome: PathBuf,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CertDevice {
    Cpu,
}

pub(crate) fn run(
    repository: &Path,
    arguments: impl Iterator<Item = String>,
) -> Result<(), Box<dyn Error>> {
    let arguments = std::iter::once(OsString::from("cert")).chain(arguments.map(OsString::from));
    let cli = CertCli::parse_from(arguments);
    match cli.command {
        CertCommand::List => list(repository),
        CertCommand::Fetch { model } => fetch(repository, &model),
        CertCommand::Run {
            model,
            device: _,
            profile,
            repeat,
        } => process::run_model(repository, &model, &RunOptions { profile, repeat }),
        CertCommand::RunAll {
            device: _,
            profile,
            repeat,
        } => process::run_all(repository, &RunOptions { profile, repeat }),
        CertCommand::Readiness { model, json } => readiness(repository, &model, json),
        CertCommand::Purge { model } => purge(repository, &model),
        CertCommand::One {
            model,
            profile,
            outcome,
        } => super::child::run(repository, &model, &profile, &outcome),
    }
}

fn list(repository: &Path) -> Result<(), Box<dyn Error>> {
    let loaded = spec::load_all(repository)?;
    println!("MODEL\tREPRESENTATION\tREVISION\tARTIFACTS\tCACHE");
    for document in loaded {
        let bytes = document.spec.expected_artifact_bytes()?;
        let cache = artifacts::cache_state(repository, &document)?;
        println!(
            "{}\t{:?}\t{}\t{}\t{}",
            document.spec.model.id,
            document.spec.model.representation,
            document.spec.model.revision,
            display_bytes(bytes),
            cache.label()
        );
    }
    Ok(())
}

fn fetch(repository: &Path, model_id: &str) -> Result<(), Box<dyn Error>> {
    let loaded = spec::load_model(repository, model_id)?;
    println!(
        "fetching {} at immutable revision {}",
        loaded.spec.model.repository, loaded.spec.model.revision
    );
    for artifact in artifacts::fetch(repository, &loaded)? {
        println!(
            "verified {} ({}; sha256 {})",
            artifact.path,
            display_bytes(artifact.size_bytes),
            artifact.sha256
        );
    }
    println!(
        "model '{}' is cached and byte-verified",
        loaded.spec.model.id
    );
    Ok(())
}

fn readiness(repository: &Path, model_id: &str, json: bool) -> Result<(), Box<dyn Error>> {
    let loaded = spec::load_model(repository, model_id)?;
    let report = readiness::evaluate(repository, &loaded);
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else if report.ready {
        println!(
            "{} is promotion-ready with {} matching runs",
            report.model_id, report.successful_runs
        );
    } else {
        println!("{} is not promotion-ready:", report.model_id);
        for reason in &report.reasons {
            println!("- {reason}");
        }
    }
    if report.ready {
        Ok(())
    } else {
        Err(format!("model '{}' is not promotion-ready", report.model_id).into())
    }
}

fn purge(repository: &Path, model_id: &str) -> Result<(), Box<dyn Error>> {
    let loaded = spec::load_model(repository, model_id)?;
    if artifacts::purge(repository, &loaded)? {
        println!(
            "removed the re-downloadable certification cache for '{}'",
            loaded.spec.model.id
        );
    } else {
        println!(
            "model '{}' has no certification cache",
            loaded.spec.model.id
        );
    }
    Ok(())
}

fn display_bytes(bytes: u64) -> String {
    const MEBIBYTE: u64 = 1024 * 1024;
    if bytes < MEBIBYTE {
        format!("{bytes} B")
    } else {
        format!("{:.1} MiB", bytes as f64 / MEBIBYTE as f64)
    }
}

#[cfg(test)]
#[path = "tests/cli.rs"]
mod tests;
