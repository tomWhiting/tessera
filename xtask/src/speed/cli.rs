use std::error::Error;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};

use super::{fixtures, measure};

pub(super) type SpeedResult<T> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(super) enum Journey {
    Backfill,
    Query,
    Windows,
    Aggregate,
    Startup,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(super) enum Route {
    Batch,
    Outcomes,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(super) enum Dataset {
    Mixed,
    Short,
    Full,
}

#[derive(Parser)]
#[command(name = "speed", about = "Offline public-API speed observations")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Verify frozen fixture hashes and token assertions, without loading weights.
    Validate {
        #[arg(long)]
        model_dir: PathBuf,
    },
    /// Measure one variant in this fresh process, recording every observation.
    Run(Options),
}

#[derive(clap::Args)]
pub(super) struct Options {
    #[arg(long)]
    pub model_dir: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, value_enum)]
    pub journey: Journey,
    #[arg(long, value_enum, default_value_t = Route::Outcomes)]
    pub route: Route,
    #[arg(long, value_enum, default_value_t = Dataset::Short)]
    pub dataset: Dataset,
    #[arg(long, default_value_t = 2)]
    pub threads: usize,
}

pub(crate) fn run(repository: &Path, arguments: impl Iterator<Item = String>) -> SpeedResult<()> {
    let cli = Cli::try_parse_from(std::iter::once("speed".to_string()).chain(arguments))?;
    match cli.command {
        Command::Validate { model_dir } => {
            let fixtures = fixtures::load(repository, &model_dir)?;
            println!("{}", serde_json::to_string_pretty(&fixtures.manifest)?);
            Ok(())
        }
        Command::Run(options) => {
            if ![2, 4, 8].contains(&options.threads) {
                return Err("speed_threads: only the agreed counts 2, 4 and 8 are accepted".into());
            }
            measure::run(repository, &options)
        }
    }
}
