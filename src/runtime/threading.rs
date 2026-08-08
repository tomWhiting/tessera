use std::num::NonZeroUsize;
use std::sync::OnceLock;

use thiserror::Error;

#[cfg(test)]
mod tests;

static CPU_THREAD_CONFIG: OnceLock<Result<CpuThreadConfig, CpuThreadConfigError>> = OnceLock::new();

/// Process-global CPU thread settings observed by Candle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuThreadConfig {
    rayon_threads: NonZeroUsize,
    candle_threads: NonZeroUsize,
}

impl CpuThreadConfig {
    /// Threads configured for Candle's Rayon-backed compute pool.
    #[must_use]
    pub const fn rayon_threads(&self) -> NonZeroUsize {
        self.rayon_threads
    }

    /// Threads configured for Candle's barrier pool.
    #[must_use]
    pub const fn candle_threads(&self) -> NonZeroUsize {
        self.candle_threads
    }
}

/// Failure to establish process-global Candle CPU thread limits.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CpuThreadConfigError {
    /// The requested default ceiling was zero.
    #[error("CPU thread ceiling must be greater than zero")]
    ZeroThreadCeiling,
    /// A pre-existing environment override was not a positive integer.
    #[error("Environment variable {name} must be a positive integer, got '{value}'")]
    InvalidEnvironmentOverride {
        /// Environment variable name.
        name: &'static str,
        /// Invalid environment value.
        value: String,
    },
}

/// Configures Candle's process-global CPU pools before their first use.
///
/// The first call wins for the lifetime of the process. Existing positive
/// `RAYON_NUM_THREADS` and `CANDLE_NUM_THREADS` values below the
/// requested ceiling are preserved. Higher overrides are clamped to the
/// ceiling. If only one is set, its effective value is copied to the other
/// variable. With no override, both are set to the smaller of available
/// parallelism and `max_threads`.
///
/// To deliberately use a higher ceiling than Tessera's two-thread default,
/// call this function before constructing any embedder. The first call wins.
///
/// Environment variables are process-global. Applications that mutate these
/// variables concurrently should call this function during single-threaded
/// startup instead. Tessera's encoder preflight calls it before initializing a
/// CPU model, which is before Tessera itself initializes Candle's private pools.
pub fn configure_cpu_threads(max_threads: usize) -> Result<CpuThreadConfig, CpuThreadConfigError> {
    let max_threads =
        NonZeroUsize::new(max_threads).ok_or(CpuThreadConfigError::ZeroThreadCeiling)?;

    CPU_THREAD_CONFIG
        .get_or_init(|| initialize_cpu_threads(max_threads))
        .clone()
}

fn initialize_cpu_threads(
    max_threads: NonZeroUsize,
) -> Result<CpuThreadConfig, CpuThreadConfigError> {
    let rayon_override =
        read_override("RAYON_NUM_THREADS")?.map(|threads| cap_threads(threads, max_threads));
    let candle_override =
        read_override("CANDLE_NUM_THREADS")?.map(|threads| cap_threads(threads, max_threads));
    let available = std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
    let default_threads = NonZeroUsize::new(available.get().min(max_threads.get()))
        .expect("minimum of positive thread counts remains positive");
    let shared_threads = rayon_override
        .or(candle_override)
        .unwrap_or(default_threads);

    let rayon_threads = rayon_override.unwrap_or(shared_threads);
    let candle_threads = candle_override.unwrap_or(shared_threads);
    std::env::set_var("RAYON_NUM_THREADS", rayon_threads.to_string());
    std::env::set_var("CANDLE_NUM_THREADS", candle_threads.to_string());

    Ok(CpuThreadConfig {
        rayon_threads,
        candle_threads,
    })
}

fn cap_threads(threads: NonZeroUsize, ceiling: NonZeroUsize) -> NonZeroUsize {
    NonZeroUsize::new(threads.get().min(ceiling.get()))
        .expect("minimum of positive thread counts remains positive")
}

fn read_override(name: &'static str) -> Result<Option<NonZeroUsize>, CpuThreadConfigError> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(None);
    };
    let value = value.to_string_lossy().into_owned();
    let parsed = value.parse::<usize>().ok().and_then(NonZeroUsize::new);
    parsed
        .map(Some)
        .ok_or(CpuThreadConfigError::InvalidEnvironmentOverride { name, value })
}
