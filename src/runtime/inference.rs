use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use thiserror::Error;

#[cfg(test)]
mod tests;

const DEFAULT_MAX_WAITERS: usize = 16;
const DEFAULT_WAIT_TIMEOUT: Duration = Duration::from_secs(30);

static PROCESS_INFERENCE_GATE: InferenceGate = InferenceGate::new();
static PROCESS_INFERENCE_CONFIG: OnceLock<InferenceGateConfig> = OnceLock::new();

/// Process-wide limits for callers waiting to execute a Candle forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InferenceGateConfig {
    max_waiters: usize,
    wait_timeout: Duration,
}

impl InferenceGateConfig {
    /// Creates an admission configuration.
    #[must_use]
    pub const fn new(max_waiters: usize, wait_timeout: Duration) -> Self {
        Self {
            max_waiters,
            wait_timeout,
        }
    }

    /// Maximum queued callers, excluding a currently executing forward pass.
    #[must_use]
    pub const fn max_waiters(self) -> usize {
        self.max_waiters
    }

    /// Maximum time a caller may wait before inference begins.
    #[must_use]
    pub const fn wait_timeout(self) -> Duration {
        self.wait_timeout
    }
}

impl Default for InferenceGateConfig {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_WAITERS, DEFAULT_WAIT_TIMEOUT)
    }
}

/// Configures process-wide inference admission before the first forward pass.
///
/// Repeating the same configuration is harmless. A different configuration
/// after first use is rejected so service behavior cannot change mid-process.
pub fn configure_inference_gate(
    config: InferenceGateConfig,
) -> Result<(), InferenceGateConfigError> {
    config.deadline()?;
    if let Some(current) = PROCESS_INFERENCE_CONFIG.get() {
        return if *current == config {
            Ok(())
        } else {
            Err(InferenceGateConfigError::AlreadyConfigured { current: *current })
        };
    }
    PROCESS_INFERENCE_CONFIG
        .set(config)
        .map_err(|_| InferenceGateConfigError::AlreadyConfigured {
            current: *PROCESS_INFERENCE_CONFIG.get().unwrap_or(&config),
        })
}

/// Failure to configure process-wide inference admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum InferenceGateConfigError {
    /// A zero timeout would reject even an immediately scheduled wakeup.
    #[error("inference wait timeout must be greater than zero")]
    ZeroTimeout,
    /// The timeout cannot be represented as a deadline on this platform.
    #[error("inference wait timeout {wait_timeout:?} is too large for this platform")]
    TimeoutTooLarge {
        /// Timeout that could not be represented as an [`Instant`] deadline.
        wait_timeout: Duration,
    },
    /// A different first-call configuration is already active.
    #[error("inference admission is already configured as {current:?}")]
    AlreadyConfigured {
        /// Active immutable configuration.
        current: InferenceGateConfig,
    },
}

/// Exclusive admission to Candle model execution within this process.
#[must_use = "the permit must remain alive for the complete model execution"]
pub struct InferencePermit<'a> {
    gate: &'a InferenceGate,
}

impl Drop for InferencePermit<'_> {
    fn drop(&mut self) {
        let mut state = match self.gate.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        state.occupied = false;
        drop(state);
        self.gate.available.notify_all();
    }
}

/// Failure to enter the process-wide inference gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum InferenceGateError {
    /// The supplied admission configuration cannot be used safely.
    #[error("invalid inference admission configuration: {source}")]
    InvalidConfiguration {
        /// Configuration validation failure.
        source: InferenceGateConfigError,
    },
    /// The gate's short-lived bookkeeping lock was poisoned.
    #[error(
        "the process-wide inference gate bookkeeping was poisoned by a panic; \
         its state was recovered, so retry only after verifying the process remains usable"
    )]
    Poisoned,
    /// Too many callers were already waiting.
    #[error("the process-wide inference queue is full ({max_waiters} waiting callers allowed)")]
    QueueFull {
        /// Configured queue bound.
        max_waiters: usize,
    },
    /// Inference did not begin before the configured deadline.
    #[error("timed out after {waited:?} waiting for process-wide inference admission")]
    TimedOut {
        /// Configured maximum wait.
        waited: Duration,
    },
}

#[derive(Default)]
struct GateState {
    occupied: bool,
    next_ticket: u64,
    queue: VecDeque<u64>,
}

struct InferenceGate {
    state: Mutex<GateState>,
    available: Condvar,
    poison_reported: AtomicBool,
}

impl InferenceGate {
    const fn new() -> Self {
        Self {
            state: Mutex::new(GateState {
                occupied: false,
                next_ticket: 0,
                queue: VecDeque::new(),
            }),
            available: Condvar::new(),
            poison_reported: AtomicBool::new(false),
        }
    }

    fn acquire(
        &self,
        config: InferenceGateConfig,
    ) -> Result<InferencePermit<'_>, InferenceGateError> {
        let deadline = config
            .deadline()
            .map_err(|source| InferenceGateError::InvalidConfiguration { source })?;
        let mut state = self.lock_state()?;
        let can_enter_now = !state.occupied && state.queue.is_empty();
        if !can_enter_now && state.queue.len() >= config.max_waiters {
            return Err(InferenceGateError::QueueFull {
                max_waiters: config.max_waiters,
            });
        }

        let ticket = state.next_ticket;
        state.next_ticket = state.next_ticket.wrapping_add(1);
        state.queue.push_back(ticket);

        loop {
            if !state.occupied && state.queue.front() == Some(&ticket) {
                state.queue.pop_front();
                state.occupied = true;
                drop(state);
                return Ok(InferencePermit { gate: self });
            }

            let now = Instant::now();
            if now >= deadline {
                remove_ticket(&mut state.queue, ticket);
                drop(state);
                self.available.notify_all();
                return Err(InferenceGateError::TimedOut {
                    waited: config.wait_timeout,
                });
            }
            let remaining = deadline.saturating_duration_since(now);
            let (next_state, wait_result) = self.wait_for_admission(state, remaining)?;
            state = next_state;
            if wait_result.timed_out() && (state.occupied || state.queue.front() != Some(&ticket)) {
                remove_ticket(&mut state.queue, ticket);
                drop(state);
                self.available.notify_all();
                return Err(InferenceGateError::TimedOut {
                    waited: config.wait_timeout,
                });
            }
        }
    }

    fn try_acquire(&self) -> Result<Option<InferencePermit<'_>>, InferenceGateError> {
        let mut state = self.lock_state()?;
        if state.occupied || !state.queue.is_empty() {
            return Ok(None);
        }
        state.occupied = true;
        drop(state);
        Ok(Some(InferencePermit { gate: self }))
    }

    fn lock_state(&self) -> Result<MutexGuard<'_, GateState>, InferenceGateError> {
        match self.state.lock() {
            Ok(state) => Ok(state),
            Err(poisoned) => self.recover_poison(poisoned.into_inner()),
        }
    }

    fn wait_for_admission<'a>(
        &self,
        state: MutexGuard<'a, GateState>,
        timeout: Duration,
    ) -> Result<(MutexGuard<'a, GateState>, std::sync::WaitTimeoutResult), InferenceGateError> {
        match self.available.wait_timeout(state, timeout) {
            Ok(result) => Ok(result),
            Err(poisoned) => {
                let (state, wait_result) = poisoned.into_inner();
                self.recover_poison(state).map(|state| (state, wait_result))
            }
        }
    }

    fn recover_poison<'a>(
        &self,
        state: MutexGuard<'a, GateState>,
    ) -> Result<MutexGuard<'a, GateState>, InferenceGateError> {
        if self.poison_reported.swap(true, Ordering::AcqRel) {
            Ok(state)
        } else {
            drop(state);
            Err(InferenceGateError::Poisoned)
        }
    }
}

impl InferenceGateConfig {
    fn deadline(self) -> Result<Instant, InferenceGateConfigError> {
        if self.wait_timeout.is_zero() {
            return Err(InferenceGateConfigError::ZeroTimeout);
        }
        Instant::now().checked_add(self.wait_timeout).ok_or(
            InferenceGateConfigError::TimeoutTooLarge {
                wait_timeout: self.wait_timeout,
            },
        )
    }
}

fn remove_ticket(queue: &mut VecDeque<u64>, ticket: u64) {
    if let Some(index) = queue.iter().position(|queued| *queued == ticket) {
        queue.remove(index);
    }
}

/// Waits for bounded, FIFO process-wide admission to Candle model execution.
pub fn acquire_inference_permit() -> Result<InferencePermit<'static>, InferenceGateError> {
    let config = *PROCESS_INFERENCE_CONFIG.get_or_init(InferenceGateConfig::default);
    PROCESS_INFERENCE_GATE.acquire(config)
}

/// Attempts to enter inference immediately without joining the wait queue.
pub fn try_acquire_inference() -> Result<Option<InferencePermit<'static>>, InferenceGateError> {
    PROCESS_INFERENCE_GATE.try_acquire()
}
