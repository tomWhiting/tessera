use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};

use thiserror::Error;

#[cfg(test)]
mod tests;

static PROCESS_INFERENCE_GATE: InferenceGate = InferenceGate::new();

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
        self.gate.available.notify_one();
    }
}

/// Failure to enter the process-wide inference gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum InferenceGateError {
    /// The gate's short-lived bookkeeping lock was poisoned.
    #[error(
        "the process-wide inference gate bookkeeping was poisoned by a panic; \
         its state was recovered, so retry only after verifying the process remains usable"
    )]
    Poisoned,
}

#[derive(Default)]
struct GateState {
    occupied: bool,
}

struct InferenceGate {
    state: Mutex<GateState>,
    available: Condvar,
    poison_reported: AtomicBool,
}

impl InferenceGate {
    const fn new() -> Self {
        Self {
            state: Mutex::new(GateState { occupied: false }),
            available: Condvar::new(),
            poison_reported: AtomicBool::new(false),
        }
    }

    fn acquire(&self) -> Result<InferencePermit<'_>, InferenceGateError> {
        let mut state = self.lock_state()?;
        while state.occupied {
            state = self.wait_for_admission(state)?;
        }
        state.occupied = true;
        drop(state);
        Ok(InferencePermit { gate: self })
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
    ) -> Result<MutexGuard<'a, GateState>, InferenceGateError> {
        match self.available.wait(state) {
            Ok(state) => Ok(state),
            Err(poisoned) => self.recover_poison(poisoned.into_inner()),
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

    #[cfg(test)]
    fn try_acquire(&self) -> Result<Option<InferencePermit<'_>>, InferenceGateError> {
        let mut state = self.lock_state()?;
        if state.occupied {
            return Ok(None);
        }
        state.occupied = true;
        drop(state);
        Ok(Some(InferencePermit { gate: self }))
    }
}

/// Waits for exclusive process-wide admission to Candle model execution.
pub fn acquire_inference_permit() -> Result<InferencePermit<'static>, InferenceGateError> {
    PROCESS_INFERENCE_GATE.acquire()
}
