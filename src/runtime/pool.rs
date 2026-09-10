//! Keyed model pool with idle-TTL eviction.
//!
//! A loaded model is expensive and the residency ledger refuses to load the same
//! model twice, so a server must share one handle per key and reclaim it when
//! nobody has used it for a while. The pool hands out [`Arc`] handles, records
//! the last use of each key, and [`ModelPool::sweep`] drops every entry that is
//! both idle past the TTL and no longer borrowed. Dropping the last handle
//! releases the residency permit exactly as before; the pool adds only the
//! bookkeeping of who is still holding one.

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Why a pool operation refused.
#[derive(Debug)]
pub enum PoolError {
    /// A previous holder panicked while the pool lock was held.
    Poisoned,
    /// The loader for a missing key failed; the pool is unchanged.
    Load(anyhow::Error),
}

impl fmt::Display for PoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Poisoned => write!(f, "model pool lock was poisoned by a panic"),
            Self::Load(error) => write!(f, "model pool loader failed: {error:#}"),
        }
    }
}

impl std::error::Error for PoolError {}

struct Entry<V> {
    handle: Arc<V>,
    last_used: Instant,
}

/// Idle statistics for one sweep, reported instead of logged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SweepReport {
    /// Entries dropped because they were idle past the TTL and unborrowed.
    pub evicted: usize,
    /// Entries idle past the TTL but still borrowed, so kept.
    pub retained_borrowed: usize,
    /// Entries remaining after the sweep.
    pub remaining: usize,
}

/// A pool of loaded models keyed by whatever identifies a load (model, device, dtype).
pub struct ModelPool<K, V> {
    idle_ttl: Duration,
    entries: Mutex<HashMap<K, Entry<V>>>,
}

impl<K, V> ModelPool<K, V>
where
    K: Eq + Hash + Clone,
{
    /// Creates a pool whose entries are eligible for eviction after `idle_ttl` unused.
    #[must_use]
    pub fn new(idle_ttl: Duration) -> Self {
        Self {
            idle_ttl,
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// The idle TTL this pool was built with.
    #[must_use]
    pub const fn idle_ttl(&self) -> Duration {
        self.idle_ttl
    }

    /// Returns the shared handle for `key`, loading it with `load` on first use.
    ///
    /// Loading runs under the pool lock so two callers racing on one key never
    /// load twice; the residency ledger would refuse the second load anyway.
    pub fn get_or_load<F>(&self, key: K, load: F) -> Result<Arc<V>, PoolError>
    where
        F: FnOnce() -> anyhow::Result<V>,
    {
        self.get_or_load_at(key, Instant::now(), load)
    }

    /// [`Self::get_or_load`] with an explicit clock, for deterministic tests.
    pub fn get_or_load_at<F>(&self, key: K, now: Instant, load: F) -> Result<Arc<V>, PoolError>
    where
        F: FnOnce() -> anyhow::Result<V>,
    {
        let mut entries = self.entries.lock().map_err(|_| PoolError::Poisoned)?;
        if let Some(entry) = entries.get_mut(&key) {
            entry.last_used = now;
            let handle = Arc::clone(&entry.handle);
            drop(entries);
            return Ok(handle);
        }
        let loaded = load();
        let handle = match loaded {
            Ok(value) => Arc::new(value),
            Err(error) => {
                drop(entries);
                return Err(PoolError::Load(error));
            }
        };
        entries.insert(
            key,
            Entry {
                handle: Arc::clone(&handle),
                last_used: now,
            },
        );
        drop(entries);
        Ok(handle)
    }

    /// Drops every entry idle past the TTL that no caller still holds.
    pub fn sweep(&self) -> Result<SweepReport, PoolError> {
        self.sweep_at(Instant::now())
    }

    /// [`Self::sweep`] with an explicit clock, for deterministic tests.
    pub fn sweep_at(&self, now: Instant) -> Result<SweepReport, PoolError> {
        let mut entries = self.entries.lock().map_err(|_| PoolError::Poisoned)?;
        let mut evicted = 0;
        let mut retained_borrowed = 0;
        entries.retain(|_, entry| {
            let idle = now.saturating_duration_since(entry.last_used) >= self.idle_ttl;
            if !idle {
                return true;
            }
            if Arc::strong_count(&entry.handle) > 1 {
                retained_borrowed += 1;
                return true;
            }
            evicted += 1;
            false
        });
        let remaining = entries.len();
        drop(entries);
        Ok(SweepReport {
            evicted,
            retained_borrowed,
            remaining,
        })
    }

    /// Number of loaded entries, borrowed or not.
    pub fn len(&self) -> Result<usize, PoolError> {
        Ok(self.entries.lock().map_err(|_| PoolError::Poisoned)?.len())
    }

    /// Whether the pool holds no entries.
    pub fn is_empty(&self) -> Result<bool, PoolError> {
        Ok(self.len()? == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn pool() -> ModelPool<&'static str, String> {
        ModelPool::new(Duration::from_mins(1))
    }

    #[test]
    fn loads_once_and_shares_the_handle() {
        let pool = pool();
        let loads = AtomicUsize::new(0);
        let load = || {
            loads.fetch_add(1, Ordering::SeqCst);
            Ok("bge".to_string())
        };
        let first = pool.get_or_load("bge", load).expect("first load");
        let second = pool
            .get_or_load("bge", || Ok("unused".to_string()))
            .expect("second use");
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert_eq!(pool.len().expect("len"), 1);
    }

    #[test]
    fn a_failed_load_leaves_the_pool_unchanged() {
        let pool = pool();
        let error = pool
            .get_or_load("bad", || Err(anyhow::anyhow!("weights missing")))
            .expect_err("loader failure must surface");
        assert!(matches!(error, PoolError::Load(_)));
        assert!(error.to_string().contains("weights missing"));
        assert!(pool.is_empty().expect("empty"));
    }

    #[test]
    fn sweep_keeps_borrowed_and_fresh_entries() {
        let pool = pool();
        let start = Instant::now();
        let held = pool
            .get_or_load_at("held", start, || Ok("a".to_string()))
            .expect("load");
        let _fresh = pool
            .get_or_load_at("fresh", start + Duration::from_secs(59), || {
                Ok("b".to_string())
            })
            .expect("load");
        let report = pool
            .sweep_at(start + Duration::from_secs(61))
            .expect("sweep");
        assert_eq!(report.evicted, 0);
        assert_eq!(report.retained_borrowed, 1);
        assert_eq!(report.remaining, 2);
        drop(held);
    }

    #[test]
    fn sweep_evicts_idle_unborrowed_entries() {
        let pool = pool();
        let start = Instant::now();
        pool.get_or_load_at("idle", start, || Ok("a".to_string()))
            .expect("load");
        let early = pool
            .sweep_at(start + Duration::from_secs(30))
            .expect("sweep");
        assert_eq!(early.evicted, 0);
        let late = pool
            .sweep_at(start + Duration::from_mins(1))
            .expect("sweep");
        assert_eq!(late.evicted, 1);
        assert_eq!(late.remaining, 0);
        assert!(pool.is_empty().expect("empty"));
    }

    #[test]
    fn a_use_resets_the_idle_clock() {
        let pool = pool();
        let start = Instant::now();
        pool.get_or_load_at("k", start, || Ok("a".to_string()))
            .expect("load");
        pool.get_or_load_at("k", start + Duration::from_secs(50), || Ok("x".to_string()))
            .expect("reuse");
        let report = pool
            .sweep_at(start + Duration::from_secs(100))
            .expect("sweep");
        assert_eq!(report.evicted, 0);
        let report = pool
            .sweep_at(start + Duration::from_secs(110))
            .expect("sweep");
        assert_eq!(report.evicted, 1);
    }
}
