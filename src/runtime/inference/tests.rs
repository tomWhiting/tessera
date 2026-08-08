use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use super::{InferenceGate, InferenceGateConfig, InferenceGateConfigError, InferenceGateError};

fn test_config(max_waiters: usize, wait_timeout: Duration) -> InferenceGateConfig {
    InferenceGateConfig::new(max_waiters, wait_timeout)
}

#[test]
fn permit_drop_releases_admission() {
    let gate = InferenceGate::new();
    let first = gate
        .acquire(test_config(1, Duration::from_secs(1)))
        .expect("fresh gate should admit inference");

    assert!(gate
        .try_acquire()
        .expect("healthy gate should not error")
        .is_none());
    drop(first);
    assert!(gate
        .try_acquire()
        .expect("dropping the permit should release admission")
        .is_some());
}

#[test]
fn concurrent_caller_waits_for_the_live_permit() {
    let gate = Arc::new(InferenceGate::new());
    let config = test_config(1, Duration::from_secs(1));
    let first = gate
        .acquire(config)
        .expect("fresh gate should admit inference");
    let waiting_gate = Arc::clone(&gate);
    let (started_tx, started_rx) = mpsc::channel();
    let (admitted_tx, admitted_rx) = mpsc::channel();

    let waiter = std::thread::spawn(move || {
        started_tx
            .send(())
            .expect("test receiver should remain live");
        let _permit = waiting_gate
            .acquire(config)
            .expect("waiter should enter after the first permit drops");
        admitted_tx
            .send(())
            .expect("test receiver should remain live");
    });

    started_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("waiter should start");
    assert!(admitted_rx.recv_timeout(Duration::from_millis(50)).is_err());
    drop(first);
    admitted_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("waiter should be admitted after permit drop");
    waiter.join().expect("waiter should not panic");
}

#[test]
fn queue_limit_rejects_excess_waiters() {
    let gate = Arc::new(InferenceGate::new());
    let config = test_config(1, Duration::from_secs(1));
    let first = gate
        .acquire(config)
        .expect("fresh gate should admit inference");
    let waiting_gate = Arc::clone(&gate);
    let (queued_tx, queued_rx) = mpsc::channel();

    let waiter = std::thread::spawn(move || {
        queued_tx.send(()).expect("receiver remains live");
        waiting_gate.acquire(config).map(drop)
    });
    queued_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("waiter should start");

    let queue_deadline = Instant::now() + Duration::from_secs(1);
    loop {
        if gate.state.lock().expect("healthy state").queue.len() == 1 {
            break;
        }
        assert!(
            Instant::now() < queue_deadline,
            "waiter should join the queue"
        );
        std::thread::yield_now();
    }
    let Err(error) = gate.acquire(test_config(1, Duration::from_millis(1))) else {
        panic!("excess waiter must not be admitted");
    };
    assert_eq!(error, InferenceGateError::QueueFull { max_waiters: 1 });
    drop(first);
    waiter
        .join()
        .expect("waiter should not panic")
        .expect("queued waiter should eventually enter");
}

#[test]
fn timed_out_ticket_is_removed_for_the_next_caller() {
    let gate = InferenceGate::new();
    let first = gate
        .acquire(test_config(1, Duration::from_secs(1)))
        .expect("fresh gate should admit inference");
    let Err(error) = gate.acquire(test_config(1, Duration::from_millis(5))) else {
        panic!("occupied gate should time out");
    };
    assert!(matches!(error, InferenceGateError::TimedOut { .. }));
    drop(first);
    assert!(gate.try_acquire().expect("gate should recover").is_some());
}

#[test]
fn unrepresentable_deadline_does_not_mutate_or_wedge_gate() {
    let gate = InferenceGate::new();
    let Err(error) = gate.acquire(test_config(1, Duration::MAX)) else {
        panic!("an unrepresentable deadline must be rejected");
    };
    assert_eq!(
        error,
        InferenceGateError::InvalidConfiguration {
            source: InferenceGateConfigError::TimeoutTooLarge {
                wait_timeout: Duration::MAX,
            },
        }
    );
    {
        let state = gate
            .state
            .lock()
            .expect("invalid input must not poison state");
        assert!(!state.occupied);
        assert!(state.queue.is_empty());
        assert_eq!(state.next_ticket, 0);
        drop(state);
    }

    let permit = gate
        .acquire(test_config(1, Duration::from_secs(1)))
        .expect("gate must remain usable after invalid input");
    drop(permit);
    assert!(gate
        .try_acquire()
        .expect("gate should remain healthy")
        .is_some());
}

#[test]
fn zero_timeout_is_rejected_before_enqueue() {
    let gate = InferenceGate::new();
    let Err(error) = gate.acquire(test_config(1, Duration::ZERO)) else {
        panic!("zero timeout must be rejected");
    };
    assert_eq!(
        error,
        InferenceGateError::InvalidConfiguration {
            source: InferenceGateConfigError::ZeroTimeout,
        }
    );
    let state = gate
        .state
        .lock()
        .expect("invalid input must not poison state");
    assert!(state.queue.is_empty());
    assert_eq!(state.next_ticket, 0);
    drop(state);
}

#[test]
fn poison_is_reported_once_and_then_recovered() {
    let gate = Arc::new(InferenceGate::new());
    let poisoning_gate = Arc::clone(&gate);
    let config = test_config(1, Duration::from_secs(1));

    let panic_result = std::thread::spawn(move || {
        let _state = poisoning_gate
            .state
            .lock()
            .expect("fresh bookkeeping lock should be healthy");
        panic!("intentional gate poison for regression coverage");
    })
    .join();
    assert!(panic_result.is_err());

    let Err(error) = gate.acquire(config) else {
        panic!("the first acquisition after a panic must report poison");
    };
    assert_eq!(error, InferenceGateError::Poisoned);
    assert!(gate.acquire(config).is_ok());
}
