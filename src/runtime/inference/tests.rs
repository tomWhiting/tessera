use std::sync::{mpsc, Arc};
use std::time::Duration;

use super::{InferenceGate, InferenceGateError};

#[test]
fn permit_drop_releases_admission() {
    let gate = InferenceGate::new();
    let first = gate.acquire().expect("fresh gate should admit inference");

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
    let first = gate.acquire().expect("fresh gate should admit inference");
    let waiting_gate = Arc::clone(&gate);
    let (started_tx, started_rx) = mpsc::channel();
    let (admitted_tx, admitted_rx) = mpsc::channel();

    let waiter = std::thread::spawn(move || {
        started_tx
            .send(())
            .expect("test receiver should remain live");
        let _permit = waiting_gate
            .acquire()
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
fn poison_is_reported_once_and_then_recovered() {
    let gate = Arc::new(InferenceGate::new());
    let poisoning_gate = Arc::clone(&gate);

    let panic_result = std::thread::spawn(move || {
        let _state = poisoning_gate
            .state
            .lock()
            .expect("fresh bookkeeping lock should be healthy");
        panic!("intentional gate poison for regression coverage");
    })
    .join();
    assert!(panic_result.is_err());

    let Err(error) = gate.acquire() else {
        panic!("the first acquisition after a panic must report poison");
    };
    assert_eq!(error, InferenceGateError::Poisoned);
    assert!(gate.acquire().is_ok());
}
