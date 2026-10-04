use crate::configure_cpu_threads;

#[test]
fn empty_environment_disables_accelerate_parallelism() {
    std::env::remove_var("RAYON_NUM_THREADS");
    std::env::remove_var("CANDLE_NUM_THREADS");
    std::env::remove_var("VECLIB_MAXIMUM_THREADS");

    configure_cpu_threads(8).unwrap();

    assert_eq!(std::env::var("VECLIB_MAXIMUM_THREADS").unwrap(), "1");
}

#[test]
fn caller_value_cannot_enable_accelerate_parallelism() {
    std::env::remove_var("RAYON_NUM_THREADS");
    std::env::remove_var("CANDLE_NUM_THREADS");
    std::env::set_var("VECLIB_MAXIMUM_THREADS", "8");

    configure_cpu_threads(8).unwrap();

    assert_eq!(std::env::var("VECLIB_MAXIMUM_THREADS").unwrap(), "1");
}
