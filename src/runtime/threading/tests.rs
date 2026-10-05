use std::num::NonZeroUsize;

use super::{cap_threads, configure_cpu_threads};

#[cfg(all(target_os = "macos", feature = "accelerate"))]
mod accelerate;

#[test]
fn environment_overrides_are_capped_but_lower_values_survive() {
    let ceiling = NonZeroUsize::new(2).unwrap();

    assert_eq!(cap_threads(8, ceiling), 2);
    assert_eq!(cap_threads(1, ceiling), 1);
}

#[test]
fn empty_environment_single_thread_reaches_candle_and_rayon() {
    std::env::remove_var("RAYON_NUM_THREADS");
    std::env::remove_var("CANDLE_NUM_THREADS");

    let configured = configure_cpu_threads(1).unwrap();
    assert_eq!(configured.rayon_threads().get(), 1);
    assert_eq!(configured.candle_threads().get(), 1);
    assert_eq!(std::env::var("RAYON_NUM_THREADS").unwrap(), "1");
    assert_eq!(std::env::var("CANDLE_NUM_THREADS").unwrap(), "1");
    assert_eq!(candle_core::utils::get_num_threads(), 1);
    assert_eq!(tokenizers::utils::parallelism::current_num_threads(), 1);
}

#[test]
fn independent_overrides_are_capped_and_first_call_wins() {
    std::env::set_var("RAYON_NUM_THREADS", "1");
    std::env::set_var("CANDLE_NUM_THREADS", "8");

    let configured = configure_cpu_threads(2).unwrap();
    assert_eq!(configured.rayon_threads().get(), 1);
    assert_eq!(configured.candle_threads().get(), 2);
    assert_eq!(std::env::var("RAYON_NUM_THREADS").unwrap(), "1");
    assert_eq!(std::env::var("CANDLE_NUM_THREADS").unwrap(), "2");
    assert_eq!(configure_cpu_threads(8).unwrap(), configured);
}

#[test]
fn one_override_is_copied_to_the_other_pool() {
    std::env::remove_var("RAYON_NUM_THREADS");
    std::env::set_var("CANDLE_NUM_THREADS", "1");

    let configured = configure_cpu_threads(2).unwrap();
    assert_eq!(configured.rayon_threads().get(), 1);
    assert_eq!(configured.candle_threads().get(), 1);
    assert_eq!(std::env::var("RAYON_NUM_THREADS").unwrap(), "1");
    assert_eq!(std::env::var("CANDLE_NUM_THREADS").unwrap(), "1");
}
