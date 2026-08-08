use std::num::NonZeroUsize;

use super::cap_threads;

#[test]
fn environment_overrides_are_capped_but_lower_values_survive() {
    let ceiling = NonZeroUsize::new(2).unwrap();

    assert_eq!(cap_threads(NonZeroUsize::new(8).unwrap(), ceiling).get(), 2);
    assert_eq!(cap_threads(NonZeroUsize::new(1).unwrap(), ceiling).get(), 1);
}
