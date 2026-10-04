mod artifacts;
mod child;
mod cli;
mod evidence;
mod install;
#[cfg(test)]
mod measure;
mod process;
mod readiness;
mod reference;
mod reference_compare;
mod smoke_math;
mod smoke_observation;
mod spec;
mod vision_smoke;

pub(crate) use cli::run;
