//! Installed dense embedding worker with a bounded Rust heap and shared messages.

use std::process::ExitCode;

mod engine;
mod failure;
mod policy;
mod process;
mod session;

#[global_allocator]
static ALLOCATOR: haem_worker::allocator::Counting = haem_worker::allocator::Counting;

fn main() -> ExitCode {
    let resources = match process::prepare() {
        Ok(resources) => resources,
        Err(error) => {
            eprintln!("worker process setup failed: {error}");
            return ExitCode::from(1);
        }
    };
    let result = session::run(
        &mut std::io::stdin().lock(),
        &mut std::io::stdout().lock(),
        &resources,
    );
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => {
            eprintln!("worker frame output failed: {error}");
            ExitCode::from(2)
        }
    }
}
