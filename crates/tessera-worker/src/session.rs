use std::io::{self, Read, Write};
use std::num::NonZeroU32;

use haem_frames::embedding::{
    check_embed, read_message, read_start, write_message, FailedCode, Message, MINIMUM_FRAME_BYTES,
};

use crate::{engine::Engine, failure::Failure, policy::Budget, process};

fn write(output: &mut impl Write, message: &Message, limit: NonZeroU32) -> io::Result<()> {
    write_message(output, message, limit).map_err(io::Error::other)
}

fn failed(output: &mut impl Write, failure: Failure, limit: NonZeroU32) -> io::Result<bool> {
    write(output, &Message::Failed(failure.into_message()), limit)?;
    Ok(false)
}

pub fn run(
    input: &mut impl Read,
    output: &mut impl Write,
    resources: &haem_worker::setup::Resources,
) -> io::Result<bool> {
    let minimum = NonZeroU32::new(MINIMUM_FRAME_BYTES)
        .ok_or_else(|| io::Error::other("shared minimum frame limit is zero"))?;
    let start = match read_start(input) {
        Ok(Some(start)) => start,
        Ok(None) => return Ok(true),
        Err(error) => return failed(output, error.into(), minimum),
    };
    let Some(limit) = NonZeroU32::new(start.limits.frame_bytes) else {
        return failed(output, Failure::limits("frame_bytes is zero"), minimum);
    };
    let budget = match Budget::new(&start.limits, start.windows) {
        Ok(budget) => budget,
        Err(error) => return failed(output, error, limit),
    };
    process::arm(budget.memory);
    if let Err(error) = tessera::configure_cpu_threads(budget.threads) {
        return failed(output, Failure::limits(format!("threads: {error}")), limit);
    }
    let engine = match Engine::load(&start, &budget, resources) {
        Ok(engine) => engine,
        Err(error) => return failed(output, error, limit),
    };
    write(output, engine.ready_message(), limit)?;
    loop {
        let request = match read_message(input, limit) {
            Ok(Some(Message::Embed(request))) => request,
            Ok(None) => return Ok(true),
            Ok(Some(_)) => {
                return failed(
                    output,
                    Failure::new(
                        FailedCode::EmbedProtocol,
                        "only Embed is accepted after Ready",
                    ),
                    limit,
                )
            }
            Err(error) => return failed(output, error.into(), limit),
        };
        if let Err(error) = check_embed(&request, &start.limits) {
            return failed(output, error.into(), limit);
        }
        let vectors = match engine.encode(&request, &start) {
            Ok(vectors) => vectors,
            Err(error) => return failed(output, error, limit),
        };
        write(output, &Message::Vectors(vectors), limit)?;
    }
}
