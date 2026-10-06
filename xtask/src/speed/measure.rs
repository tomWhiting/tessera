use std::path::Path;
use std::time::Instant;

use sha2::{Digest, Sha256};
use tessera::{
    ContextWindowConfig, Device, EmbeddingOutcome, ModelDType, ResourcePolicy, Role, TesseraDense,
    WindowEmbeddingOutcome,
};

use super::cli::{Dataset, Journey, Options, Route, SpeedResult};
use super::fixtures::{self, Fixtures, MODEL};
use super::record::{self, Counts, Recorder};

const fn policy() -> ResourcePolicy {
    ResourcePolicy::new(512, 4, 2048, 2_147_483_648)
        .with_max_input_bytes_per_sequence(65_536)
        .with_max_attention_cells(1_048_576)
        .with_max_job_items(1024)
        .with_max_job_input_bytes(67_108_864)
        .with_max_output_bytes(67_108_864)
        .with_max_activation_bytes(1_073_741_824)
}

fn build(model_dir: &Path, batch_size: usize) -> SpeedResult<TesseraDense> {
    Ok(TesseraDense::builder()
        .model(MODEL)
        .model_dir(model_dir)
        .device(Device::Cpu)
        .dtype(ModelDType::F32)
        .batch_size(batch_size)
        .resource_policy(policy())
        .build()?)
}

fn consume(values: impl Iterator<Item = f32>, hash: &mut Sha256) -> SpeedResult<()> {
    let mut dimensions = 0;
    for value in values {
        if !value.is_finite() {
            return Err("speed_output_non_finite".into());
        }
        hash.update(value.to_le_bytes());
        dimensions += 1;
    }
    if dimensions != 768 {
        return Err(format!("speed_output_dimensions: {dimensions}").into());
    }
    Ok(())
}

fn query(model: &TesseraDense, fixtures: &Fixtures, hash: &mut Sha256) -> SpeedResult<()> {
    let EmbeddingOutcome::Embedded(output) =
        model.encode_outcome(fixtures.input("query")?, Some(Role::Query))?
    else {
        return Err("speed_query_refused".into());
    };
    if output.tokens_total() != fixtures.manifest.query_tokens {
        return Err("speed_query_count_mismatch".into());
    }
    consume(output.values().iter().copied(), hash)
}

fn query_result(model: &TesseraDense, fixtures: &Fixtures) -> SpeedResult<String> {
    let mut hash = Sha256::new();
    query(model, fixtures, &mut hash)?;
    Ok(format!("{:x}", hash.finalize()))
}

fn query_counts(fixtures: &Fixtures) -> Counts {
    let mut counts = record::counts(&[fixtures.manifest.query_tokens], Route::Outcomes);
    counts.source_content_tokens = 6;
    counts
}

fn backfill(
    model: &TesseraDense,
    texts: &[&str],
    lengths: &[usize],
    route: Route,
) -> SpeedResult<String> {
    let mut hash = Sha256::new();
    for (chunk, counts) in texts.chunks(4).zip(lengths.chunks(4)) {
        match route {
            Route::Batch => {
                let outputs = model.encode_batch(chunk)?;
                if outputs.len() != chunk.len() {
                    return Err("speed_batch_item_count_mismatch".into());
                }
                for (output, input) in outputs.iter().zip(chunk) {
                    if output.text() != *input {
                        return Err("speed_batch_order_mismatch".into());
                    }
                    consume(output.values().iter().copied(), &mut hash)?;
                }
            }
            Route::Outcomes => {
                let outputs = model.encode_batch_outcomes(chunk, Some(Role::Document))?;
                if outputs.len() != chunk.len() {
                    return Err("speed_outcome_item_count_mismatch".into());
                }
                for (output, &count) in outputs.iter().zip(counts) {
                    let EmbeddingOutcome::Embedded(output) = output else {
                        return Err("speed_document_refused".into());
                    };
                    if output.tokens_total() != count {
                        return Err("speed_document_count_mismatch".into());
                    }
                    consume(output.values().iter().copied(), &mut hash)?;
                }
            }
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn windows(model: &TesseraDense, text: &str, aggregate: bool) -> SpeedResult<(String, Counts)> {
    let mut counts = record::counts(
        &[512, 512, 512, 512, 512, 512, 512, 512, 512, 84],
        Route::Outcomes,
    );
    counts.items = 1;
    counts.windows = 10;
    counts.source_content_tokens = 4096;
    let mut hash = Sha256::new();
    if aggregate {
        let output = model.encode_windowed(text, ContextWindowConfig::new(512, 64))?;
        consume(output.values().iter().copied(), &mut hash)?;
        counts.raw_output_bytes = 768 * 4;
        counts.vectors = 1;
        counts.admission_token_rows_derived = counts.real_tokens;
        counts.admission_squared_lengths_derived = counts.real_squared_lengths;
    } else {
        let WindowEmbeddingOutcome::Embedded(output) = model.encode_windows(
            text,
            Some(Role::Document),
            Some(ContextWindowConfig::new(512, 64)),
        )?
        else {
            return Err("speed_windows_refused".into());
        };
        if output.tokens_total() != 4096 || output.windows().len() != 10 {
            return Err("speed_window_count_mismatch".into());
        }
        let mut previous_end = 0;
        for (i, window) in output.windows().iter().enumerate() {
            if window.tokens() != if i == 9 { 84 } else { 512 }
                || window.byte_start() > previous_end
                || window.byte_end() <= window.byte_start()
                || !text.is_char_boundary(window.byte_start())
                || !text.is_char_boundary(window.byte_end())
            {
                return Err("speed_window_span_or_tokens_mismatch".into());
            }
            previous_end = window.byte_end();
            consume(window.values().iter().copied(), &mut hash)?;
        }
        if previous_end != text.len() {
            return Err("speed_window_coverage_mismatch".into());
        }
    }
    Ok((format!("{:x}", hash.finalize()), counts))
}

pub(super) fn run(repository: &Path, options: &Options) -> SpeedResult<()> {
    let fixtures = fixtures::load(repository, &options.model_dir)?;
    let manifest = fixtures::digest(&std::fs::read(options.model_dir.join("manifest.json"))?);
    let context = record::context(
        repository,
        fixtures.manifest_sha256.clone(),
        manifest,
        options.threads,
    )?;
    let mut recorder = Recorder::new(&options.output, context)?;
    if matches!(options.journey, Journey::Startup) {
        let start = Instant::now();
        let model = build(&options.model_dir, 1)?;
        let hash = query_result(&model, &fixtures)?;
        recorder.write(
            ("J4", "outcomes", "query", 0),
            start.elapsed(),
            &query_counts(&fixtures),
            hash,
        )?;
        return recorder.finish();
    }
    let model = build(
        &options.model_dir,
        if matches!(options.journey, Journey::Query) {
            1
        } else {
            4
        },
    )?;
    match options.journey {
        Journey::Backfill => {
            let (texts, lengths) = fixtures.job(options.dataset)?;
            for _ in 0..4 {
                backfill(&model, &texts[..4], &lengths[..4], options.route)?;
            }
            let counts = record::counts(&lengths, options.route);
            let route = match options.route {
                Route::Batch => "batch",
                Route::Outcomes => "outcomes",
            };
            let dataset = match options.dataset {
                Dataset::Mixed => "mixed",
                Dataset::Short => "short",
                Dataset::Full => "full",
            };
            for observation in 0..3 {
                let start = Instant::now();
                let hash = backfill(&model, &texts, &lengths, options.route)?;
                recorder.write(
                    ("J1", route, dataset, observation),
                    start.elapsed(),
                    &counts,
                    hash,
                )?;
            }
        }
        Journey::Query => {
            query(&model, &fixtures, &mut Sha256::new())?;
            for observation in 0..100 {
                let start = Instant::now();
                let hash = query_result(&model, &fixtures)?;
                recorder.write(
                    ("J2", "outcomes", "query", observation),
                    start.elapsed(),
                    &query_counts(&fixtures),
                    hash,
                )?;
            }
        }
        Journey::Windows | Journey::Aggregate => {
            let text = fixtures.input("long")?;
            let aggregate = matches!(options.journey, Journey::Aggregate);
            windows(&model, text, aggregate)?;
            for observation in 0..20 {
                let start = Instant::now();
                let (hash, counts) = windows(&model, text, aggregate)?;
                recorder.write(
                    (
                        "J3",
                        if aggregate { "aggregate" } else { "windows" },
                        "long",
                        observation,
                    ),
                    start.elapsed(),
                    &counts,
                    hash,
                )?;
            }
        }
        Journey::Startup => return Err("speed_internal_startup_dispatch".into()),
    }
    recorder.finish()
}
