use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use super::cli::{Dataset, SpeedResult};

pub(super) const MODEL: &str = "bge-base-en-v1.5";
pub(super) const REVISION: &str = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a";
pub(super) const QUERY: &str = "where is the red book?";

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Manifest {
    pub schema_version: u32,
    pub model: String,
    pub revision: String,
    pub tokenizer_sha256: String,
    pub query_prefix: String,
    pub query_tokens: usize,
    pub fixtures: Vec<Fixture>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Fixture {
    pub name: String,
    pub repetitions: usize,
    pub raw_bytes: usize,
    pub content_tokens: usize,
    pub complete_tokens: usize,
    pub sha256: String,
}

pub(super) struct Fixtures {
    pub manifest: Manifest,
    pub texts: Vec<String>,
    pub manifest_sha256: String,
}

pub(super) fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(super) fn text(fixture: &Fixture) -> String {
    if fixture.repetitions == 0 {
        QUERY.to_string()
    } else {
        vec!["red"; fixture.repetitions].join(" ")
    }
}

pub(super) fn check(
    fixture: &Fixture,
    text: &str,
    content: usize,
    complete: usize,
) -> SpeedResult<()> {
    if text.len() != fixture.raw_bytes
        || digest(text.as_bytes()) != fixture.sha256
        || content != fixture.content_tokens
        || complete != fixture.complete_tokens
    {
        return Err(format!(
            "speed_fixture_mismatch: {} bytes={} content={} complete={}",
            fixture.name,
            text.len(),
            content,
            complete
        )
        .into());
    }
    Ok(())
}

pub(super) fn load(repository: &Path, model_dir: &Path) -> SpeedResult<Fixtures> {
    let installed: serde_json::Value =
        serde_json::from_slice(&std::fs::read(model_dir.join("manifest.json"))?)?;
    let specification: serde_json::Value = serde_json::from_slice(&std::fs::read(
        repository.join("certification/specs/bge-base-en-v1.5.json"),
    )?)?;
    check_declarations(&installed, &specification)?;
    let bytes = std::fs::read(repository.join("xtask/speed-fixtures.json"))?;
    let manifest: Manifest = serde_json::from_slice(&bytes)?;
    let registry = tessera::models::registry::get_model(MODEL).ok_or("speed_model_missing")?;
    let prompts = registry.prompts.ok_or("speed_prompts_missing")?;
    if manifest.schema_version != 1
        || manifest.model != MODEL
        || manifest.revision != REVISION
        || registry.revision != Some(REVISION)
        || manifest.query_prefix != prompts.query
        || !prompts.document.is_empty()
        || manifest.query_tokens != 16
        || manifest.fixtures.len() != 5
    {
        return Err("speed_fixture_identity_mismatch".into());
    }
    let tokenizer_bytes = std::fs::read(model_dir.join("tokenizer.json"))?;
    if digest(&tokenizer_bytes) != manifest.tokenizer_sha256 {
        return Err("speed_tokenizer_hash_mismatch".into());
    }
    let mut tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
        .map_err(|error| format!("speed_tokenizer: {error}"))?;
    tokenizer
        .with_truncation(None)
        .map_err(|error| error.to_string())?;
    tokenizer.with_padding(None);
    let mut texts = Vec::new();
    let definitions = [
        ("short", 30, 30),
        ("medium", 126, 126),
        ("full", 510, 510),
        ("long", 4096, 4096),
        ("query", 0, 6),
    ];
    for (fixture, (name, repetitions, content_tokens)) in manifest.fixtures.iter().zip(definitions)
    {
        if fixture.name != name
            || fixture.repetitions != repetitions
            || fixture.content_tokens != content_tokens
            || fixture.complete_tokens != content_tokens + 2
        {
            return Err("speed_fixture_definition_mismatch".into());
        }
        let input = text(fixture);
        let content = tokenizer
            .encode(input.as_str(), false)
            .map_err(|e| e.to_string())?
            .len();
        let complete = tokenizer
            .encode(input.as_str(), true)
            .map_err(|e| e.to_string())?
            .len();
        check(fixture, &input, content, complete)?;
        texts.push(input);
    }
    let joined = format!("{}{QUERY}", manifest.query_prefix);
    let query = tokenizer.encode(joined, true).map_err(|e| e.to_string())?;
    if query.len() != manifest.query_tokens {
        return Err(format!(
            "speed_query_tokens: observed={}, expected={}",
            query.len(),
            manifest.query_tokens
        )
        .into());
    }
    Ok(Fixtures {
        manifest,
        texts,
        manifest_sha256: digest(&bytes),
    })
}

pub(super) fn check_declarations(
    installed: &serde_json::Value,
    specification: &serde_json::Value,
) -> SpeedResult<()> {
    let artifacts = specification
        .get("artifacts")
        .ok_or("speed_pinned_artifacts_missing")?;
    let expected = serde_json::json!({"schema_version":1,"model":{"id":MODEL,"repository":"BAAI/bge-base-en-v1.5","revision":REVISION},"artifacts":artifacts});
    if installed != &expected {
        return Err("speed_installed_declarations_mismatch".into());
    }
    Ok(())
}

impl Fixtures {
    pub fn input(&self, name: &str) -> SpeedResult<&str> {
        self.manifest
            .fixtures
            .iter()
            .position(|entry| entry.name == name)
            .and_then(|position| self.texts.get(position))
            .map(String::as_str)
            .ok_or_else(|| format!("speed_fixture_missing: {name}").into())
    }

    pub fn job(&self, dataset: Dataset) -> SpeedResult<(Vec<&str>, Vec<usize>)> {
        let short = self.input("short")?;
        let medium = self.input("medium")?;
        let full = self.input("full")?;
        match dataset {
            Dataset::Short => Ok((vec![short; 2000], vec![32; 2000])),
            Dataset::Full => Ok((vec![full; 500], vec![512; 500])),
            Dataset::Mixed => Ok((
                (0..2000)
                    .map(|i| [short, short, short, medium, full][i % 5])
                    .collect(),
                (0..2000).map(|i| [32, 32, 32, 128, 512][i % 5]).collect(),
            )),
        }
    }
}
