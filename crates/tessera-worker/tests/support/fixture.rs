use std::fs;

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

pub fn installed() -> TempDir {
    installed_with_bias(1.0)
}

fn configuration() -> Vec<u8> {
    serde_json::to_vec(&json!({
        "model_type": "bert", "vocab_size": 5, "hidden_size": 768,
        "num_hidden_layers": 1, "num_attention_heads": 12, "intermediate_size": 4,
        "hidden_act": "gelu", "hidden_dropout_prob": 0.0, "max_position_embeddings": 32,
        "type_vocab_size": 2, "initializer_range": 0.02, "layer_norm_eps": 1e-12,
        "pad_token_id": 1, "position_embedding_type": "absolute", "use_cache": true
    }))
    .unwrap()
}

fn tokenizer() -> Vec<u8> {
    serde_json::to_vec(&json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken":{"id":"[START]","type_id":0}},
                {"Sequence":{"id":"A","type_id":0}},
                {"SpecialToken":{"id":"[END]","type_id":0}}],
            "pair": [{"Sequence":{"id":"A","type_id":0}},
                {"Sequence":{"id":"B","type_id":1}}],
            "special_tokens": {
                "[START]":{"id":"[START]","ids":[0],"tokens":["[START]"]},
                "[END]":{"id":"[END]","ids":[0],"tokens":["[END]"]}
            }
        },
        "decoder": null,
        "model": {"type":"WordLevel","vocab":{"[UNK]":0,"[PAD]":1,"one":2,"two":3,"three":4},
            "unk_token":"[UNK]"}
    }))
    .unwrap()
}

fn weights(bias: f32) -> Vec<u8> {
    let mut header = Map::new();
    let mut data = Vec::new();
    let mut tensor = |name: &str, shape: &[usize], value: f32| {
        let begin = data.len();
        for _ in 0..shape.iter().product::<usize>() {
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            name.into(),
            json!({"dtype":"F32","shape":shape,
            "data_offsets":[begin,data.len()]}),
        );
    };
    for (name, shape) in [
        ("word_embeddings", vec![5, 768]),
        ("position_embeddings", vec![32, 768]),
        ("token_type_embeddings", vec![2, 768]),
    ] {
        tensor(&format!("embeddings.{name}.weight"), &shape, 0.0);
    }
    for name in [
        "embeddings.LayerNorm",
        "encoder.layer.0.attention.output.LayerNorm",
        "encoder.layer.0.output.LayerNorm",
    ] {
        tensor(&format!("{name}.weight"), &[768], 1.0);
        tensor(&format!("{name}.bias"), &[768], bias);
    }
    for name in ["query", "key", "value"] {
        tensor(
            &format!("encoder.layer.0.attention.self.{name}.weight"),
            &[768, 768],
            0.0,
        );
        tensor(
            &format!("encoder.layer.0.attention.self.{name}.bias"),
            &[768],
            0.0,
        );
    }
    for (name, shape, bias) in [
        ("attention.output.dense", vec![768, 768], 768),
        ("intermediate.dense", vec![4, 768], 4),
        ("output.dense", vec![768, 4], 768),
    ] {
        tensor(&format!("encoder.layer.0.{name}.weight"), &shape, 0.0);
        tensor(&format!("encoder.layer.0.{name}.bias"), &[bias], 0.0);
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut weights = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
    weights.extend_from_slice(&header);
    weights.extend_from_slice(&data);
    weights
}

pub fn installed_with_bias(bias: f32) -> TempDir {
    installed_with_normalizer(bias, None)
}

pub fn installed_with_normalizer(bias: f32, normalizer: Option<Value>) -> TempDir {
    let mut tokenizer: Value = serde_json::from_slice(&tokenizer()).unwrap();
    if let Some(normalizer) = normalizer {
        tokenizer["normalizer"] = normalizer;
    }
    installed_with_tokenizer(bias, serde_json::to_vec(&tokenizer).unwrap())
}

fn installed_with_tokenizer(bias: f32, tokenizer: Vec<u8>) -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    let config = configuration();
    let weights = weights(bias);
    let mut artifacts = Vec::<Value>::new();
    for (name, bytes) in [
        ("config.json", config),
        ("tokenizer.json", tokenizer),
        ("model.safetensors", weights),
    ] {
        fs::write(dir.path().join(name), &bytes).unwrap();
        artifacts.push(json!({"path":name,"size_bytes":bytes.len(),
            "sha256":format!("{:x}",Sha256::digest(&bytes))}));
    }
    let manifest = json!({"schema_version":1,
        "model":{"id":"bge-base-en-v1.5","repository":"BAAI/bge-base-en-v1.5",
            "revision":"a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"},"artifacts":artifacts});
    fs::write(
        dir.path().join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    dir
}
