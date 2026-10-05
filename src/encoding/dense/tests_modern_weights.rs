use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::modernbert::{Config, ModernBert};

use super::{assert_tensors_close, BertVariant, CandleDenseEncoder};

const CONFIG: &str = r#"{
    "vocab_size":16,"hidden_size":4,"num_hidden_layers":1,"num_attention_heads":2,
    "intermediate_size":8,"max_position_embeddings":16,"layer_norm_eps":0.00001,
    "pad_token_id":0,"global_attn_every_n_layers":2,"global_rope_theta":160000.0,
    "local_attention":4,"local_rope_theta":10000.0
}"#;

fn tiny_weights(prefixed: bool) -> Result<(tempfile::TempDir, PathBuf, Tensor)> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("model.safetensors");
    let variables = VarMap::new();
    let builder = VarBuilder::from_varmap(&variables, DType::F32, &Device::Cpu);
    let model = ModernBert::load(builder, &serde_json::from_str::<Config>(CONFIG)?)?;
    let inputs = Tensor::new(&[[1_i64, 2, 3]], &Device::Cpu)?;
    let expected = model.forward(&inputs, &inputs.ones_like()?)?;
    let tensors: HashMap<String, Tensor> = variables
        .data()
        .lock()
        .unwrap()
        .iter()
        .map(|(name, variable)| {
            let name = if prefixed {
                name.clone()
            } else {
                name.strip_prefix("model.").unwrap().to_string()
            };
            (name, variable.as_tensor().clone())
        })
        .collect();
    candle_core::safetensors::save(&tensors, &path)?;
    Ok((directory, path, expected))
}

fn check_file(prefixed: bool) -> Result<()> {
    let (directory, path, expected) = tiny_weights(prefixed)?;
    let builder =
        VarBuilder::from_buffered_safetensors(std::fs::read(path)?, DType::F32, &Device::Cpu)?;
    let model = CandleDenseEncoder::load_model(CONFIG, builder, "modernbert", 16)?;
    assert!(matches!(model, BertVariant::ModernBert(_)));
    let inputs = Tensor::new(&[[1_i64, 2, 3]], &Device::Cpu)?;
    let actual = model.forward(&inputs, &inputs.ones_like()?)?;
    assert_tensors_close(&actual, &expected, 0.000_001)?;
    directory.close()?;
    Ok(())
}

#[test]
fn bare_modernbert_safetensors_load_and_forward() -> Result<()> {
    check_file(false)
}

#[test]
fn prefixed_modernbert_safetensors_keep_their_output() -> Result<()> {
    check_file(true)
}

#[test]
fn unrelated_safetensors_refuse_both_names_before_model_construction() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("model.safetensors");
    let tensor = Tensor::new(&[1.0_f32], &Device::Cpu)?;
    candle_core::safetensors::save(&HashMap::from([("unrelated.weight", tensor)]), &path)?;
    let builder =
        VarBuilder::from_buffered_safetensors(std::fs::read(path)?, DType::F32, &Device::Cpu)?;
    let error = CandleDenseEncoder::load_model("{}", builder, "modernbert", 16)
        .err()
        .ok_or_else(|| anyhow::anyhow!("unrelated weights were accepted"))?
        .to_string();
    assert!(error.contains("modernbert_weight_name_missing"), "{error}");
    assert!(
        error.contains("model.embeddings.tok_embeddings.weight"),
        "{error}"
    );
    assert!(
        error.contains("embeddings.tok_embeddings.weight"),
        "{error}"
    );
    directory.close()?;
    Ok(())
}
