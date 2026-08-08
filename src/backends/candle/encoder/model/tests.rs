use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

use super::BertVariant;

fn tiny_bert() -> Result<BertVariant> {
    let config = candle_transformers::models::bert::Config {
        vocab_size: 16,
        hidden_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        intermediate_size: 16,
        hidden_act: candle_transformers::models::bert::HiddenAct::Gelu,
        hidden_dropout_prob: 0.0,
        max_position_embeddings: 16,
        type_vocab_size: 2,
        initializer_range: 0.02,
        layer_norm_eps: 1e-12,
        pad_token_id: 0,
        position_embedding_type: candle_transformers::models::bert::PositionEmbeddingType::Absolute,
        use_cache: false,
        classifier_dropout: None,
        model_type: Some("bert".to_string()),
    };
    let variables = VarMap::new();
    let builder = VarBuilder::from_varmap(&variables, DType::F32, &Device::Cpu);
    let model = candle_transformers::models::bert::BertModel::load(builder, &config)?;
    Ok(BertVariant::Bert(model))
}

fn assert_close(left: &Tensor, right: &Tensor, tolerance: f32) -> Result<()> {
    let left = left.flatten_all()?.to_vec1::<f32>()?;
    let right = right.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(left.len(), right.len());
    let max_difference = left
        .iter()
        .zip(right.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_difference <= tolerance,
        "maximum element difference {max_difference} exceeds {tolerance}"
    );
    Ok(())
}

#[test]
fn bert_forward_uses_zero_token_types_and_the_attention_mask() -> Result<()> {
    let variant = tiny_bert()?;
    let token_ids = Tensor::new(&[[1_i64, 2, 3, 0, 0]], &Device::Cpu)?;
    let attention_mask = Tensor::new(&[[1_i64, 1, 1, 0, 0]], &Device::Cpu)?;

    let actual = variant.forward(&token_ids, &attention_mask)?;
    let BertVariant::Bert(model) = &variant else {
        unreachable!("test constructs a BERT variant")
    };
    let token_type_ids = token_ids.zeros_like()?;
    let expected = model.forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

    assert_close(&actual, &expected, 0.0)
}

#[test]
fn mixed_length_batch_matches_sequential_bert_output() -> Result<()> {
    let variant = tiny_bert()?;
    let batch_ids = Tensor::new(&[[1_i64, 2, 3, 0, 0], [1_i64, 2, 3, 4, 5]], &Device::Cpu)?;
    let batch_mask = Tensor::new(&[[1_i64, 1, 1, 0, 0], [1_i64, 1, 1, 1, 1]], &Device::Cpu)?;
    let sequential_ids = Tensor::new(&[[1_i64, 2, 3]], &Device::Cpu)?;
    let sequential_mask = Tensor::new(&[[1_i64, 1, 1]], &Device::Cpu)?;

    let batch = variant.forward(&batch_ids, &batch_mask)?;
    let sequential = variant.forward(&sequential_ids, &sequential_mask)?;
    let short_batch_item = batch.get(0)?.narrow(0, 0, 3)?;

    assert_close(&short_batch_item, &sequential.squeeze(0)?, 1e-5)
}
