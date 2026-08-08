use super::{CandleDenseEncoder, ModelTypeDetector};

const NOMIC_CONFIG: &str = r#"
{
  "architectures": ["NomicBertModel"],
  "model_type": "nomic_bert",
  "vocab_size": 30528,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12,
  "n_inner": 3072,
  "n_positions": 8192,
  "type_vocab_size": 2,
  "layer_norm_epsilon": 1e-12,
  "rotary_emb_fraction": 1.0,
  "rotary_emb_base": 1000.0,
  "rotary_emb_interleaved": false,
  "qkv_proj_bias": false,
  "mlp_fc1_bias": false,
  "mlp_fc2_bias": false,
  "activation_function": "swiglu",
  "prenorm": false
}
"#;

#[test]
fn detects_nomic_bert_before_generic_bert() {
    let detector: ModelTypeDetector =
        serde_json::from_str(r#"{"model_type":"nomic_bert"}"#).unwrap();

    assert_eq!(
        CandleDenseEncoder::detect_model_type(&detector).unwrap(),
        "nomic-bert"
    );
}

#[test]
fn detects_nomic_bert_from_architecture_without_model_type() {
    let detector: ModelTypeDetector =
        serde_json::from_str(r#"{"architectures":["NomicBertModel"]}"#).unwrap();

    assert_eq!(
        CandleDenseEncoder::detect_model_type(&detector).unwrap(),
        "nomic-bert"
    );
}

#[test]
fn stock_nomic_config_accepts_upstream_fields() {
    let config: candle_transformers::models::nomic_bert::Config =
        serde_json::from_str(NOMIC_CONFIG).unwrap();

    assert_eq!(config.n_embd, 768);
    assert_eq!(config.n_positions, 8192);
    assert_eq!(config.model_type.as_deref(), Some("nomic_bert"));
}

#[test]
fn recognizes_upstream_nomic_prefixless_weight_layout() {
    assert!(!CandleDenseEncoder::tensor_names_have_model_prefix([
        "embeddings.word_embeddings.weight",
        "emb_ln.weight",
        "encoder.layers.0.attn.Wqkv.weight",
    ]));
    assert_eq!(
        CandleDenseEncoder::model_weight_prefix(false, "nomic-bert"),
        None
    );
}

#[test]
fn leaves_optional_nomic_model_type_prefix_to_stock_loader() {
    assert!(CandleDenseEncoder::tensor_names_have_model_prefix([
        "nomic_bert.embeddings.word_embeddings.weight"
    ]));
    assert_eq!(
        CandleDenseEncoder::model_weight_prefix(true, "nomic-bert"),
        None
    );
}
