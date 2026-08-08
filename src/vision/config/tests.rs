use super::*;
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

const PINNED_CONFIG: &str = r#"{
  "do_convert_rgb": null,
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [0.5, 0.5, 0.5],
  "image_processor_type": "SiglipImageProcessor",
  "image_seq_length": 1024,
  "image_std": [0.5, 0.5, 0.5],
  "processor_class": "PaliGemmaProcessor",
  "resample": 3,
  "rescale_factor": 0.00392156862745098,
  "size": {"height": 448, "width": 448}
}"#;

static NEXT_FILE_ID: AtomicU64 = AtomicU64::new(0);

struct TestConfigFile(PathBuf);

impl TestConfigFile {
    fn write(contents: &str) -> Self {
        let id = NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "tessera-colpali-preprocessor-{}-{id}.json",
            std::process::id()
        ));
        fs::write(&path, contents).unwrap();
        Self(path)
    }
}

impl Drop for TestConfigFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[test]
fn parses_pinned_colpali_preprocessor_contract() {
    let file = TestConfigFile::write(PINNED_CONFIG);
    let config = ColPaliPreprocessorConfig::from_path(&file.0).unwrap();

    assert_eq!(config.target_size(), (448, 448));
    assert!(config
        .image_mean()
        .iter()
        .all(|value| (*value - 0.5).abs() < f32::EPSILON));
    assert!(config
        .image_std()
        .iter()
        .all(|value| (*value - 0.5).abs() < f32::EPSILON));
    assert_eq!(config.image_seq_length(), 1024);
    assert!((config.rescale_factor() - 1.0 / 255.0).abs() < f32::EPSILON);

    let processor = crate::vision::ImageProcessor::from_preprocessor_config(&config);
    assert_eq!(processor.target_size, (448, 448));
    assert!((processor.rescale_factor - config.rescale_factor()).abs() < f32::EPSILON);
}

#[test]
fn rejects_incompatible_processor_semantics() {
    let incompatible = PINNED_CONFIG.replace("\"resample\": 3", "\"resample\": 2");
    let file = TestConfigFile::write(&incompatible);
    let error = ColPaliPreprocessorConfig::from_path(&file.0).unwrap_err();

    assert!(error.to_string().contains("Invalid ColPali preprocessor"));
    assert!(format!("{error:#}").contains("only PIL bicubic (3) is supported"));
}

#[test]
fn rejects_non_finite_or_zero_normalization_values() {
    let invalid = PINNED_CONFIG.replace(
        "\"image_std\": [0.5, 0.5, 0.5]",
        "\"image_std\": [0.5, 0.0, 0.5]",
    );
    let file = TestConfigFile::write(&invalid);
    let error = ColPaliPreprocessorConfig::from_path(&file.0).unwrap_err();

    assert!(format!("{error:#}").contains("image_std[1] must be finite"));
}
