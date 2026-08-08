use super::*;
use image::ImageFormat;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_TEST_FILE_ID: AtomicU64 = AtomicU64::new(0);

struct TestFile {
    path: PathBuf,
}

impl TestFile {
    fn new(extension: &str) -> Self {
        let id = NEXT_TEST_FILE_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "tessera-image-preprocessing-{}-{id}.{extension}",
            std::process::id()
        ));
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TestFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[test]
fn image_processor_creation_uses_paligemma_defaults() {
    let processor = ImageProcessor::new();
    assert_eq!(processor.target_size, (448, 448));
    assert_eq!(processor.mean.len(), 3);
    assert_eq!(processor.std.len(), 3);
    assert!((processor.rescale_factor - 1.0 / 255.0).abs() < f32::EPSILON);
}

#[test]
fn normalization_values_match_paligemma() {
    let processor = ImageProcessor::new();
    assert!((processor.mean[0] - 0.5).abs() < f32::EPSILON);
    assert!((processor.std[0] - 0.5).abs() < f32::EPSILON);
}

#[test]
fn custom_config_is_preserved() {
    let processor = ImageProcessor::with_config((224, 224), [0.5; 3], [0.5; 3]);
    assert_eq!(processor.target_size, (224, 224));
    assert!(processor
        .mean
        .iter()
        .all(|value| (*value - 0.5).abs() < f32::EPSILON));
    assert!(processor
        .std
        .iter()
        .all(|value| (*value - 0.5).abs() < f32::EPSILON));
}

#[test]
fn normalization_output_has_three_channels() {
    let processor = ImageProcessor::new();
    let image: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(10, 10, |_, _| Rgb([128; 3]));

    let normalized = processor.normalize_image(&image).unwrap();

    assert_eq!(normalized.len(), 3 * 10 * 10);
}

#[test]
fn normalization_formula_is_applied_per_channel() {
    let processor = ImageProcessor::new();
    let image: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_fn(2, 2, |_, _| Rgb([255, 0, 128]));

    let normalized = processor.normalize_image(&image).unwrap();
    let expected_red = (1.0 - processor.mean[0]) / processor.std[0];
    let expected_green = (0.0 - processor.mean[1]) / processor.std[1];

    assert!((normalized[0] - expected_red).abs() < 1e-5);
    assert!((normalized[4] - expected_green).abs() < 1e-5);
}

#[test]
fn source_size_limit_reports_measured_and_allowed_bytes() {
    validate_source_size(MAX_IMAGE_SOURCE_BYTES).unwrap();

    let error = validate_source_size(MAX_IMAGE_SOURCE_BYTES + 1).unwrap_err();
    let message = error.to_string();
    assert!(message.contains(&(MAX_IMAGE_SOURCE_BYTES + 1).to_string()));
    assert!(message.contains(&MAX_IMAGE_SOURCE_BYTES.to_string()));
}

#[test]
fn oversized_source_is_rejected_before_header_decode() {
    let test_file = TestFile::new("png");
    let file = fs::File::create(test_file.path()).unwrap();
    file.set_len(MAX_IMAGE_SOURCE_BYTES + 1).unwrap();

    let error = ImageProcessor::new()
        .preprocess_from_path(test_file.path(), &Device::Cpu)
        .unwrap_err();

    assert!(error.to_string().contains("Image source is"));
}

#[test]
fn dimension_limits_accept_boundaries_and_reject_excess() {
    validate_image_dimensions(MAX_IMAGE_EDGE, 1).unwrap();
    validate_image_dimensions(6_000, 4_000).unwrap();

    let edge_error = validate_image_dimensions(MAX_IMAGE_EDGE + 1, 1).unwrap_err();
    assert!(edge_error
        .to_string()
        .contains(&format!("{}x1", MAX_IMAGE_EDGE + 1)));
    assert!(edge_error.to_string().contains(&MAX_IMAGE_EDGE.to_string()));

    let pixel_error = validate_image_dimensions(6_000, 4_001).unwrap_err();
    assert!(pixel_error.to_string().contains("24006000"));
    assert!(pixel_error.to_string().contains("24000000"));
}

#[test]
fn oversized_header_is_rejected_before_pixel_decode() {
    let test_file = TestFile::new("bmp");
    fs::write(test_file.path(), bmp_header(6_000, 5_000)).unwrap();

    let error = ImageProcessor::new()
        .preprocess_from_path(test_file.path(), &Device::Cpu)
        .unwrap_err();
    let message = error.to_string();

    assert!(message.contains("30000000"));
    assert!(message.contains("24000000"));
}

#[test]
fn oversized_dynamic_image_is_rejected() {
    let image = DynamicImage::ImageRgb8(ImageBuffer::new(MAX_IMAGE_EDGE + 1, 1));

    let error = ImageProcessor::new()
        .preprocess_image(&image, &Device::Cpu)
        .unwrap_err();

    assert!(error
        .to_string()
        .contains(&format!("{}x1", MAX_IMAGE_EDGE + 1)));
}

#[test]
fn normal_dynamic_image_still_preprocesses() {
    let processor = ImageProcessor::with_config((4, 4), [0.5; 3], [0.5; 3]);
    let image = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(2, 3, Rgb([128; 3])));

    let tensor = processor.preprocess_image(&image, &Device::Cpu).unwrap();

    assert_eq!(tensor.dims(), &[3, 4, 4]);
}

#[test]
fn content_sniffing_decodes_a_valid_image_with_unknown_extension() {
    let test_file = TestFile::new("data");
    let image = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(2, 3, Rgb([128; 3])));
    image
        .save_with_format(test_file.path(), ImageFormat::Png)
        .unwrap();
    let processor = ImageProcessor::with_config((4, 4), [0.5; 3], [0.5; 3]);

    let tensor = processor
        .preprocess_from_path(test_file.path(), &Device::Cpu)
        .unwrap();

    assert_eq!(tensor.dims(), &[3, 4, 4]);
}

fn bmp_header(width: u32, height: u32) -> [u8; 54] {
    let mut header = [0_u8; 54];
    header[0..2].copy_from_slice(b"BM");
    header[2..6].copy_from_slice(&54_u32.to_le_bytes());
    header[10..14].copy_from_slice(&54_u32.to_le_bytes());
    header[14..18].copy_from_slice(&40_u32.to_le_bytes());
    header[18..22].copy_from_slice(&width.to_le_bytes());
    header[22..26].copy_from_slice(&height.to_le_bytes());
    header[26..28].copy_from_slice(&1_u16.to_le_bytes());
    header[28..30].copy_from_slice(&24_u16.to_le_bytes());
    header
}
