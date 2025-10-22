//! Image preprocessing for vision-language models.
//!
//! Handles image loading, resizing, and normalization to match
//! `PaliGemma` model expectations.

use crate::error::{Result, TesseraError};
use candle_core::{Device, Tensor};
use image::{DynamicImage, ImageBuffer, Rgb};
use std::path::Path;

/// Image preprocessor for ColPali/PaliGemma models.
///
/// Handles image loading, resizing, and normalization to prepare
/// images for vision transformer processing.
pub struct ImageProcessor {
    /// Target image size (width, height) - typically (448, 448)
    pub target_size: (u32, u32),

    /// Normalization mean values [R, G, B]
    /// SigLIP/ImageNet: [0.48145466, 0.4578275, 0.40821073]
    pub mean: [f32; 3],

    /// Normalization std values [R, G, B]
    /// SigLIP/ImageNet: [0.26862954, 0.26130258, 0.27577711]
    pub std: [f32; 3],
}

impl ImageProcessor {
    /// Create new image processor with `PaliGemma` defaults.
    ///
    /// Uses 448×448 target size and `SigLIP` normalization parameters.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            target_size: (448, 448),
            mean: [0.481_454_66, 0.457_827_5, 0.408_210_73],
            std: [0.268_629_54, 0.261_302_6, 0.275_777_1],
        }
    }

    /// Create processor with custom parameters.
    #[must_use]
    pub const fn with_config(target_size: (u32, u32), mean: [f32; 3], std: [f32; 3]) -> Self {
        Self {
            target_size,
            mean,
            std,
        }
    }

    /// Preprocess image from file path.
    ///
    /// # Arguments
    ///
    /// * `image_path` - Path to image file
    /// * `device` - Device to create tensor on
    ///
    /// # Returns
    ///
    /// Normalized image tensor with shape [3, height, width] (channels-first).
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Image file cannot be loaded
    /// - Image format is unsupported
    /// - Tensor creation fails
    pub fn preprocess_from_path(&self, image_path: &Path, device: &Device) -> Result<Tensor> {
        // Load image
        let img = image::open(image_path)
            .map_err(|e| TesseraError::ConfigError(format!("Failed to load image: {e}")))?;

        self.preprocess_image(&img, device)
    }

    /// Preprocess a `DynamicImage`.
    ///
    /// # Arguments
    ///
    /// * `img` - Image to preprocess
    /// * `device` - Device to create tensor on
    ///
    /// # Returns
    ///
    /// Normalized image tensor with shape [3, height, width].
    pub fn preprocess_image(&self, img: &DynamicImage, device: &Device) -> Result<Tensor> {
        // 1. Convert to RGB
        let rgb_img = img.to_rgb8();

        // 2. Resize to target size (bicubic interpolation)
        let resized = image::imageops::resize(
            &rgb_img,
            self.target_size.0,
            self.target_size.1,
            image::imageops::FilterType::CatmullRom, // Bicubic
        );

        // 3. Convert to f32 and normalize
        let normalized = self.normalize_image(&resized)?;

        // 4. Create tensor [3, H, W] (channels-first)
        let tensor = Tensor::from_vec(
            normalized,
            (3, self.target_size.1 as usize, self.target_size.0 as usize),
            device,
        )
        .map_err(|e| TesseraError::EncodingError {
            context: "Failed to create image tensor".to_string(),
            source: e.into(),
        })?;

        Ok(tensor)
    }

    /// Normalize image pixels using mean/std.
    ///
    /// Formula: `normalized = (pixel / 255.0 - mean) / std`
    ///
    /// # Arguments
    ///
    /// * `img` - RGB8 image buffer
    ///
    /// # Returns
    ///
    /// Normalized pixel values as flat Vec (channels-first: RGBRGB...)
    fn normalize_image(&self, img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<Vec<f32>> {
        let (width, height) = img.dimensions();
        let num_pixels = (width * height) as usize;

        // Pre-allocate for all channels (R, G, B)
        let mut normalized = Vec::with_capacity(num_pixels * 3);

        // Process channels separately (channels-first layout)
        for channel in 0..3 {
            for pixel in img.pixels() {
                // Convert to [0, 1]
                let value = f32::from(pixel[channel]) / 255.0;

                // Apply normalization
                let normed = (value - self.mean[channel]) / self.std[channel];
                normalized.push(normed);
            }
        }

        Ok(normalized)
    }
}

impl Default for ImageProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_processor_creation() {
        let processor = ImageProcessor::new();
        assert_eq!(processor.target_size, (448, 448));
        assert_eq!(processor.mean.len(), 3);
        assert_eq!(processor.std.len(), 3);
    }

    #[test]
    fn test_normalization_values() {
        let processor = ImageProcessor::new();
        // Verify SigLIP mean/std values
        assert!((processor.mean[0] - 0.48145466).abs() < 1e-6);
        assert!((processor.std[0] - 0.26862954).abs() < 1e-6);
    }

    #[test]
    fn test_custom_config() {
        let processor = ImageProcessor::with_config((224, 224), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
        assert_eq!(processor.target_size, (224, 224));
        assert_eq!(processor.mean, [0.5, 0.5, 0.5]);
        assert_eq!(processor.std, [0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_normalization_output_size() {
        let processor = ImageProcessor::new();
        // Create a small test image
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(10, 10, |_, _| Rgb([128u8, 128u8, 128u8]));

        let normalized = processor.normalize_image(&img).unwrap();
        // Should have 3 channels * width * height
        assert_eq!(normalized.len(), 3 * 10 * 10);
    }

    #[test]
    fn test_normalization_formula() {
        let processor = ImageProcessor::new();
        // Create a test image with known pixel value
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(2, 2, |_, _| Rgb([255u8, 0u8, 128u8]));

        let normalized = processor.normalize_image(&img).unwrap();

        // Check first pixel of R channel (255)
        let r_normalized = (255.0 / 255.0 - processor.mean[0]) / processor.std[0];
        assert!((normalized[0] - r_normalized).abs() < 1e-5);

        // Check first pixel of G channel (0)
        let g_normalized = (0.0 / 255.0 - processor.mean[1]) / processor.std[1];
        assert!((normalized[4] - g_normalized).abs() < 1e-5);
    }
}
