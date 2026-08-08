//! Image preprocessing for vision-language models.
//!
//! Handles image loading, resizing, and normalization to match
//! `PaliGemma` model expectations.

use crate::error::{Result, TesseraError};
use candle_core::{Device, Tensor};
use image::{DynamicImage, ImageBuffer, ImageDecoder, ImageReader, Limits, Rgb};
use std::{fs::File, io::BufReader, path::Path};

const MAX_IMAGE_SOURCE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_IMAGE_EDGE: u32 = 16_384;
const MAX_IMAGE_PIXELS: u64 = 24_000_000;
const MAX_DECODED_IMAGE_BYTES: u64 = MAX_IMAGE_PIXELS * 16;

/// Image preprocessor for ColPali/PaliGemma models.
///
/// Handles image loading, resizing, and normalization to prepare
/// images for vision transformer processing.
pub struct ImageProcessor {
    /// Target image size (width, height) - typically (448, 448)
    pub target_size: (u32, u32),

    /// Normalization mean values [R, G, B].
    /// The pinned PaliGemma processor publishes `[0.5, 0.5, 0.5]`.
    pub mean: [f32; 3],

    /// Normalization std values [R, G, B].
    /// The pinned PaliGemma processor publishes `[0.5, 0.5, 0.5]`.
    pub std: [f32; 3],

    /// Scale applied to each byte-valued channel before normalization.
    pub rescale_factor: f32,
}

impl ImageProcessor {
    /// Create new image processor with `PaliGemma` defaults.
    ///
    /// Uses the published `PaliGemma` 448 defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            target_size: (448, 448),
            mean: [0.5; 3],
            std: [0.5; 3],
            rescale_factor: 1.0 / 255.0,
        }
    }

    /// Create processor with custom parameters.
    #[must_use]
    #[cfg(test)]
    pub const fn with_config(target_size: (u32, u32), mean: [f32; 3], std: [f32; 3]) -> Self {
        Self {
            target_size,
            mean,
            std,
            rescale_factor: 1.0 / 255.0,
        }
    }

    pub(crate) fn from_preprocessor_config(config: &super::ColPaliPreprocessorConfig) -> Self {
        Self {
            target_size: config.target_size(),
            mean: config.image_mean(),
            std: config.image_std(),
            rescale_factor: config.rescale_factor(),
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
    /// - The encoded file is larger than 64 MiB
    /// - An image edge is larger than 16,384 pixels
    /// - The image contains more than 24 million pixels
    /// - Tensor creation fails
    pub fn preprocess_from_path(&self, image_path: &Path, device: &Device) -> Result<Tensor> {
        let file = File::open(image_path).map_err(|error| {
            TesseraError::ConfigError(format!(
                "Failed to open image '{}': {error}",
                image_path.display()
            ))
        })?;
        let source_bytes = file.metadata().map_err(|error| {
            TesseraError::ConfigError(format!(
                "Failed to inspect image '{}': {error}",
                image_path.display()
            ))
        })?;
        validate_source_size(source_bytes.len())?;

        let reader = ImageReader::new(BufReader::new(file))
            .with_guessed_format()
            .map_err(|error| {
                TesseraError::ConfigError(format!(
                    "Failed to inspect image format for '{}': {error}",
                    image_path.display()
                ))
            })?;
        let mut decoder = reader.into_decoder().map_err(|error| {
            TesseraError::ConfigError(format!(
                "Failed to read image header for '{}': {error}",
                image_path.display()
            ))
        })?;

        let (width, height) = decoder.dimensions();
        validate_image_dimensions(width, height)?;
        validate_decoded_size(decoder.total_bytes())?;

        let mut limits = Limits::default();
        limits.max_image_width = Some(MAX_IMAGE_EDGE);
        limits.max_image_height = Some(MAX_IMAGE_EDGE);
        limits.max_alloc = Some(MAX_DECODED_IMAGE_BYTES);
        decoder.set_limits(limits).map_err(|error| {
            TesseraError::ConfigError(format!(
                "Image decoder rejected '{}': {error}",
                image_path.display()
            ))
        })?;

        let img = DynamicImage::from_decoder(decoder).map_err(|error| {
            TesseraError::ConfigError(format!(
                "Failed to decode image '{}': {error}",
                image_path.display()
            ))
        })?;

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
    ///
    /// # Errors
    ///
    /// Returns an error before RGB conversion if an image edge is larger than
    /// 16,384 pixels or the image contains more than 24 million pixels. Tensor
    /// creation errors are also returned.
    pub fn preprocess_image(&self, img: &DynamicImage, device: &Device) -> Result<Tensor> {
        validate_image_dimensions(img.width(), img.height())?;
        validate_image_dimensions(self.target_size.0, self.target_size.1)?;

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
    /// Formula: `normalized = (pixel * rescale_factor - mean) / std`
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
                let value = f32::from(pixel[channel]) * self.rescale_factor;

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
mod tests;

fn validate_source_size(source_bytes: u64) -> Result<()> {
    if source_bytes > MAX_IMAGE_SOURCE_BYTES {
        return Err(TesseraError::ConfigError(format!(
            "Image source is {source_bytes} bytes; maximum allowed is {MAX_IMAGE_SOURCE_BYTES} bytes"
        )));
    }

    Ok(())
}

fn validate_image_dimensions(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(TesseraError::ConfigError(format!(
            "Image dimensions are {width}x{height}; width and height must each be at least 1 pixel"
        )));
    }

    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(|| {
            TesseraError::ConfigError(format!(
                "Image dimensions {width}x{height} overflow the pixel count"
            ))
        })?;

    if width > MAX_IMAGE_EDGE || height > MAX_IMAGE_EDGE {
        return Err(TesseraError::ConfigError(format!(
            "Image dimensions are {width}x{height}; maximum allowed edge is {MAX_IMAGE_EDGE} pixels"
        )));
    }

    if pixels > MAX_IMAGE_PIXELS {
        return Err(TesseraError::ConfigError(format!(
            "Image has {pixels} pixels ({width}x{height}); maximum allowed is {MAX_IMAGE_PIXELS} pixels"
        )));
    }

    Ok(())
}

fn validate_decoded_size(decoded_bytes: u64) -> Result<()> {
    if decoded_bytes > MAX_DECODED_IMAGE_BYTES {
        return Err(TesseraError::ConfigError(format!(
            "Decoded image requires {decoded_bytes} bytes; maximum allowed is {MAX_DECODED_IMAGE_BYTES} bytes"
        )));
    }

    Ok(())
}
