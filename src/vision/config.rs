//! Pinned `PaliGemma` image-processor configuration.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::{fs::File, path::Path};

const SIGLIP_PROCESSOR: &str = "SiglipImageProcessor";
const PALIGEMMA_PROCESSOR: &str = "PaliGemmaProcessor";
const PIL_BICUBIC: u8 = 3;

/// Image preprocessing metadata published with the pinned ColPali checkpoint.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ColPaliPreprocessorConfig {
    do_resize: bool,
    do_rescale: bool,
    do_normalize: bool,
    do_convert_rgb: Option<bool>,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    image_processor_type: String,
    image_seq_length: usize,
    processor_class: String,
    resample: u8,
    rescale_factor: f32,
    size: ImageSize,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
struct ImageSize {
    height: u32,
    width: u32,
}

impl ColPaliPreprocessorConfig {
    /// Parses and validates a checkpoint's `preprocessor_config.json`.
    pub(crate) fn from_path(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("Failed to open preprocessor config {}", path.display()))?;
        let config: Self = serde_json::from_reader(file)
            .with_context(|| format!("Failed to parse preprocessor config {}", path.display()))?;
        config.validate().with_context(|| {
            format!("Invalid ColPali preprocessor config at {}", path.display())
        })?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.image_processor_type == SIGLIP_PROCESSOR,
            "image_processor_type is {:?}; expected {SIGLIP_PROCESSOR:?}",
            self.image_processor_type
        );
        anyhow::ensure!(
            self.processor_class == PALIGEMMA_PROCESSOR,
            "processor_class is {:?}; expected {PALIGEMMA_PROCESSOR:?}",
            self.processor_class
        );
        anyhow::ensure!(self.do_resize, "do_resize must be true");
        anyhow::ensure!(self.do_rescale, "do_rescale must be true");
        anyhow::ensure!(self.do_normalize, "do_normalize must be true");
        anyhow::ensure!(
            self.do_convert_rgb != Some(false),
            "do_convert_rgb=false is incompatible with the RGB model input"
        );
        anyhow::ensure!(
            self.resample == PIL_BICUBIC,
            "resample is {}; only PIL bicubic ({PIL_BICUBIC}) is supported",
            self.resample
        );
        anyhow::ensure!(
            self.size.width > 0 && self.size.height > 0,
            "image size is {}x{}; both dimensions must be greater than zero",
            self.size.width,
            self.size.height
        );
        anyhow::ensure!(
            self.image_seq_length > 0,
            "image_seq_length must be greater than zero"
        );
        anyhow::ensure!(
            self.rescale_factor.is_finite() && self.rescale_factor > 0.0,
            "rescale_factor must be finite and greater than zero, got {}",
            self.rescale_factor
        );
        for (channel, (&mean, &std)) in self
            .image_mean
            .iter()
            .zip(self.image_std.iter())
            .enumerate()
        {
            anyhow::ensure!(mean.is_finite(), "image_mean[{channel}] is not finite");
            anyhow::ensure!(
                std.is_finite() && std.abs() >= f32::MIN_POSITIVE,
                "image_std[{channel}] must be finite and at least {} in magnitude, got {std}",
                f32::MIN_POSITIVE
            );
        }
        Ok(())
    }

    pub(crate) const fn target_size(&self) -> (u32, u32) {
        (self.size.width, self.size.height)
    }

    pub(crate) const fn image_mean(&self) -> [f32; 3] {
        self.image_mean
    }

    pub(crate) const fn image_std(&self) -> [f32; 3] {
        self.image_std
    }

    pub(crate) const fn rescale_factor(&self) -> f32 {
        self.rescale_factor
    }

    pub(crate) const fn image_seq_length(&self) -> usize {
        self.image_seq_length
    }
}

#[cfg(test)]
mod tests;
