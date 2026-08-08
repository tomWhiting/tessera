//! Vision-language embedding support (`ColPali`).
//!
//! Implements image preprocessing and vision-language retrieval
//! using `ColPali` architecture.

pub mod config;
pub mod preprocessing;
pub mod processor;

pub use config::ColPaliPreprocessorConfig;
pub use preprocessing::ImageProcessor;
pub use processor::ColPaliProcessor;
