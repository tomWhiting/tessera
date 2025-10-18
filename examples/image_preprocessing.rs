//! Image preprocessing example for ColPali/PaliGemma models.
//!
//! Demonstrates how to use ImageProcessor to prepare images for
//! vision-language model inference.

use tessera::backends::candle::get_device;
use tessera::vision::ImageProcessor;

fn main() -> anyhow::Result<()> {
    println!("=== Image Preprocessing Example ===\n");

    // 1. Create image processor with PaliGemma defaults
    let processor = ImageProcessor::new();
    println!("Created ImageProcessor:");
    println!(
        "  Target size: {}x{}",
        processor.target_size.0, processor.target_size.1
    );
    println!("  Mean (RGB): {:?}", processor.mean);
    println!("  Std (RGB): {:?}", processor.std);
    println!();

    // 2. Create a test image in memory
    use image::{ImageBuffer, Rgb};
    let test_img = ImageBuffer::from_fn(800, 600, |x, y| {
        // Create a gradient pattern
        let r = (x % 256) as u8;
        let g = (y % 256) as u8;
        let b = ((x + y) % 256) as u8;
        Rgb([r, g, b])
    });
    println!(
        "Created test image: {}x{}",
        test_img.width(),
        test_img.height()
    );
    println!();

    // 3. Preprocess the image
    let device = get_device()?;
    println!("Using device: {:?}", device);

    let img_dynamic = image::DynamicImage::ImageRgb8(test_img);
    let tensor = processor.preprocess_image(&img_dynamic, &device)?;

    println!("\nPreprocessed tensor shape: {:?}", tensor.shape());
    println!("Expected shape: [3, 448, 448] (channels-first)");

    // Verify shape
    assert_eq!(tensor.shape().dims(), &[3, 448, 448]);
    println!("✓ Shape verification passed!");

    // 4. Demonstrate custom configuration
    println!("\n--- Custom Configuration ---");
    let custom_processor =
        ImageProcessor::with_config((224, 224), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
    println!("Custom processor:");
    println!(
        "  Target size: {}x{}",
        custom_processor.target_size.0, custom_processor.target_size.1
    );
    println!("  Mean (RGB): {:?}", custom_processor.mean);
    println!("  Std (RGB): {:?}", custom_processor.std);

    let custom_tensor = custom_processor.preprocess_image(&img_dynamic, &device)?;
    println!("Custom tensor shape: {:?}", custom_tensor.shape());
    assert_eq!(custom_tensor.shape().dims(), &[3, 224, 224]);
    println!("✓ Custom configuration verification passed!");

    println!("\n=== Example Complete ===");
    Ok(())
}
