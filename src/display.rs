// src/display.rs
//
// This file demonstrates displaying an in-memory TGA image in the terminal.
// It uses the "image" crate to decode the TGA data, and "termimage" to
// resize and print ANSI colors.

use std::io::{self, Write};
use image::{DynamicImage, ImageFormat};
use termimage::ops;

/// A small 4×4, 24-bit uncompressed TGA image.
/// Each pixel is B,G,R, and the header is 18 bytes.
/// After the header, we have 4×4×3 = 48 bytes of pixel data.
static TGA_DATA: &[u8] = &[
    // TGA header (18 bytes)
    0, 0, 2, 0, 0, 0, 0, 0,
    0, 0, 0, 0,
    4, 0, // width  = 4
    4, 0, // height = 4
    24,   // 24 bits per pixel
    0,    // image descriptor
    // Pixel data, row by row, in BGR
    0,0,0,   0,85,0,   0,170,0,   0,255,0,
    85,0,0,  85,85,0,  85,170,0,  85,255,0,
    170,0,0, 170,85,0, 170,170,0, 170,255,0,
    255,0,0, 255,85,0, 255,170,0, 255,255,0,
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the TGA image from memory
    let raw_img: DynamicImage = image::load_from_memory_with_format(TGA_DATA, ImageFormat::Tga)?;

    // Use a fixed terminal size of 80×24 for this example
    let (term_w, term_h) = (80, 24);

    // Calculate the display size with aspect-ratio preservation
    let resized_size = ops::image_resized_size(raw_img.dimensions(), (term_w, term_h), true);

    // Resize
    let resized_img = ops::resize_image(&raw_img, resized_size);

    // Output in truecolor
    ops::write_ansi_truecolor(&mut io::stdout(), &resized_img);

    // Flush
    io::stdout().flush()?;

    Ok(())
}
