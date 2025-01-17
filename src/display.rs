// src/display.rs
//
// This file demonstrates displaying an in-memory TGA image in the terminal,
// using only the version of the "image" crate that "termimage" itself uses.
// This prevents any version conflicts. We decode the TGA data, then
// resize/print with termimage.

use std::io::{self, Write};
// Import the "image" types from the same version that termimage uses.
use termimage::ops::image::{self, DynamicImage, GenericImageView, ImageFormat};
use termimage::ops;

/// A small 4×4, 24-bit uncompressed TGA image in BGR format.
/// The TGA header is 18 bytes. Then we have 4×4×3 = 48 bytes of pixel data.
static TGA_DATA: &[u8] = &[
    // TGA header (18 bytes)
    0, 0, 2, 0, 0, 0, 0, 0,
    0, 0, 0, 0,
    4, 0, // width = 4
    4, 0, // height = 4
    24,   // bits per pixel
    0,    // image descriptor
    // Pixel data in BGR
    0,0,0,   0,85,0,   0,170,0,   0,255,0,
    85,0,0,  85,85,0,  85,170,0,  85,255,0,
    170,0,0, 170,85,0, 170,170,0, 170,255,0,
    255,0,0, 255,85,0, 255,170,0, 255,255,0,
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Decode TGA from memory, using the same image crate version as termimage
    let raw_img: DynamicImage =
        image::load_from_memory_with_format(TGA_DATA, ImageFormat::Tga)?;

    // We'll pretend our terminal is 80 wide × 24 high
    let (term_w, term_h) = (80, 24);

    // Use termimage's resizing logic (which expects the same DynamicImage type)
    let (img_w, img_h) = raw_img.dimensions();
    let resized_size = ops::image_resized_size((img_w, img_h), (term_w, term_h), true);

    // Resize the image
    let resized_img = ops::resize_image(&raw_img, resized_size);

    // Print in ANSI truecolor mode
    ops::write_ansi_truecolor(&mut io::stdout(), &resized_img);

    // Flush stdout
    io::stdout().flush()?;
    Ok(())
}
