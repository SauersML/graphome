// src/display.rs
//
// Demonstrates displaying an in‐memory TGA image in the terminal
// using the "image" crate (for loading) and "termimage" (for terminal display).

use std::io::{self, Write};
use image::{DynamicImage, ImageFormat};
use termimage::ops;
use termimage::util;

/// A tiny 4×4, 24‐bit uncompressed TGA image.
/// Each pixel is stored in BGR order (per TGA specs).
/// The header is 18 bytes, followed by 4×4×3 = 48 bytes of pixel data.
static TGA_DATA: &[u8] = &[
    // -- TGA header (18 bytes) --
    0, 0, 2, 0, 0, 0, 0, 0,     // idLength=0, colorMapType=0, dataTypeCode=2, ...
    0, 0, 0, 0,                 // colorMapSpec=all zeros, XOrigin=0, YOrigin=0
    4, 0,                       // width = 4 (low byte, then high byte)
    4, 0,                       // height= 4
    24,                         // bitsPerPixel = 24
    0,                          // imageDescriptor=0
    // -- Pixel data: row-by-row (4×4), each pixel B,G,R (3 bytes) --
    // Row 0:
    0,0,0,   0,85,0,   0,170,0,   0,255,0,
    // Row 1:
    85,0,0,  85,85,0,  85,170,0,  85,255,0,
    // Row 2:
    170,0,0, 170,85,0, 170,170,0, 170,255,0,
    // Row 3:
    255,0,0, 255,85,0, 255,170,0, 255,255,0,
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load from memory as a TGA image using the "image" crate
    let raw_img: DynamicImage = image::load_from_memory_with_format(TGA_DATA, ImageFormat::Tga)?;

    // Detect terminal size (fallback to 80×24 if detection fails)
    let (term_w, term_h) = util::terminal_size().unwrap_or((80, 24));

    // Calculate how big to resize the image so it fits in terminal
    //    The last 'true' => preserve aspect ratio
    let resized_size = ops::image_resized_size(raw_img.dimensions(), (term_w, term_h), true);

    // Resize the image
    let resized_img = ops::resize_image(&raw_img, resized_size);

    // Display in Truecolor. This function returns (), so no “?”.
    ops::write_ansi_truecolor(&mut io::stdout(), &resized_img);

    // Flush just to be sure
    io::stdout().flush()?;

    Ok(())
}
