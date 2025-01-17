// src/display.rs
//
// This file demonstrates how to display an in-memory TGA image with termimage,
// without importing the image crate directly. It writes the TGA data to a
// temporary file, then uses termimage's internal logic to guess the format,
// load, resize, and print the image in truecolor.

use std::io::{self, Write};
use std::path::PathBuf;
use tempfile::NamedTempFile;
use termimage::ops;

/// A 4×4, 24-bit uncompressed TGA image in BGR format, including an 18-byte header.
/// After the header are 4×4×3 = 48 bytes of pixel data.
static TGA_DATA: &[u8] = &[
    0, 0, 2, 0, 0, 0, 0, 0,
    0, 0, 0, 0,
    4, 0, // width=4
    4, 0, // height=4
    24,   // bits per pixel
    0,    // image descriptor
    // Pixel data, row by row in BGR
    0,0,0,   0,85,0,   0,170,0,   0,255,0,
    85,0,0,  85,85,0,  85,170,0,  85,255,0,
    170,0,0, 170,85,0, 170,170,0, 170,255,0,
    255,0,0, 255,85,0, 255,170,0, 255,255,0,
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Write TGA data to a temporary file so that termimage can load it
    let mut tmp = NamedTempFile::new()?;
    tmp.write_all(TGA_DATA)?;

    // Make a file-reference pair in the format termimage expects
    let path_str = tmp.path().display().to_string();
    let path_info = (String::new(), PathBuf::from(&path_str));

    // Guess the image format (should be Tga)
    let guessed = ops::guess_format(&path_info)?;
    // Load the image using termimage's built-in approach
    let img = ops::load_image(&path_info, guessed)?;

    // Assume a terminal of size 80x24
    let (term_w, term_h) = (80, 24);

    // Calculate how large to display, preserving aspect ratio
    let new_size = ops::image_resized_size(img.dimensions(), (term_w, term_h), true);

    // Resize
    let resized = ops::resize_image(&img, new_size);

    // Output with ANSI truecolor
    ops::write_ansi_truecolor(&mut io::stdout(), &resized);

    io::stdout().flush()?;
    Ok(())
}
