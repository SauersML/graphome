// display.rs
//
// Demonstrates displaying an in‐memory TGA image in the terminal
// using termimage (uncompressed 24‐bit TGA = fast to load).

use std::io::{self, Write};
use termimage::ops::{self, ImageFormat};

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
    // We know it's TGA, so we specify ImageFormat::Tga directly.
    let format = Some(ImageFormat::Tga);

    // Load the image from our in-memory buffer.
    // If you had a file, you could do ops::guess_format() + ops::load_image(...)
    let img = ops::load_image_buf(TGA_DATA, format)?;

    // Determine suitable display size.
    // Passing None => auto-detect terminal size if possible.
    // The `true` means "preserve aspect ratio."
    let size = ops::image_resized_size(img.dimensions(), None, true);

    // Resize the image (e.g. to fit the terminal).
    let resized = ops::resize_image(&img, size);

    // Now print it with Truecolor ANSI codes for best color fidelity.
    ops::write_ansi_truecolor(&mut io::stdout(), &resized)?;

    // Optionally flush stdout just to be sure.
    io::stdout().flush()?;

    Ok(())
}
