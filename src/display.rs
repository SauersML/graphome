// src/display.rs
//
// Demonstrates displaying a 4×4 TGA image (stored in-memory) with termimage,
// including fixing the error "GuessingFormatFailed(\"\")" by giving our temp
// file a `.tga` extension. That way, termimage's `guess_format()` can detect
// the format properly.

use std::fmt;
use std::io::{self, Write};
use std::path::PathBuf;
use tempfile::Builder;
use termimage::ops;

#[derive(Debug)]
enum DisplayError {
    Io(std::io::Error),
    Term(termimage::Error),
}

impl From<std::io::Error> for DisplayError {
    fn from(e: std::io::Error) -> Self {
        DisplayError::Io(e)
    }
}
impl From<termimage::Error> for DisplayError {
    fn from(e: termimage::Error) -> Self {
        DisplayError::Term(e)
    }
}
impl fmt::Display for DisplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DisplayError::Io(e) => write!(f, "I/O error: {}", e),
            DisplayError::Term(e) => write!(f, "Termimage error: {:?}", e),
        }
    }
}
impl std::error::Error for DisplayError {}

/// A 4×4, 24-bit uncompressed TGA image in memory. Each pixel is BGR.
/// Header = 18 bytes, plus 4×4×3 = 48 bytes for pixels.
static TGA_DATA: &[u8] = &[
    // TGA header (18 bytes)
    0, 0, 2, 0, 0, 0, 0, 0,
    0, 0, 0, 0,
    4, 0,  // width = 4
    4, 0,  // height=4
    24,    // bitsPerPixel=24
    0,     // imageDescriptor=0
    // Pixel data: 4×4 in BGR
    0,0,0,   0,85,0,   0,170,0,   0,255,0,
    85,0,0,  85,85,0,  85,170,0,  85,255,0,
    170,0,0, 170,85,0, 170,170,0, 170,255,0,
    255,0,0, 255,85,0, 255,170,0, 255,255,0,
];

fn main() -> Result<(), DisplayError> {
    // 1) Create a temp file with ".tga" extension so guess_format sees "something.tga"
    let mut tmp_file = Builder::new()
        .prefix("tiny_image_")
        .suffix(".tga")
        .tempfile()?;

    // 2) Write the TGA data
    tmp_file.write_all(TGA_DATA)?;

    // 3) Convert to the format termimage expects: (String, PathBuf)
    let path_info = (String::new(), tmp_file.path().to_path_buf());

    // 4) Let termimage guess the format
    let guessed_fmt = ops::guess_format(&path_info)?;

    // 5) Load the image
    let img = ops::load_image(&path_info, guessed_fmt)?;

    // 6) We know the TGA is 4×4. Suppose the user wants 80×24 in the terminal.
    //    Just pass (4,4) directly to avoid private trait issues.
    let original_size = (4, 4);
    let term_size = (80, 24);
    let resized_size = ops::image_resized_size(original_size, term_size, true);

    // 7) Resize
    let resized = ops::resize_image(&img, resized_size);

    // 8) Print in truecolor
    ops::write_ansi_truecolor(&mut io::stdout(), &resized);

    // 9) Flush
    io::stdout().flush()?;
    Ok(())
}
