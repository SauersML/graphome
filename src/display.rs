use std::fmt;
use std::io::{self, Write};
use tempfile::Builder;
use viuer;

/// Possible errors when displaying the image.
#[derive(Debug)]
enum DisplayError {
    Io(std::io::Error),
    Viuer(viuer::ViuError),
}

impl From<std::io::Error> for DisplayError {
    fn from(e: std::io::Error) -> Self {
        DisplayError::Io(e)
    }
}

impl From<viuer::ViuError> for DisplayError {
    fn from(e: viuer::ViuError) -> Self {
        DisplayError::Viuer(e)
    }
}

impl fmt::Display for DisplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DisplayError::Io(e) => write!(f, "I/O error: {}", e),
            DisplayError::Viuer(e) => write!(f, "Viuer error: {:?}", e),
        }
    }
}

impl std::error::Error for DisplayError {}

/// Displays TGA image data in the terminal.
/// Takes raw TGA data (including header) as input.
pub fn display_tga(tga_data: &[u8]) -> Result<(), DisplayError> {
    // Write TGA data to a temporary file
    let mut tmp_file = Builder::new()
        .prefix("image_")
        .suffix(".tga")
        .tempfile()?;
    tmp_file.write_all(tga_data)?;

    let conf = viuer::Config {
        transparent: false,
        absolute_offset: false,
        width: None,       // Let viuer determine based on terminal
        height: None,      // Let viuer determine based on terminal
        x: 0,             // Start at left edge
        y: 0,             // Start at top
        restore_cursor: true,
        ..Default::default()
    };

    // First try to load the image using image crate
    let img = image::open(tmp_file.path())?;
    
    // Display using viuer
    viuer::print(&img, &conf)?;

    Ok(())
}
