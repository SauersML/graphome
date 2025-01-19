use std::fmt;
use std::io::{self, Write};
use tempfile::Builder;
use viuer;
use image::ImageError;

/// Possible errors when displaying the image.
#[derive(Debug)]
enum DisplayError {
    Io(std::io::Error),
    Viuer(viuer::ViuError),
    Image(image::error::ImageError)
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

impl From<ImageError> for DisplayError {
    fn from(e: ImageError) -> Self {
        DisplayError::Image(e)
    }
}

impl fmt::Display for DisplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DisplayError::Io(e) => write!(f, "I/O error: {}", e),
            DisplayError::Viuer(e) => write!(f, "Viuer error: {:?}", e),
            DisplayError::Image(e) => write!(f, "Image error: {}", e),
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
        width: None,
        height: None,
        x: 0,
        y: 0,
        restore_cursor: true,
        ..Default::default()
    };

    // Load image from the memory buffer directly instead of using the file
    let img = image::load_from_memory(tga_data)?;
    
    // Display using viuer
    viuer::print(&img, &conf)?;

    Ok(())
}
