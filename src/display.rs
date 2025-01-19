use std::fmt;
use std::io::{self, Write, BufWriter};
use tempfile::Builder;
use termsize;
use termimage::ops;
use viuer;
use image::{ImageError, ImageFormat};

/// Possible errors when displaying the image.
#[derive(Debug)]
pub enum DisplayError {
    Io(std::io::Error),
    Viuer(viuer::ViuError),
    Image(image::error::ImageError),
    Term(termimage::Error)
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

impl From<termimage::Error> for DisplayError {
    fn from(e: termimage::Error) -> Self {
        DisplayError::Term(e)
    }
}

impl fmt::Display for DisplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DisplayError::Io(e) => write!(f, "I/O error: {}", e),
            DisplayError::Viuer(e) => write!(f, "Viuer error: {:?}", e),
            DisplayError::Image(e) => write!(f, "Image error: {}", e),
            DisplayError::Term(e) => write!(f, "Termimage error: {:?}", e),
        }
    }
}

impl std::error::Error for DisplayError {}

/// Displays TGA image data in the terminal.
/// Takes raw TGA data (including header) as input.
pub fn display_tga(tga_data: &[u8]) -> Result<(), DisplayError> {
    // Set Kitty terminal if possible
    if env::var("KITTY_WINDOW_ID").is_ok() {
        env::set_var("TERM", "xterm-kitty");
    }
    // Try viuer first
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

    // Load image from the memory buffer
    let img = image::load_from_memory_with_format(tga_data, image::ImageFormat::Tga)?;
    
    // Try viuer first
    if let Err(_) = viuer::print(&img, &conf) {
        // Fall back to termimage
        // Get terminal dimensions for scaling
        let size = termsize::get().unwrap_or(termsize::Size { rows: 24, cols: 80 });

        // Write to temporary file for termimage
        let mut tmp_file = Builder::new()
            .prefix("image_")
            .suffix(".tga")
            .tempfile()?;
        tmp_file.write_all(tga_data)?;
        
        let path_info = (String::new(), tmp_file.path().to_path_buf());
        let guessed_fmt = ops::guess_format(&path_info)?;
        let term_img = ops::load_image(&path_info, guessed_fmt)?;

        // Scale image to terminal size
        let original_size = (term_img.width(), term_img.height());
        let term_size = (size.cols as u32, size.rows as u32);
        let resized_size = ops::image_resized_size(original_size, term_size, true);
        let resized_img = ops::resize_image(&term_img, resized_size);

        let stdout = io::stdout();
        let mut writer = BufWriter::new(stdout.lock());
        
        // Write using ANSI Truecolor
        ops::write_ansi_truecolor(&mut writer, &resized_img);
        writer.flush()?;
    }

    Ok(())
}
