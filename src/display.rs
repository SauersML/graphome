use std::fmt;
use std::io::{self, Write, BufWriter};
use std::path::PathBuf;
use tempfile::Builder;
use termsize;
use termimage::ops;

/// Possible errors when displaying the image.
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

/// Displays TGA image data in the terminal.
/// Takes raw TGA data (including header) as input.
pub fn display_tga(tga_data: &[u8]) -> Result<(), DisplayError> {
    // Write TGA data to a temporary file
    let mut tmp_file = Builder::new()
        .prefix("image_")
        .suffix(".tga")
        .tempfile()?;
    tmp_file.write_all(tga_data)?;

    // Get terminal dimensions for scaling
    let size = termsize::get().unwrap_or(termsize::Size { rows: 24, cols: 80 });
    
    // Try viu first
    let path_str = tmp_file.path().to_string_lossy().to_string();
    let result_viu = viu::print_from_file(&path_str, &viu::Config::default());

    // Fall back to termimage if viu fails
    if let Err(_err) = result_viu {
        // Use termimage
        let path_info = (String::new(), tmp_file.path().to_path_buf());
        let guessed_fmt = ops::guess_format(&path_info)?;
        let img = ops::load_image(&path_info, guessed_fmt)?;

        // Scale image to terminal size
        let original_size = (img.width(), img.height());
        let term_size = (size.cols as u32, size.rows as u32);
        let resized_size = ops::image_resized_size(original_size, term_size, true);
        let resized_img = ops::resize_image(&img, resized_size);

        let stdout = io::stdout();
        let mut writer = BufWriter::new(stdout.lock());
        
        // Write using ANSI Truecolor
        ops::write_ansi_truecolor(&mut writer, &resized_img);
        writer.flush()?;
    }

    Ok(())
}
