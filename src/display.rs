use std::fmt;
use std::io::{self, Write, BufWriter};
use std::env;
use tempfile::Builder;
use termsize;
use termimage::ops;
use viuer;
use image::{ImageError, ImageFormat};
use base64;

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

/// Display a TGA image in the terminal.
pub fn display_tga(tga_data: &[u8]) -> Result<(), DisplayError> {
    println!("Starting display_tga function...");

    // Force use of a terminal that supports inline images (like Kitty).
    env::set_var("TERM", "xterm-kitty");
    println!("Set terminal to 'xterm-kitty'.");

    // Basic viuer configuration.
    let conf = viuer::Config {
        transparent: false,
        absolute_offset: false,
        width: None,
        height: None,
        x: 0,
        y: 0,
        restore_cursor: false,
        ..Default::default()
    };

    println!("viuer configuration set.");

    // Attempt to load this data as TGA, then display with viuer.
    println!("Attempting to load TGA image...");
    let img = image::load_from_memory_with_format(tga_data, ImageFormat::Tga)?;
    println!("TGA image loaded successfully.");

    if let Err(_) = viuer::print(&img, &conf) {
        println!("Failed to display image with viuer. Falling back to termimage.");

        // If that fails (e.g., viuer doesn't support the protocol), fall back to termimage.
        let size = termsize::get().unwrap_or(termsize::Size { rows: 24, cols: 80 });
        println!("Terminal size detected: {} rows, {} columns.", size.rows, size.cols);

        // Write the bytes to a temporary TGA file so termimage can handle it.
        let mut tmp_file = Builder::new().prefix("image_").suffix(".tga").tempfile()?;
        tmp_file.write_all(tga_data)?;
        println!("Temporary TGA file created at: {}", tmp_file.path().display());

        let path_info = (String::new(), tmp_file.path().to_path_buf());
        let guessed_fmt = ops::guess_format(&path_info)?;
        println!("Guessed format for termimage: {:?}", guessed_fmt);

        let term_img = ops::load_image(&path_info, guessed_fmt)?;
        println!("Termimage image loaded successfully.");

        // Resize to fit the terminal, preserving aspect ratio.
        let original_size = (term_img.width(), term_img.height());
        let term_size = (size.cols as u32, size.rows as u32);
        let resized_size = ops::image_resized_size(original_size, term_size, true);
        let resized_img = ops::resize_image(&term_img, resized_size);
        println!("Image resized to fit terminal: {:?}", resized_size);

        // Output the resized image to terminal.
        let stdout = io::stdout();
        let mut writer = BufWriter::new(stdout.lock());
        ops::write_ansi_truecolor(&mut writer, &resized_img);
        writer.flush()?;
        println!("Image successfully written to terminal using termimage.");
    }

    println!("Finished displaying TGA image.");
    Ok(())
}

pub fn display_gif(gif_data: &[u8]) -> Result<(), DisplayError> {
    println!("Starting display_gif function...");

    // Get terminal size
    let term_size = termsize::get()
        .map(|size| (size.cols as u32, size.rows as u32))
        .unwrap_or((80, 24));

    // Set the terminal to support inline images (Kitty terminal).
    env::set_var("TERM", "xterm-kitty");

    // Format the image escape sequence
    let inline_image_esc = format!(
        "\x1b]1337;File=inline=1;width=400%:{}\x07",
        base64::encode(gif_data)
    );

    // Output the escape sequence to display the GIF.
    println!("Displaying GIF...");
    print!("\x1b[2J"); // Clear screen
    print!("\x1b[H");  // Move cursor to home position

    // Output the escape sequence to display the GIF.
    print!("{}", inline_image_esc);
    io::stdout().flush()?;

    // Reset the terminal.
    print!("\x1b[?25h"); // Show cursor
    print!("\x1b[?1049l"); // Exit alternate screen buffer
    io::stdout().flush()?;
    Ok(())
}
