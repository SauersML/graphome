use base64::{engine::general_purpose, Engine as _};
use image::{ImageError, ImageFormat};
use std::env;
use std::fmt;
use std::io::{self, BufWriter, Write};
use tempfile::Builder;
use termimage::ops;
use termsize;
use viuer;

#[derive(Debug)]
pub enum DisplayError {
    Io(std::io::Error),
    Viuer(viuer::ViuError),
    Image(image::error::ImageError),
    Term(termimage::Error),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InlineImageProtocol {
    Kitty,
    ITerm2,
    None,
}

fn detect_inline_image_protocol() -> InlineImageProtocol {
    let term = env::var("TERM").unwrap_or_default().to_lowercase();
    let term_program = env::var("TERM_PROGRAM").unwrap_or_default().to_lowercase();
    let lc_terminal = env::var("LC_TERMINAL").unwrap_or_default().to_lowercase();

    let is_kitty_like = term.contains("kitty")
        || term.contains("ghostty")
        || term_program.contains("kitty")
        || term_program.contains("ghostty");
    if is_kitty_like {
        return InlineImageProtocol::Kitty;
    }

    let is_iterm2 = term_program.contains("iterm") || lc_terminal.contains("iterm");
    if is_iterm2 {
        return InlineImageProtocol::ITerm2;
    }

    InlineImageProtocol::None
}

fn write_kitty_inline_image(data: &[u8], kitty_format: u32) -> io::Result<()> {
    let encoded = general_purpose::STANDARD.encode(data);
    let mut stdout = io::stdout().lock();

    let mut offset = 0usize;
    let chunk_size = 4096usize;
    let mut first = true;

    while offset < encoded.len() {
        let end = (offset + chunk_size).min(encoded.len());
        let chunk = &encoded[offset..end];
        let more = if end < encoded.len() { 1 } else { 0 };

        if first {
            write!(
                stdout,
                "\x1b_Ga=T,t=d,f={},m={};",
                kitty_format, more
            )?;
            first = false;
        } else {
            write!(stdout, "\x1b_Gm={};", more)?;
        }

        stdout.write_all(chunk.as_bytes())?;
        stdout.write_all(b"\x1b\\")?;
        offset = end;
    }

    stdout.flush()
}

fn write_iterm2_inline_image(data: &[u8], width: &str) -> io::Result<()> {
    let encoded = general_purpose::STANDARD.encode(data);
    let mut stdout = io::stdout().lock();
    write!(
        stdout,
        "\x1b]1337;File=inline=1;preserveAspectRatio=1;width={}:{}\x07",
        width, encoded
    )?;
    stdout.flush()
}

/// Display a TGA image in the terminal.
pub fn display_tga(tga_data: &[u8]) -> Result<(), DisplayError> {
    println!("Starting display_tga function...");

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

    if viuer::print(&img, &conf).is_err() {
        println!("Failed to display image with viuer. Falling back to termimage.");

        // If that fails (e.g., viuer doesn't support the protocol), fall back to termimage.
        let size = termsize::get().unwrap_or(termsize::Size { rows: 24, cols: 80 });
        println!(
            "Terminal size detected: {} rows, {} columns.",
            size.rows, size.cols
        );

        // Write the bytes to a temporary TGA file so termimage can handle it.
        let mut tmp_file = Builder::new().prefix("image_").suffix(".tga").tempfile()?;
        tmp_file.write_all(tga_data)?;
        println!(
            "Temporary TGA file created at: {}",
            tmp_file.path().display()
        );

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

    println!("Displaying GIF...");
    print!("\x1b[2J");
    println!("\n\n\n      ");

    match detect_inline_image_protocol() {
        InlineImageProtocol::Kitty => {
            // Kitty/Ghostty graphics protocol. f=100 indicates PNG/JPEG/GIF encoded payload.
            write_kitty_inline_image(gif_data, 100)?;
        }
        InlineImageProtocol::ITerm2 => {
            // iTerm2 OSC 1337 inline images.
            write_iterm2_inline_image(gif_data, "100%")?;
        }
        InlineImageProtocol::None => {
            // Fallback to static rendering via viuer if inline image protocols are unavailable.
            let img = image::load_from_memory(gif_data)?;
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
            viuer::print(&img, &conf)?;
        }
    }

    io::stdout().flush()?;
    Ok(())
}
