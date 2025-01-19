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

/// Creates a circular rainbow TGA image of the given dimensions.
fn create_gradient_tga(width: u16, height: u16) -> Vec<u8> {
    // TGA header (18 bytes)
    let mut data = vec![
        0, 0, 2, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
        (width & 0xFF) as u8,
        (width >> 8) as u8,
        (height & 0xFF) as u8,
        (height >> 8) as u8,
        24, // bits per pixel
        0,  // image descriptor
    ];

    for y in 0..height {
        for x in 0..width {
            let cx = x as f32 / width as f32 - 0.5;
            let cy = y as f32 / height as f32 - 0.5;
            let dist = (cx * cx + cy * cy).sqrt() * 2.0;

            // Create a rainbow effect based on angle
            let angle = cy.atan2(cx);
            let hue = (angle / std::f32::consts::PI + 1.0) * 180.0;

            // Convert HSV to RGB with distance-based saturation and value
            let saturation = (1.0 - dist).max(0.0);
            let value = (1.0 - dist * 0.5).max(0.0);
            let (r, g, b) = hsv_to_rgb(hue, saturation, value);

            // TGA stores color in BGR order
            data.push(b);
            data.push(g);
            data.push(r);
        }
    }

    data
}

/// Convert hue-saturation-value to an RGB triple (0..=255 each).
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let h = h % 360.0;
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r1, g1, b1) = match (h / 60.0) as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    (
        ((r1 + m) * 255.0) as u8,
        ((g1 + m) * 255.0) as u8,
        ((b1 + m) * 255.0) as u8,
    )
}

/// Creates and displays a gradient TGA image. First tries using the viu crate,
/// and if that fails, falls back to the original `termimage` approach.
pub fn display() -> Result<(), DisplayError> {
    // Determine dimensions from the terminal size
    let size = termsize::get().unwrap_or(termsize::Size { rows: 24, cols: 80 });
    let width = size.cols as u16;
    let height = (size.rows * 2) as u16; // 2 px per text row for half-block style

    // Create the gradient in TGA format
    let tga_data = create_gradient_tga(width, height);

    // Write TGA data to a temporary file
    let mut tmp_file = Builder::new()
        .prefix("gradient_")
        .suffix(".tga")
        .tempfile()?;
    tmp_file.write_all(&tga_data)?;

    // Attempt to display using the viu library:
    let path_str = tmp_file.path().to_string_lossy().to_string();
    let result_viu = viu::print_from_file(path_str, &viu::Config::default());

    // If viu fails for any reason, fall back to the original `termimage` method.
    if let Err(_err) = result_viu {
        println!("viu failed.");
        // Fallback: termimage
        let path_info = (String::new(), tmp_file.path().to_path_buf());
        let guessed_fmt = ops::guess_format(&path_info)?;  // Might fail -> DisplayError::Term
        let img = ops::load_image(&path_info, guessed_fmt)?;  // Might fail -> DisplayError::Term

        // We scale the image to fit the terminal
        let original_size = (width as u32, height as u32);
        let term_size = (size.cols as u32, size.rows as u32);
        let resized_size = ops::image_resized_size(original_size, term_size, true);
        let resized_img = ops::resize_image(&img, resized_size);

        let stdout = io::stdout();
        let mut writer = BufWriter::new(stdout.lock());

        // Write to terminal with ANSI Truecolor
        ops::write_ansi_truecolor(&mut writer, &resized_img);
        writer.flush()?;
    }

    Ok(())
}
