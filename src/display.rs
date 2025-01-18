use std::fmt;
use std::io::{self, Write, BufWriter};
use termsize;
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

fn create_gradient_tga(width: u16, height: u16) -> Vec<u8> {
    // TGA header (18 bytes)
    let mut data = vec![
        0, 0, 2, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
        (width & 0xFF) as u8,
        (width >> 8) as u8,
        (height & 0xFF) as u8,
        (height >> 8) as u8,
        24,    // bits per pixel
        0,     // image descriptor
    ];

    // Generate a circular gradient pattern
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

            // Add BGR values (TGA format)
            data.push(b);
            data.push(g);
            data.push(r);
        }
    }

    data
}

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

fn main() -> Result<(), DisplayError> {
    // Create a larger, more interesting image
    let size = termsize::get().unwrap_or(termsize::Size { rows: 24, cols: 80 });
    let width = (size.cols * 2) as u16;  // Double for better resolution
    let height = (size.rows * 4) as u16;  // Quadruple height for half-block characters
    let tga_data = create_gradient_tga(width, height);

    // Create a temp file with ".tga" extension
    let mut tmp_file = Builder::new()
        .prefix("gradient_")
        .suffix(".tga")
        .tempfile()?;
    
    // Write the TGA data
    tmp_file.write_all(&tga_data)?;
    
    // Set up for termimage
    let path_info = (String::new(), tmp_file.path().to_path_buf());
    let guessed_fmt = ops::guess_format(&path_info)?;
    let img = ops::load_image(&path_info, guessed_fmt)?;
    
    // Get terminal size for better fitting
    let original_size = (width as u32, height as u32);
    let size = termsize::get().unwrap_or(termsize::Size { rows: 24, cols: 80 });
    let term_size = (size.cols as u32, size.rows as u32);
    let resized_size = ops::image_resized_size(original_size, term_size, true);
    
    // Resize and display
    let resized = ops::resize_image(&img, resized_size);
    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());
    ops::write_ansi_truecolor(&mut writer, &resized);
    writer.flush()?;

    Ok(())
}
