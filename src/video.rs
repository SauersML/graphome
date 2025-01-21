use std::fmt;
use std::fs::File;
use std::io::{self, Write};
use nalgebra::{Rotation3, Vector3, Point3};
use image::{
    ImageBuffer,
    Rgb,
    Rgba,
    RgbaImage,
    DynamicImage,
    ImageError,
    Frame,
    Delay,
    codecs::gif::{GifEncoder, Repeat},
};
use crate::embed::Point3D;
use crate::display::{display_gif, DisplayError};

#[derive(Debug)]
pub enum VideoError {
    Io(std::io::Error),
    Image(ImageError),
    Display(DisplayError),
}

impl fmt::Display for VideoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VideoError::Io(e) => write!(f, "IO error: {}", e),
            VideoError::Image(e) => write!(f, "Image error: {}", e),
            VideoError::Display(e) => write!(f, "Display error: {}", e),
        }
    }
}

impl std::error::Error for VideoError {}

impl From<std::io::Error> for VideoError {
    fn from(e: std::io::Error) -> Self {
        VideoError::Io(e)
    }
}

impl From<ImageError> for VideoError {
    fn from(e: ImageError) -> Self {
        VideoError::Image(e)
    }
}

impl From<DisplayError> for VideoError {
    fn from(e: DisplayError) -> Self {
        VideoError::Display(e)
    }
}

/// Renders a multi-frame GIF of rotating 3D points:
/// 1. Generates ~60 frames for a full 360° rotation.
/// 2. Encodes those frames into one animated GIF.
/// 3. Saves that GIF to disk.
/// 4. Displays it once in the terminal (via `display_gif`).
pub fn render(points: Vec<Point3D>) -> Result<(), VideoError> {
    let width = 800;
    let height = 600;
    let total_frames = 60; // Number of frames in one full rotation
    let angle_step = 2.0 * std::f32::consts::PI / total_frames as f32;

    // Collect each frame as an RGBA buffer.
    let mut frames = Vec::with_capacity(total_frames);

    for i in 0..total_frames {
        let angle = i as f32 * angle_step;
        let rotation = Rotation3::from_euler_angles(0.0, angle, 0.0);

        // Create a blank image and Z-buffer
        let mut img = ImageBuffer::new(width, height);
        let mut z_buffer = vec![f32::INFINITY; (width * height) as usize];

        // Fill the image by projecting each point
        for point in &points {
            let rotated = rotation.transform_point(&point.pos);
            let camera_offset = 5.0;
            let transformed = Vector3::new(rotated.x, rotated.y, rotated.z - camera_offset);

            if let Some((sx, sy, depth)) = project_to_screen(
                transformed, width, height, 60.0_f32.to_radians()
            ) {
                // Draw a small 2×2 "dot"
                let size = 2;
                for dx in 0..size {
                    for dy in 0..size {
                        let px = sx.saturating_add(dx).min(width - 1);
                        let py = sy.saturating_add(dy).min(height - 1);
                        let idx = (py * width + px) as usize;

                        // Z-test
                        if depth < z_buffer[idx] {
                            z_buffer[idx] = depth;
                            img.put_pixel(px, py, point.color);
                        }
                    }
                }
            }
        }

        // Convert RGB image to RGBA for the GIF frames.
        let rgba: RgbaImage = DynamicImage::ImageRgb8(img).to_rgba8();

        // Create a Frame with a small delay (~16 ms -> ~60 fps).
        let frame = Frame::from_parts(
            rgba,
            0, 0,
            Delay::from_numer_denom_ms(16, 1), // 16ms per frame
        );

        frames.push(frame);
    }

    // Encode all frames into one multi-frame GIF.
    let mut gif_data = Vec::new();
    {
        let mut encoder = GifEncoder::new(&mut gif_data);
        // Loop infinitely.
        encoder.set_repeat(Repeat::Infinite)?;

        for frame in frames {
            encoder.encode_frame(frame)?;
        }
    }

    // Save the resulting GIF to disk.
    let mut file = File::create("myanim.gif")?;
    file.write_all(&gif_data)?;

    // Display it once via viuer
    display_gif(&gif_data)?;

    Ok(())
}

/// Projects a 3D vector onto a 2D screen.
/// Returns (screen_x, screen_y, depth).
fn project_to_screen(
    v: Vector3<f32>,
    width: u32,
    height: u32,
    fov: f32,
) -> Option<(u32, u32, f32)> {
    let aspect = width as f32 / height as f32;
    let half_fov_tan = (fov * 0.5).tan();
    let f = 1.0 / half_fov_tan;

    let near = 0.1;
    let far = 100.0;

    // If the point is behind the camera, skip.
    if v.z >= 0.0 {
        return None;
    }

    let depth = -v.z;
    if depth >= near && depth <= far {
        let x_ndc = (f / aspect) * (v.x / -v.z);
        let y_ndc = f * (v.y / -v.z);

        let sx = ((x_ndc + 1.0) * 0.5 * width as f32).round() as i32;
        let sy = (((-y_ndc) + 1.0) * 0.5 * height as f32).round() as i32;

        if sx >= 0 && sx < width as i32 && sy >= 0 && sy < height as i32 {
            return Some((sx as u32, sy as u32, depth));
        }
    }
    None
}
