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

/// Renders a multi-frame GIF of rotating 3D points with axes and tick marks.
pub fn render(points: Vec<Point3D>) -> Result<(), VideoError> {
    let width = 1600;
    let height = 1200;
    let total_frames = 60; // Number of frames in one full rotation
    let angle_step = 2.0 * std::f32::consts::PI / total_frames as f32;

    // Collect each frame as an RGBA buffer.
    let mut frames = Vec::with_capacity(total_frames);

    println!("Render started.");
    println!("Image width: {}", width);
    println!("Image height: {}", height);
    println!("Total frames to generate: {}", total_frames);
    println!("Angle step per frame: {}", angle_step);
    println!("Begin rendering frames...");

    for i in 0..total_frames {
        println!("\nProcessing frame {}/{}", i + 1, total_frames);

        let angle = i as f32 * angle_step;
        let rotation = Rotation3::from_euler_angles(0.0, angle, 0.0);

        // Create a blank image and Z-buffer
        let mut img = ImageBuffer::new(width, height);
        let mut z_buffer = vec![f32::INFINITY; (width * height) as usize];

        println!("  - Rotation matrix for this frame: {:?}", rotation);

        // Fill the image by projecting each point
        for point in &points {
            let rotated = rotation.transform_point(&point.pos);
            let camera_pos = Vector3::new(3.0, 2.0, 5.0);
            let transformed = Vector3::new(
                camera_pos.x - rotated.x,
                camera_pos.y - rotated.y,
                camera_pos.z - rotated.z
            );

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

        println!("  - Frame {} processed: image filled with points.", i + 1);

        // Draw axes and ticks after points
        draw_axes(&mut img, width, height, &rotation);

        // Convert RGB image to RGBA for the GIF frames.
        let rgba: RgbaImage = DynamicImage::ImageRgb8(img).to_rgba8();

        // Create a Frame with a small delay (~16 ms -> ~60 fps).
        let frame = Frame::from_parts(
            rgba,
            0, 0,
            Delay::from_numer_denom_ms(16, 1), // 16ms per frame
        );

        frames.push(frame);

        println!("  - Frame {} ready for GIF encoding.", i + 1);
    }

    println!("\nEncoding GIF...");
    let mut gif_data = Vec::new();
    {
        let mut encoder = GifEncoder::new(&mut gif_data);
        // Loop infinitely.
        encoder.set_repeat(Repeat::Infinite)?;

        for frame in frames {
            encoder.encode_frame(frame)?;
        }
    }

    println!("GIF encoding complete!");

    // Save the resulting GIF to disk.
    println!("\nSaving GIF to disk...");
    let mut file = File::create("graph.gif")?;
    file.write_all(&gif_data)?;
    println!("GIF saved successfully to 'graph.gif'.");

    // Display it once via viuer
    println!("\nDisplaying GIF...");
    display_gif(&gif_data)?;

    Ok(())
}

/// Draws axes with units and tick marks on the image.
pub fn draw_axes(
    img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    width: u32,
    height: u32,
    rotation: &Rotation3<f32>,
    z_buffer: &mut [f32],
) {
    const AXIS_LENGTH: f32 = 2.0;
    const AXIS_STEPS: usize = 100;

    // X Axis (Red)
    for t in -AXIS_STEPS..=AXIS_STEPS {
        let t = t as f32 / AXIS_STEPS as f32 * AXIS_LENGTH;
        let pt = rotation * Vector3::new(t, 0.0, 0.0);
        draw_line(pt, img, width, height, z_buffer, Rgb([255, 0, 0]));
    }

    // Y Axis (Green)
    for t in -AXIS_STEPS..=AXIS_STEPS {
        let t = t as f32 / AXIS_STEPS as f32 * AXIS_LENGTH;
        let pt = rotation * Vector3::new(0.0, t, 0.0);
        draw_line(pt, img, width, height, z_buffer, Rgb([0, 255, 0]));
    }

    // Z Axis (Blue)
    for t in -AXIS_STEPS..=AXIS_STEPS {
        let t = t as f32 / AXIS_STEPS as f32 * AXIS_LENGTH;
        let pt = rotation * Vector3::new(0.0, 0.0, t);
        draw_line(pt, img, width, height, z_buffer, Rgb([0, 0, 255]));
    }
}

fn draw_line(
    point: Vector3<f32>,
    img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    width: u32,
    height: u32,
    z_buffer: &mut [f32],
    color: Rgb<u8>,
) {
    if let Some((sx, sy, depth)) = project_to_screen(point, width, height, 60.0_f32.to_radians()) {
        let idx = (sy * width + sx) as usize;
        if depth < z_buffer[idx] {
            z_buffer[idx] = depth;
            img.put_pixel(sx, sy, color);
        }
    }
}

/// Helper function to get tick mark coordinates
fn get_tick_coordinates(center_x: u32, center_y: u32, size: u32, axis: char) -> Option<Vec<(u32, u32)>> {
    let mut coords = Vec::new();
    
    // Calculate safe ranges for tick marks
    let x_start = if center_x >= size { center_x - size } else { return None };
    let y_start = if center_y >= size { center_y - size } else { return None };
    
    // Create a horizontal or vertical line
    match axis {
        'x' => {
            // Horizontal ticks for X axis
            for dx in 0..=size*2 {
                coords.push((x_start + dx, center_y));
            }
        },
        'y' | 'z' => {
            // Vertical ticks for Y and Z axes
            for dy in 0..=size*2 {
                coords.push((center_x, y_start + dy));
            }
        },
        _ => return None
    }
    
    Some(coords)
}

/// Projects a 3D vector onto a 2D screen.
/// Returns (screen_x, screen_y, depth).
fn project_to_screen(
    point: Vector3<f32>,  // Point in WORLD SPACE
    width: u32,
    height: u32,
    fov: f32,
) -> Option<(u32, u32, f32)> {
    // Camera configuration
    let camera_pos = Vector3::new(0.0, 0.0, 5.0);  // Centered camera
    let look_at = Vector3::new(0.0, 0.0, 0.0);      // Look at origin
    let up = Vector3::y();                          // Up direction

    // Create view matrix
    let view = Matrix4::look_at_rh(
        &Point3::from(camera_pos),
        &Point3::from(look_at),
        &up,
    );

    // Transform point to view space
    let view_point = view.transform_vector(&point);

    // Skip points behind camera
    if view_point.z <= 0.0 {
        return None;
    }

    // Perspective projection
    let aspect = width as f32 / height as f32;
    let f = 1.0 / (fov * 0.5).tan();
    
    let x_proj = (f * view_point.x) / (aspect * view_point.z);
    let y_proj = (f * view_point.y) / view_point.z;

    // Convert to screen coordinates (centered)
    let sx = ((x_proj + 1.0) * 0.5 * width as f32).clamp(0.0, width as f32 - 1.0) as u32;
    let sy = ((1.0 - y_proj) * 0.5 * height as f32).clamp(0.0, height as f32 - 1.0) as u32;

    Some((sx, sy, view_point.z))
}
