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
pub fn draw_axes(img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, width: u32, height: u32, rotation: &Rotation3<f32>) {
    let axis_color = Rgb([255, 255, 255]); // White color for all axes
    let axis_length = 2.0; // Length of each axis in 3D space
    let steps = 100; // Number of points to draw per axis
    
    // Draw all three axes using proper 3D projection
    let camera_pos = Vector3::new(3.0, 2.0, 5.0);
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        
        // Create vectors for each axis in original position
        let mut x_vec = Vector3::new(axis_length * (2.0 * t - 1.0), 0.0, 0.0);
        let mut y_vec = Vector3::new(0.0, axis_length * (2.0 * t - 1.0), 0.0);
        let mut z_vec = Vector3::new(0.0, 0.0, axis_length * (2.0 * t - 1.0));

        // Apply rotation
        x_vec = rotation * x_vec;
        y_vec = rotation * y_vec;
        z_vec = rotation * z_vec;

        // Apply camera position
        x_vec.x -= camera_pos.x;
        x_vec.y -= camera_pos.y;
        x_vec.z -= camera_pos.z;
        y_vec.x -= camera_pos.x;
        y_vec.y -= camera_pos.y;
        y_vec.z -= camera_pos.z;
        z_vec.x -= camera_pos.x;
        z_vec.y -= camera_pos.y;
        z_vec.z -= camera_pos.z;
        
        // Project and draw X axis
        if let Some((sx, sy, _)) = project_to_screen(
            x_vec,
            width,
            height,
            60.0_f32.to_radians()
        ) {
            img.put_pixel(sx, sy, axis_color);
            
            // Tick marks at regular intervals
            if ((t * 2.0 - 1.0) * axis_length).abs().fract() < 0.01 {
                if let Some(tick_coords) = get_tick_coordinates(sx, sy, 6) {
                    for (tx, ty) in tick_coords {
                        if tx < width && ty < height {
                            img.put_pixel(tx, ty, axis_color);
                        }
                    }
                }
            }
        }
        
        // Project and draw Y axis
        if let Some((sx, sy, _)) = project_to_screen(
            y_vec,
            width,
            height,
            60.0_f32.to_radians()
        ) {
            img.put_pixel(sx, sy, axis_color);
            
            // Tick marks at regular intervals
            if ((t * 2.0 - 1.0) * axis_length).abs().fract() < 0.01 {
                if let Some(tick_coords) = get_tick_coordinates(sx, sy, 6) {
                    for (tx, ty) in tick_coords {
                        if tx < width && ty < height {
                            img.put_pixel(tx, ty, axis_color);
                        }
                    }
                }
            }
        }
        
        // Project and draw Z axis
        if let Some((sx, sy, _)) = project_to_screen(
            z_vec,
            width,
            height,
            60.0_f32.to_radians()
        ) {
            img.put_pixel(sx, sy, axis_color);
            
            // Tick marks at regular intervals
            if (t * axis_length).fract() < 0.01 {
                if let Some(tick_coords) = get_tick_coordinates(sx, sy, 6) {
                    for (tx, ty) in tick_coords {
                        if tx < width && ty < height {
                            img.put_pixel(tx, ty, axis_color);
                        }
                    }
                }
            }
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
