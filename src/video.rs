use std::fmt;
use std::fs::File;
use std::io::{self, Write};
use nalgebra::{Matrix4, Point3, Rotation3, Vector3};
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
    let total_frames = 60;
    let angle_step = 2.0 * std::f32::consts::PI / total_frames as f32;

    let mut frames = Vec::with_capacity(total_frames);

    println!("Render started.\nImage: {}x{}\nFrames: {}\nAngle step: {:.2} rad/frame", 
        width, height, total_frames, angle_step);

    for i in 0..total_frames {
        println!("\nProcessing frame {}/{}", i + 1, total_frames);

        let angle = i as f32 * angle_step;
        let rotation = Rotation3::from_euler_angles(0.0, angle, angle * 0.3);

        let mut img = ImageBuffer::new(width, height);
        let mut z_buffer = vec![f32::INFINITY; (width * height) as usize];

        // Render points with depth testing
        for point in &points {
            let rotated = rotation.transform_point(&point.pos);
            if let Some((sx, sy, depth)) = project_to_screen(
                rotated.coords, width, height, 60.0_f32.to_radians()
            ) {
                render_point(&mut img, &mut z_buffer, sx, sy, depth, point.color);
            }
        }

        // Render axes with improved visibility
        draw_axes(&mut img, width, height, &rotation, &mut z_buffer);

        // Add frame to GIF
        let rgba = DynamicImage::ImageRgb8(img).to_rgba8();
        frames.push(Frame::from_parts(rgba, 0, 0, Delay::from_numer_denom_ms(16, 1)));
    }

    encode_and_save_gif(frames)?;
    Ok(())
}

/// Improved axis drawing with tick marks
pub fn draw_axes(
    img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    width: u32,
    height: u32,
    rotation: &Rotation3<f32>,
    z_buffer: &mut [f32],
) {
    const AXIS_LENGTH: f32 = 3.5;
    const AXIS_STEPS: i32 = 50;
    const TICK_INTERVAL: f32 = 0.5;

    // X Axis (Red) with ticks
    draw_axis(rotation, AXIS_LENGTH, 'x', Rgb([255, 50, 50]), img, width, height, z_buffer);
    // Y Axis (Green) with ticks
    draw_axis(rotation, AXIS_LENGTH, 'y', Rgb([50, 255, 50]), img, width, height, z_buffer);
    // Z Axis (Blue) with ticks
    draw_axis(rotation, AXIS_LENGTH, 'z', Rgb([50, 50, 255]), img, width, height, z_buffer);

    // Draw axis ticks
    for t in (-(AXIS_LENGTH as i32)..=(AXIS_LENGTH as i32)).map(|t| t as f32) {
        if t % TICK_INTERVAL == 0.0 && t != 0.0 {
            draw_tick(rotation, t, 'x', img, width, height, z_buffer);
            draw_tick(rotation, t, 'y', img, width, height, z_buffer);
            draw_tick(rotation, t, 'z', img, width, height, z_buffer);
        }
    }
}

fn draw_axis(
    rotation: &Rotation3<f32>,
    length: f32,
    axis: char,
    color: Rgb<u8>,
    img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    width: u32,
    height: u32,
    z_buffer: &mut [f32],
) {
    let steps = (length * 2.0) as i32 * 10;
    for t in -steps..=steps {
        let t = t as f32 / 10.0;
        let pt = match axis {
            'x' => rotation * Vector3::new(t, 0.0, 0.0),
            'y' => rotation * Vector3::new(0.0, t, 0.0),
            'z' => rotation * Vector3::new(0.0, 0.0, t),
            _ => continue,
        };
        draw_line(pt, img, width, height, z_buffer, color);
    }
}

fn draw_tick(
    rotation: &Rotation3<f32>,
    position: f32,
    axis: char,
    img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    width: u32,
    height: u32,
    z_buffer: &mut [f32],
) {
    let tick_size = 0.1;
    for offset in [-tick_size, tick_size] {
        let pt = match axis {
            'x' => rotation * Vector3::new(position, offset, 0.0),
            'y' => rotation * Vector3::new(offset, position, 0.0),
            'z' => rotation * Vector3::new(offset, 0.0, position),
            _ => continue,
        };
        draw_line(pt, img, width, height, z_buffer, Rgb([200, 200, 200]));
    }
}

fn render_point(
    img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    z_buffer: &mut [f32],
    x: u32,
    y: u32,
    depth: f32,
    color: Rgb<u8>,
) {
    let size = 2;
    for dx in 0..size {
        for dy in 0..size {
            let px = x.saturating_add(dx).min(img.width() - 1);
            let py = y.saturating_add(dy).min(img.height() - 1);
            let idx = (py * img.width() + px) as usize;
            if depth < z_buffer[idx] {
                z_buffer[idx] = depth;
                img.put_pixel(px, py, color);
            }
        }
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

fn encode_and_save_gif(frames: Vec<Frame>) -> Result<(), VideoError> {
    println!("\nEncoding GIF...");
    let mut gif_data = Vec::new();
    {
        let mut encoder = GifEncoder::new(&mut gif_data);
        encoder.set_repeat(Repeat::Infinite)?;
        for frame in frames {
            encoder.encode_frame(frame)?;
        }
    }

    println!("Saving GIF...");
    let mut file = File::create("graph.gif")?;
    file.write_all(&gif_data)?;

    println!("Displaying preview...");
    display_gif(&gif_data)?;

    Ok(())
}

/// Enhanced projection system with proper depth handling
fn project_to_screen(
    point: Vector3<f32>,
    width: u32,
    height: u32,
    fov: f32,
) -> Option<(u32, u32, f32)> {
    let camera_pos = Vector3::new(3.0, 2.0, 5.0);
    let view = Matrix4::look_at_rh(
        &Point3::from(camera_pos),
        &Point3::origin(),
        &Vector3::y(),
    );

    let view_point = view.transform_vector(&point);

    // Only cull points behind camera (positive Z in view space)
    if view_point.z > 0.0 {
        return None;
    }

    let aspect = width as f32 / height as f32;
    let f = 1.0 / (fov * 0.5).tan();
    
    let x_ndc = (f * view_point.x) / (aspect * view_point.z);
    let y_ndc = (f * view_point.y) / view_point.z;

    // Convert to screen coordinates with dynamic scaling
    let sx = ((x_ndc + 1.0) * 0.45 * width as f32).clamp(0.0, width as f32 - 1.0) as u32;
    let sy = ((1.0 - y_ndc) * 0.5 * height as f32).clamp(0.0, height as f32 - 1.0) as u32;

    Some((sx, sy, view_point.z.abs()))
}
