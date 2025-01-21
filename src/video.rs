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

pub fn render(points: Vec<Point3D>) -> Result<(), VideoError> {
    const WIDTH: u32 = 1600;
    const HEIGHT: u32 = 1200;
    const TOTAL_FRAMES: usize = 60;
    const FOV: f32 = std::f32::consts::PI / 3.0; // 60 degrees
    
    let mut frames = Vec::with_capacity(TOTAL_FRAMES);
    let camera_pos = Point3::new(0.0, 0.0, 15.0); // Centered camera
    let look_at = Point3::origin();
    let up = Vector3::y();
    let view = Matrix4::look_at_rh(&camera_pos, &look_at, &up);

    println!("Rendering {} frames...", TOTAL_FRAMES);

    for frame_num in 0..TOTAL_FRAMES {
        let progress = (frame_num + 1) as f32 / TOTAL_FRAMES as f32;
        let angle = progress * 2.0 * std::f32::consts::PI;
        let rotation = Rotation3::from_euler_angles(angle, angle * 0.5, 0.0);

        let mut img = ImageBuffer::new(WIDTH, HEIGHT);
        let mut z_buffer = vec![f32::INFINITY; (WIDTH * HEIGHT) as usize];

        // Draw axes first
        draw_axes(&mut img, WIDTH, HEIGHT, &rotation, &view, &mut z_buffer, FOV);

        // Draw points
        for point in &points {
            let rotated = rotation.transform_point(&point.pos);
            if let Some((sx, sy, depth)) = project(view, rotated.coords, WIDTH, HEIGHT, FOV) {
                render_point(&mut img, &mut z_buffer, sx, sy, depth, point.color);
            }
        }

        frames.push(Frame::from_parts(
            DynamicImage::ImageRgb8(img).to_rgba8(),
            0, 0,
            Delay::from_numer_denom_ms(16, 1)
        ));
    }

    let mut gif_data = Vec::new();
    {
        let mut encoder = GifEncoder::new(&mut gif_data);
        encoder.set_repeat(Repeat::Infinite)?;
        for frame in frames {
            encoder.encode_frame(frame)?;
        }
    }

    File::create("graph.gif")?.write_all(&gif_data)?;
    display_gif(&gif_data)?;

    Ok(())
}

fn draw_axes(
    img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    width: u32,
    height: u32,
    rotation: &Rotation3<f32>,
    view: &Matrix4<f32>,
    z_buffer: &mut [f32],
    fov: f32,
) {
    const AXIS_LENGTH: f32 = 10.0;
    const AXIS_STEPS: i32 = 200;

    let axes = [
        (Vector3::x(), Rgb([255, 0, 0])),   // Red X
        (Vector3::y(), Rgb([0, 255, 0])),   // Green Y
        (Vector3::z(), Rgb([0, 0, 255]))    // Blue Z
    ];

    for (dir, color) in axes {
        for t in (-AXIS_STEPS..=AXIS_STEPS).map(|t| {
            t as f32 / AXIS_STEPS as f32 * AXIS_LENGTH
        }) {
            let world_pos = rotation * (dir * t);
            if let Some((sx, sy, depth)) = project(*view, world_pos, width, height, fov) {
                let idx = (sy * width + sx) as usize;
                if depth < z_buffer[idx] {
                    z_buffer[idx] = depth;
                    img.put_pixel(sx, sy, color);
                }
            }
        }
    }
}

fn project(
    view: Matrix4<f32>,
    point: Vector3<f32>,
    width: u32,
    height: u32,
    fov: f32,
) -> Option<(u32, u32, f32)> {
    let view_point = view.transform_vector(&point);
    
    // Cull points behind camera
    if view_point.z > 0.0 {
        return None;
    }

    let aspect = width as f32 / height as f32;
    let f = 1.0 / (fov * 0.5).tan();

    // Perspective projection
    let x_ndc = (f * view_point.x) / (aspect * -view_point.z);
    let y_ndc = (f * view_point.y) / -view_point.z;

    // Convert to screen coordinates (center at (width/2, height/2))
    let sx = ((x_ndc + 1.0) * 0.5 * width as f32).clamp(0.0, width as f32 - 1.0) as u32;
    let sy = ((1.0 - y_ndc) * 0.5 * height as f32).clamp(0.0, height as f32 - 1.0) as u32;

    Some((sx, sy, -view_point.z))
}

fn render_point(
    img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    z_buffer: &mut [f32],
    x: u32,
    y: u32,
    depth: f32,
    color: Rgb<u8>,
) {
    // Draw 2x2 pixel square for visibility
    for dx in 0..2 {
        for dy in 0..2 {
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

fn get_tick_coordinates(center_x: u32, center_y: u32, size: u32, axis: char) -> Option<Vec<(u32, u32)>> {
    let mut coords = Vec::new();
    let x_start = center_x.saturating_sub(size);
    let y_start = center_y.saturating_sub(size);
    
    match axis {
        'x' => (0..=size*2).for_each(|dx| coords.push((x_start + dx, center_y))),
        'y' | 'z' => (0..=size*2).for_each(|dy| coords.push((center_x, y_start + dy))),
        _ => return None
    }
    
    Some(coords)
}
