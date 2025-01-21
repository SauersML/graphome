// video.rs

use std::thread;
use std::time::Duration;
use std::io;
use std::fmt;
use nalgebra::{Rotation3, Vector3, Point3};
use image::{ImageBuffer, Rgb, codecs::tga::TgaEncoder, ExtendedColorType, ImageError};
use crate::embed::Point3D;
use crate::display::{display_tga, DisplayError};

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
    let mut angle = 0.0;
    let width = 800;
    let height = 600;
    let mut tga_data = Vec::new();

    loop {
        let rotation = Rotation3::from_euler_angles(0.0, angle, 0.0);
        let mut img = ImageBuffer::new(width, height);
        
        let mut z_buffer = vec![f32::INFINITY; (width * height) as usize];
        
        for point in &points {
            let rotated = rotation.transform_point(&point.pos);
            let camera_offset = 5.0;
            let transformed = Vector3::new(rotated.x, rotated.y, rotated.z - camera_offset);
    
            if let Some((sx, sy, depth)) = project_to_screen(transformed, width, height, 60.0_f32.to_radians()) {
                let size = 2;
                for dx in 0..size {
                    for dy in 0..size {
                        let px = sx.saturating_add(dx).min(width - 1);
                        let py = sy.saturating_add(dy).min(height - 1);
                        let idx = (py * width + px) as usize;
        
                        // Z-buffer test: only draw if this point is closer.
                        if depth < z_buffer[idx] {
                            z_buffer[idx] = depth;
                            img.put_pixel(px, py, point.color);
                        }
                    }
                }
            }
        }


        tga_data.clear();
        TgaEncoder::new(&mut tga_data).encode(img.as_raw(), width, height, ExtendedColorType::Rgb8)?;
        display_tga(&tga_data)?;

        angle = (angle + 0.01) % (2.0 * std::f32::consts::PI);
        thread::sleep(Duration::from_millis(16));
    }
}

fn project_to_screen(v: Vector3<f32>, width: u32, height: u32, fov: f32) -> Option<(u32, u32, f32)> {
    let aspect = width as f32 / height as f32;
    let half_fov_tan = (fov * 0.5).tan();
    let f = 1.0 / half_fov_tan;

    let near = 0.1;
    let far = 100.0;

    // Backface "culling": if z >= 0, skip.
    if v.z >= 0.0 {
        return None;
    }

    // We'll store "depth" as positive distance = -z
    let depth = -v.z;

    // Must be between near and far
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
