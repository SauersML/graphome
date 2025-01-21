use std::thread;
use std::time::Duration;
use std::io;
use nalgebra::{Rotation3, Vector3};
use image::{ImageBuffer, Rgb, codecs::tga::TgaEncoder, ExtendedColorType, ImageError};
use crate::embed::Point3D;
use crate::display::{display_tga, DisplayError};

#[derive(Debug)]
pub enum VideoError {
    Io(std::io::Error),
    Image(ImageError),
    Display(DisplayError),
}

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
    loop {
        let rotation = Rotation3::from_euler_angles(0.0, angle, 0.0);
        let mut img = ImageBuffer::new(width, height);
        for point in &points {
            let rotated = rotation * point.pos;
            let camera_offset = 5.0;
            let transformed = Vector3::new(rotated.x, rotated.y, rotated.z - camera_offset);
            if let Some((sx, sy)) = project_to_screen(transformed, width, height, 60.0_f32.to_radians()) {
                let size = 2;
                for dx in 0..size {
                    for dy in 0..size {
                        let px = sx.saturating_add(dx).min(width - 1);
                        let py = sy.saturating_add(dy).min(height - 1);
                        img.put_pixel(px, py, point.color);
                    }
                }
            }
        }
        let mut tga_data = Vec::new();
        TgaEncoder::new(&mut tga_data).encode(
            img.as_raw(),
            width,
            height,
            ExtendedColorType::Rgb8
        )?;
        display_tga(&tga_data)?;
        angle += 0.01;
        thread::sleep(Duration::from_millis(16));
    }
}

fn project_to_screen(v: Vector3<f32>, width: u32, height: u32, fov: f32) -> Option<(u32, u32)> {
    let aspect = width as f32 / height as f32;
    let f = 1.0 / (fov * 0.5).tan();
    let near = 0.1;
    let far = 100.0;
    if v.z < -near && v.z > -far {
        let x = (f / aspect) * v.x / -v.z;
        let y = f * v.y / -v.z;
        let sx = ((x + 1.0) * 0.5 * width as f32).round() as i32;
        let sy = ((-y + 1.0) * 0.5 * height as f32).round() as i32;
        if sx >= 0 && sx < width as i32 && sy >= 0 && sy < height as i32 {
            return Some((sx as u32, sy as u32));
        }
    }
    None
}
