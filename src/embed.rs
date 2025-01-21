use nalgebra as na;
use rand_distr::{Distribution, Normal};
use image::{ImageBuffer, Rgb, ImageEncoder};
use rand::thread_rng;
use std::error::Error;
use crate::display::display_tga;

struct Point3D {
    pos: na::Point3<f32>,
    color: Rgb<u8>
}

pub fn embed() -> Result<(), Box<dyn Error>> {
    // Generate random points with Gaussian distribution
    let normal = Normal::new(0.0, 1.0)?;
    let mut rng = thread_rng();
    let num_points = 1000;

    let points: Vec<Point3D> = (0..num_points)
        .map(|_| {
            let x = normal.sample(&mut rng);
            let y = normal.sample(&mut rng); 
            let z = normal.sample(&mut rng);
            
            // Color based on position
            let r = ((x + 2.0) * 127.5) as u8;
            let g = ((y + 2.0) * 127.5) as u8;
            let b = ((z + 2.0) * 127.5) as u8;

            Point3D {
                pos: na::Point3::new(x, y, z),
                color: Rgb([r, g, b])
            }
        })
        .collect();

    // Render loop
    let mut angle = 0.0f32;
    loop {
        // Create rotation matrix
        let rotation = na::Rotation3::from_euler_angles(0.0, angle, 0.0);
        
        // Create image buffer
        let width = 800;
        let height = 600;
        let mut img = ImageBuffer::new(width, height);

        // Project points to 2D
        for point in &points {
            // Apply rotation
            let rotated = rotation * point.pos;
            
            // Simple perspective projection
            let z_factor = (rotated.z + 3.0) / 6.0;  // Normalize z to 0-1 range
            if z_factor > 0.0 {  // Only draw points in front
                let screen_x = ((rotated.x / z_factor + 1.0) * width as f32 / 2.0) as u32;
                let screen_y = ((rotated.y / z_factor + 1.0) * height as f32 / 2.0) as u32;
                
                // Draw point if in bounds
                if screen_x < width && screen_y < height {
                    // Make closer points bigger and brighter
                    let size = (2.0 / z_factor) as u32;
                    let brightness = (1.0 / z_factor) as f32;
                    
                    for dx in 0..size {
                        for dy in 0..size {
                            let px = screen_x.saturating_add(dx).min(width-1);
                            let py = screen_y.saturating_add(dy).min(height-1);
                            
                            let color = Rgb([
                                (point.color[0] as f32 * brightness) as u8,
                                (point.color[1] as f32 * brightness) as u8,
                                (point.color[2] as f32 * brightness) as u8,
                            ]);
                            
                            img.put_pixel(px, py, color);
                        }
                    }
                }
            }
        }

        // Convert to TGA data
        let mut tga_data = Vec::new();
        image::tga::TgaEncoder::new(&mut tga_data)
            .encode(
                img.as_raw(),
                width,
                height,
                image::ColorType::Rgb8
            )?;  // Handle error

        // Display frame
        display_tga(&tga_data)?;

        // Update rotation
        angle += 0.02;
        
        // Frame timing
        std::thread::sleep(std::time::Duration::from_millis(16));
    }
}
