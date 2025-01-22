use crate::display::{display_gif, DisplayError};
use crate::embed::Point3D;
use gif::{Encoder as GifEncoder, Frame as GifFrame, Repeat};
use image::{ImageBuffer, Rgba, RgbaImage};
use nalgebra as na;
use std::io::Cursor;

/// Draws a line between two points using Bresenham's algorithm.
fn draw_line(
    img: &mut RgbaImage,
    (x0, y0): (u32, u32),
    (x1, y1): (u32, u32),
    color: Rgba<u8>,
) {
    let (mut x0, mut y0) = (x0 as i32, y0 as i32);
    let (x1, y1) = (x1 as i32, y1 as i32);
    
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    let width = img.width() as i32;
    let height = img.height() as i32;
    
    loop {
        if x0 >= 0 && x0 < width && y0 >= 0 && y0 < height {
            img.put_pixel(x0 as u32, y0 as u32, color);
        }
        
        if x0 == x1 && y0 == y1 {
            break;
        }
        
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

/// Creates a rotating 3D plot with angled camera view
pub fn make_video(points: &[Point3D]) -> Result<(), DisplayError> {
    // Increased resolution
    let width = 800;
    let height = 800;
    
    let max_dist = points
        .iter()
        .map(|p| p.pos.coords.norm())
        .fold(0.0_f32, f32::max)
        .max(1.0_f32);

    let num_frames = 36;
    
    // Camera parameters
    let camera_elevation = 15.0_f32.to_radians(); // Camera angle above horizontal
    let camera_distance = 2.5 * max_dist; // Distance from origin

    // Combined rotation and projection function
    fn transform_point(
        point: &na::Point3<f32>,
        y_angle_deg: f32,
        elevation: f32,
        camera_dist: f32,
        max_dist: f32,
        width: u32,
        height: u32,
    ) -> Option<(u32, u32)> {
        // First rotate around Y axis
        let y_angle_rad = y_angle_deg.to_radians();
        let y_rotation = na::Rotation3::from_euler_angles(0.0, y_angle_rad, 0.0);
        let rotated = y_rotation.transform_point(point);
        
        // Then apply camera elevation rotation around X axis
        let x_rotation = na::Rotation3::from_euler_angles(elevation, 0.0, 0.0);
        let elevated = x_rotation.transform_point(&rotated);
        
        // Perspective projection
        let z_factor = (elevated.z + camera_dist) / camera_dist;
        if z_factor <= 0.0 { return None; }
        
        let perspective_x = elevated.x / z_factor;
        let perspective_y = elevated.y / z_factor;
        
        // Scale to image coordinates
        let half_w = width as f32 / 2.0;
        let half_h = height as f32 / 2.0;
        let scale = half_w / max_dist;
        
        let x_img = half_w + perspective_x * scale;
        let y_img = half_h - perspective_y * scale;
        
        if x_img < 0.0 || y_img < 0.0 || x_img >= width as f32 || y_img >= height as f32 {
            None
        } else {
            Some((x_img as u32, y_img as u32))
        }
    }

    // Define main axes
    let axes = [
        (na::Point3::new(-max_dist, 0.0, 0.0), na::Point3::new(max_dist, 0.0, 0.0)),
        (na::Point3::new(0.0, -max_dist, 0.0), na::Point3::new(0.0, max_dist, 0.0)),
        (na::Point3::new(0.0, 0.0, -max_dist), na::Point3::new(0.0, 0.0, max_dist)),
    ];

    // Generate tick marks (reduced frequency)
    let tick_count = 11; // Will create marks at -1.0, -0.8, -0.6, ..., 0.8, 1.0
    let mut tick_points: Vec<na::Point3<f32>> = Vec::new();
    
    fn generate_cube_vertices(center: na::Point3<f32>, size: f32) -> Vec<na::Point3<f32>> {
        let half = size / 2.0;
        vec![
            na::Point3::new(center.x - half, center.y - half, center.z - half),
            na::Point3::new(center.x + half, center.y - half, center.z - half),
            na::Point3::new(center.x - half, center.y + half, center.z - half),
            na::Point3::new(center.x + half, center.y + half, center.z - half),
            na::Point3::new(center.x - half, center.y - half, center.z + half),
            na::Point3::new(center.x + half, center.y - half, center.z + half),
            na::Point3::new(center.x - half, center.y + half, center.z + half),
            na::Point3::new(center.x + half, center.y + half, center.z + half),
        ]
    }

    for i in 0..tick_count {
        let fraction = i as f32 / (tick_count - 1) as f32;
        let val = -max_dist + 2.0 * max_dist * fraction;
        let cube_size = max_dist * 0.015; // Slightly smaller ticks for higher resolution
        
        // X-axis ticks
        tick_points.extend(generate_cube_vertices(na::Point3::new(val, 0.0, 0.0), cube_size));
        // Y-axis ticks
        tick_points.extend(generate_cube_vertices(na::Point3::new(0.0, val, 0.0), cube_size));
        // Z-axis ticks
        tick_points.extend(generate_cube_vertices(na::Point3::new(0.0, 0.0, val), cube_size));
    }

    let mut frame_buffers = Vec::with_capacity(num_frames);
    let axis_color = Rgba([255u8, 255u8, 255u8, 255u8]);
    let tick_color = Rgba([128u8, 128u8, 128u8, 255u8]);

    for frame_idx in 0..num_frames {
        let angle_deg = (frame_idx as f32) * (360.0 / num_frames as f32);
        let mut img: RgbaImage = ImageBuffer::new(width, height);

        // Fill background with black
        for pixel in img.pixels_mut() {
            *pixel = Rgba([0, 0, 0, 255]);
        }

        // Draw tick marks
        for tick_pt in &tick_points {
            if let Some((px, py)) = transform_point(
                tick_pt,
                angle_deg,
                camera_elevation,
                camera_distance,
                max_dist,
                width,
                height,
            ) {
                img.put_pixel(px, py, tick_color);
            }
        }

        // Draw main axes
        for (start, end) in &axes {
            if let (Some(start_px), Some(end_px)) = (
                transform_point(start, angle_deg, camera_elevation, camera_distance, max_dist, width, height),
                transform_point(end, angle_deg, camera_elevation, camera_distance, max_dist, width, height),
            ) {
                draw_line(&mut img, start_px, end_px, axis_color);
            }
        }

        // Draw data points
        for pt in points {
            if let Some((px, py)) = transform_point(
                &pt.pos,
                angle_deg,
                camera_elevation,
                camera_distance,
                max_dist,
                width,
                height,
            ) {
                let rgb = pt.color.0;
                let rgba = Rgba([rgb[0], rgb[1], rgb[2], 255]);
                img.put_pixel(px, py, rgba);
            }
        }

        let raw_data = img.into_raw();
        frame_buffers.push(raw_data);
    }

    // Encode frames
    let mut gif_data = Vec::new();
    {
        let mut encoder = GifEncoder::new(&mut gif_data, width as u16, height as u16, &[])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        encoder
            .set_repeat(Repeat::Infinite)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        for mut raw in frame_buffers {
            let mut frame = GifFrame::from_rgba_speed(width as u16, height as u16, &mut raw, 10);
            frame.delay = 5;
            encoder
                .write_frame(&frame)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        }
    }

    display_gif(&gif_data)?;

    Ok(())
}
