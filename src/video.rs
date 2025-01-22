use crate::display::{display_gif, DisplayError};
use crate::embed::Point3D;
use gif::{Encoder as GifEncoder, Frame as GifFrame, Repeat};
use image::{ImageBuffer, Rgba, RgbaImage};
use nalgebra as na;
use std::io::Cursor;

/// Draws a line between two points using Bresenham's algorithm
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

/// Creates a rotating 3D plot of the given points, draws continuous XYZ axes, encodes it as a GIF,
/// and displays it in the terminal.
pub fn make_video(points: &[Point3D]) -> Result<(), DisplayError> {
    let max_dist = points
        .iter()
        .map(|p| p.pos.coords.norm())
        .fold(0.0_f32, f32::max)
        .max(1.0_f32);

    let num_frames = 36;
    let width = 400;
    let height = 400;

    fn rotate_y(point: &na::Point3<f32>, angle_deg: f32) -> na::Point3<f32> {
        let angle_rad = angle_deg.to_radians();
        let rotation = na::Rotation3::from_euler_angles(0.0, angle_rad, 0.0);
        rotation.transform_point(point)
    }

    fn project_to_image(
        p: &na::Point3<f32>,
        max_dist: f32,
        width: u32,
        height: u32,
    ) -> Option<(u32, u32)> {
        let half_w = width as f32 / 2.0;
        let half_h = height as f32 / 2.0;

        let x_img = half_w + (p.x / max_dist) * half_w;
        let y_img = half_h - (p.y / max_dist) * half_h;

        if x_img < 0.0 || y_img < 0.0 || x_img >= width as f32 || y_img >= height as f32 {
            None
        } else {
            Some((x_img as u32, y_img as u32))
        }
    }

    // Define axis endpoints
    let axis_color = Rgba([255u8, 255u8, 255u8, 255u8]);
    let axes = [
        // X-axis: from (-max_dist, 0, 0) to (max_dist, 0, 0)
        (na::Point3::new(-max_dist, 0.0, 0.0), na::Point3::new(max_dist, 0.0, 0.0)),
        // Y-axis: from (0, -max_dist, 0) to (0, max_dist, 0)
        (na::Point3::new(0.0, -max_dist, 0.0), na::Point3::new(0.0, max_dist, 0.0)),
        // Z-axis: from (0, 0, -max_dist) to (0, 0, max_dist)
        (na::Point3::new(0.0, 0.0, -max_dist), na::Point3::new(0.0, 0.0, max_dist)),
    ];

    let mut frame_buffers = Vec::with_capacity(num_frames);

    for frame_idx in 0..num_frames {
        let angle_deg = (frame_idx as f32) * (360.0 / num_frames as f32);

        let mut img: RgbaImage = ImageBuffer::new(width, height);

        // Fill background with black
        for pixel in img.pixels_mut() {
            *pixel = Rgba([0, 0, 0, 255]);
        }

        // Draw the axes as continuous lines
        for (start, end) in &axes {
            let rotated_start = rotate_y(start, angle_deg);
            let rotated_end = rotate_y(end, angle_deg);
            
            if let (Some(start_px), Some(end_px)) = (
                project_to_image(&rotated_start, max_dist, width, height),
                project_to_image(&rotated_end, max_dist, width, height),
            ) {
                draw_line(&mut img, start_px, end_px, axis_color);
            }
        }

        // Draw each data point
        for pt in points {
            let rotated_pt = rotate_y(&pt.pos, angle_deg);
            if let Some((px, py)) = project_to_image(&rotated_pt, max_dist, width, height) {
                let rgb = pt.color.0;
                let rgba = Rgba([rgb[0], rgb[1], rgb[2], 255]);
                img.put_pixel(px, py, rgba);
            }
        }

        let raw_data = img.into_raw();
        frame_buffers.push(raw_data);
    }

    // Encode frames into GIF
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
