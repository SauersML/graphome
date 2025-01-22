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

/// Generates the vertices of a small cube for a given center point and edge size.
/// (Kept here for compatibility if used elsewhere in the crate.)
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

/// Creates a short line segment for a tick mark on a specified axis at a given value.
/// Each segment is represented as a pair of start/end 3D points.
fn generate_tick_segment(axis: char, value: f32, tick_len: f32) -> (na::Point3<f32>, na::Point3<f32>) {
    // The offset is half of tick_len in ± directions orthogonal to the axis
    match axis {
        'x' => (
            na::Point3::new(value, -tick_len * 0.5, 0.0),
            na::Point3::new(value, tick_len * 0.5, 0.0),
        ),
        'y' => (
            na::Point3::new(-tick_len * 0.5, value, 0.0),
            na::Point3::new(tick_len * 0.5, value, 0.0),
        ),
        'z' => (
            na::Point3::new(-tick_len * 0.5, 0.0, value),
            na::Point3::new(tick_len * 0.5, 0.0, value),
        ),
        _ => (
            na::Point3::origin(),
            na::Point3::origin(),
        ),
    }
}

/// Creates a rotating 3D plot (with a single consistent rotation applied to both axes and data).
pub fn make_video(points: &[Point3D]) -> Result<(), DisplayError> {
    // Image resolution
    let width = 800;
    let height = 800;

    // Determine maximum distance from the origin of any point
    let max_dist = points
        .iter()
        .map(|p| p.pos.coords.norm())
        .fold(0.0_f32, f32::max)
        .max(1.0_f32);

    // Number of frames for the rotation
    let num_frames = 36;

    // Camera parameters
    // Camera parameters
    let camera_distance = 2.5 * max_dist;   // Distance from origin
    let camera_elev_deg: f32 = -30.0;        // Elevation in degrees (X-axis tilt)
    let camera_y_deg: f32 = 10.0;          // Y-axis tilt
    let camera_z_deg: f32 = 5.0;           // Z-axis tilt
    let camera_elev_rad = camera_elev_deg.to_radians();
    let camera_y_rad = camera_y_deg.to_radians();
    let camera_z_rad = camera_z_deg.to_radians();

    // This function applies the same rotation+projection to any 3D point (axes, ticks, data).
    //
    // Steps:
    // 1. Rotate around Y-axis by 'angle_rad'
    // 2. Rotate around X-axis by camera_elev_rad
    // 3. Apply perspective transform
    // 4. Convert to image coords
    fn transform_point(
        point: &na::Point3<f32>,
        angle_rad: f32,
        camera_elev: f32,
        camera_y_rad: f32,
        camera_z_rad: f32,
        camera_dist: f32,
        max_dist: f32,
        width: u32,
        height: u32,
    ) -> Option<(u32, u32)> {
        // Fixed camera angles first
        let y_fixed = na::Rotation3::from_axis_angle(&na::Vector3::y_axis(), camera_y_rad);
        let z_fixed = na::Rotation3::from_axis_angle(&na::Vector3::z_axis(), camera_z_rad);
        let point = z_fixed.transform_point(&y_fixed.transform_point(point));
        
        // Then animation rotation and elevation
        let y_rotation = na::Rotation3::from_axis_angle(&na::Vector3::y_axis(), angle_rad);
        let rotated = y_rotation.transform_point(&point);
        let x_rotation = na::Rotation3::from_axis_angle(&na::Vector3::x_axis(), camera_elev);
        let elevated = x_rotation.transform_point(&rotated);

        // We'll consider the camera to be looking along +Z, so we shift by camera_dist
        let z_factor = (elevated.z + camera_dist) / camera_dist;
        if z_factor <= 0.0 {
            return None;
        }
        let perspective_x = elevated.x / z_factor;
        let perspective_y = elevated.y / z_factor;

        // Convert to 2D image coords
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

    // Define main axes from -max_dist to +max_dist for X, Y, Z
    let axes = [
        (na::Point3::new(-max_dist, 0.0, 0.0), na::Point3::new(max_dist, 0.0, 0.0)), // X
        (na::Point3::new(0.0, -max_dist, 0.0), na::Point3::new(0.0, max_dist, 0.0)), // Y
        (na::Point3::new(0.0, 0.0, -max_dist), na::Point3::new(0.0, 0.0, max_dist)), // Z
    ];

    // Tick marks
    let tick_count = 11; // e.g. from -max_dist to +max_dist in steps
    let tick_step = (2.0 * max_dist) / (tick_count - 1) as f32;
    let tick_len = 0.05 * max_dist; // length of each tick line

    // Collect line segments (start & end) for all ticks on X, Y, Z
    let mut tick_segments: Vec<(na::Point3<f32>, na::Point3<f32>)> = Vec::new();
    for i in 0..tick_count {
        let v = -max_dist + i as f32 * tick_step;

        // For each axis, add a short orthogonal line
        let (tx1, tx2) = generate_tick_segment('x', v, tick_len);
        let (ty1, ty2) = generate_tick_segment('y', v, tick_len);
        let (tz1, tz2) = generate_tick_segment('z', v, tick_len);
        tick_segments.push((tx1, tx2));
        tick_segments.push((ty1, ty2));
        tick_segments.push((tz1, tz2));
    }

    // Store each frame as raw RGBA
    let mut frame_buffers = Vec::with_capacity(num_frames);

    // Colors
    let axis_color = Rgba([255, 255, 255, 255]);  // White
    let tick_color = Rgba([128, 128, 128, 255]);  // Gray

    // We do a full 360 rotation across 'num_frames'
    // angle goes from 0 to 2π
    for frame_idx in 0..num_frames {
        let fraction = frame_idx as f32 / num_frames as f32;
        let angle_rad = 2.0 * std::f32::consts::PI * fraction; // 0 .. 2π

        let mut img: RgbaImage = ImageBuffer::new(width, height);

        // Fill background
        for pixel in img.pixels_mut() {
            *pixel = Rgba([0, 0, 0, 255]);
        }

        // Draw tick segments
        for (start_pt, end_pt) in &tick_segments {
            if let (Some((sx, sy)), Some((ex, ey))) = (
                transform_point(start_pt, angle_rad, camera_elev_rad, camera_y_rad, camera_z_rad, camera_distance, max_dist, width, height),
                transform_point(end_pt,   angle_rad, camera_elev_rad, camera_y_rad, camera_z_rad, camera_distance, max_dist, width, height),
            ) {
                draw_line(&mut img, (sx, sy), (ex, ey), tick_color);
            }
        }

        // Draw main axes
        for (start, end) in &axes {
            if let (Some((sx, sy)), Some((ex, ey))) = (
                transform_point(start, angle_rad, camera_elev_rad, camera_y_rad, camera_z_rad, camera_distance, max_dist, width, height),
                transform_point(end,   angle_rad, camera_elev_rad, camera_y_rad, camera_z_rad, camera_distance, max_dist, width, height),
            ) {
                draw_line(&mut img, (sx, sy), (ex, ey), axis_color);
            }
        }

        // Draw data points
        for pt in points {
            if let Some((px, py)) = transform_point(
                &pt.pos,
                angle_rad,
                camera_elev_rad,
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

        // Convert image to raw RGBA buffer
        let raw_data = img.into_raw();
        frame_buffers.push(raw_data);
    }

    // Encode frames into a GIF
    let mut gif_data = Vec::new();
    {
        let mut encoder = GifEncoder::new(&mut gif_data, width as u16, height as u16, &[])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Infinite looping
        encoder
            .set_repeat(Repeat::Infinite)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Each frame is 5 hundredths of a second
        for mut raw in frame_buffers {
            let mut frame = GifFrame::from_rgba_speed(width as u16, height as u16, &mut raw, 10);
            frame.delay = 5;
            encoder
                .write_frame(&frame)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        }
    }

    // Display the resulting GIF
    display_gif(&gif_data)?;

    Ok(())
}
