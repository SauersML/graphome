use crate::display::{display_gif, DisplayError};
use crate::embed::Point3D;
use gif::{Encoder as GifEncoder, Frame as GifFrame, Repeat};
use image::{ImageBuffer, Rgba, RgbaImage};
use nalgebra as na;
use std::io::Cursor;

/// Draws a line in screen coordinates (x0, y0) -> (x1, y1) using Bresenham's algorithm.
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
/// (Retained for compatibility with other code.)
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

/// Returns two 3D points that define a short line segment for a tick mark on an axis.
fn generate_tick_segment(axis: char, value: f32, tick_len: f32) -> (na::Point3<f32>, na::Point3<f32>) {
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
        _ => (na::Point3::origin(), na::Point3::origin()),
    }
}

/// A more robust method for drawing each axis:
/// Subdivide the axis into many small segments in 3D, then draw them piecewise
/// after transforming to screen coordinates. This ensures the axis remains fully
/// drawn even if perspective transforms or partial clipping occur.
fn draw_axis_subdiv(
    img: &mut RgbaImage,
    start_pt: na::Point3<f32>,
    end_pt: na::Point3<f32>,
    steps: usize,
    transform_fn: &dyn Fn(&na::Point3<f32>) -> Option<(u32, u32)>,
    color: Rgba<u8>,
) {
    // Parametric line: P(t) = start + t*(end-start), for t in [0..1]
    let stepf = steps as f32;
    let delta = end_pt - start_pt;
    for i in 0..steps {
        let t0 = i as f32 / stepf;
        let t1 = (i as f32 + 1.0) / stepf;
        let p0_3d = start_pt + delta * t0;
        let p1_3d = start_pt + delta * t1;
        if let (Some(sxy), Some(exy)) = (transform_fn(&p0_3d), transform_fn(&p1_3d)) {
            draw_line(img, sxy, exy, color);
        }
    }
}

/// Applies the same transformations as in the user's existing code.
/// 1) Fixed camera rotations about Z and Y (camera_y_deg, camera_z_deg)
/// 2) Rotation about Y for the animation
/// 3) Rotation about X for camera elevation
/// 4) Perspective projection
fn transform_point(
    point: &na::Point3<f32>,
    angle_rad: f32,       // the "slow" rotation about Y for the animation
    camera_elev: f32,     // rotation about X
    camera_y_rad: f32,    // fixed camera rotation about Y
    camera_z_rad: f32,    // fixed camera rotation about Z
    camera_dist: f32,
    max_dist: f32,
    width: u32,
    height: u32,
) -> Option<(u32, u32)> {
    // 1) Pre-rotate the point by the camera's fixed Y and Z angles
    let y_fixed = na::Rotation3::from_axis_angle(&na::Vector3::y_axis(), camera_y_rad);
    let z_fixed = na::Rotation3::from_axis_angle(&na::Vector3::z_axis(), camera_z_rad);
    let rotated_fixed = z_fixed.transform_point(&y_fixed.transform_point(point));

    // 2) Animate rotation around Y
    let y_anim = na::Rotation3::from_axis_angle(&na::Vector3::y_axis(), angle_rad);
    let rotated_anim = y_anim.transform_point(&rotated_fixed);

    // 3) Rotate about X for camera elevation
    let x_rot = na::Rotation3::from_axis_angle(&na::Vector3::x_axis(), camera_elev);
    let elevated = x_rot.transform_point(&rotated_anim);

    // 4) Perspective projection (shift z by +camera_dist)
    let z_factor = (elevated.z + camera_dist) / camera_dist;
    if z_factor <= 0.0 {
        return None;
    }
    let perspective_x = elevated.x / z_factor;
    let perspective_y = elevated.y / z_factor;

    // Convert to 2D image coordinates
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

/// Creates a rotating 3D plot with a single axis of rotation (Y) plus optional fixed camera tilts.
/// The axes are now drawn using a more robust subdivided approach, ensuring they are fully visible.
pub fn make_video(points: &[Point3D]) -> Result<(), DisplayError> {
    let width = 800;
    let height = 800;

    // Determine maximum distance from the origin of any point
    let max_dist = points
        .iter()
        .map(|p| p.pos.coords.norm())
        .fold(0.0_f32, f32::max)
        .max(1.0_f32);

    let num_frames = 648;

    // Original camera parameters in the snippet:
    let camera_distance = 5.5 * max_dist;   // Distance from origin
    let camera_elev_deg: f32 = -30.0;       // Elevation in degrees (X-axis tilt)
    let camera_y_deg: f32 = 10.0;          // Y-axis tilt
    let camera_z_deg: f32 = 5.0;           // Z-axis tilt

    let camera_elev_rad = camera_elev_deg.to_radians();
    let camera_y_rad = camera_y_deg.to_radians();
    let camera_z_rad = camera_z_deg.to_radians();

    // We'll rotate about Y from 0..2π over all frames, but we can easily reduce to half-turn if desired.
    // The user asked for a single slow rotation around Y axis. Let's do a half-turn (180°) or full:
    // We'll do a full 360 for clarity, matching the snippet. Adjust if needed:
    let full_rotation = 2.0 * std::f32::consts::PI;

    // Main axes from -max_dist to +max_dist
    let axes = [
        (na::Point3::new(-max_dist, 0.0, 0.0), na::Point3::new(max_dist, 0.0, 0.0)), // X
        (na::Point3::new(0.0, -max_dist, 0.0), na::Point3::new(0.0, max_dist, 0.0)), // Y
        (na::Point3::new(0.0, 0.0, -max_dist), na::Point3::new(0.0, 0.0, max_dist)), // Z
    ];

    // Tick marks: from -max_dist to +max_dist in steps
    let tick_count = 11;
    let tick_step = (2.0 * max_dist) / (tick_count - 1) as f32;
    let tick_len = 0.05 * max_dist;

    let mut tick_segments = Vec::new();
    for i in 0..tick_count {
        let v = -max_dist + i as f32 * tick_step;
        let (tx1, tx2) = generate_tick_segment('x', v, tick_len);
        let (ty1, ty2) = generate_tick_segment('y', v, tick_len);
        let (tz1, tz2) = generate_tick_segment('z', v, tick_len);
        tick_segments.push((tx1, tx2));
        tick_segments.push((ty1, ty2));
        tick_segments.push((tz1, tz2));
    }

    // Prepare buffers for each frame
    let mut frame_buffers = Vec::with_capacity(num_frames);

    // Colors
    let axis_color = Rgba([255, 255, 255, 255]); // White
    let tick_color = Rgba([128, 128, 128, 255]); // Gray

    for frame_idx in 0..num_frames {
        let fraction = frame_idx as f32 / num_frames as f32;
        let angle_rad = full_rotation * fraction; // rotating around Y

        let mut img: RgbaImage = ImageBuffer::new(width, height);

        // Fill background with black
        for pixel in img.pixels_mut() {
            *pixel = Rgba([0, 0, 0, 255]);
        }

        // Closure to transform 3D -> 2D for axes, ticks, data
        let do_transform = |p: &na::Point3<f32>| {
            transform_point(
                p,
                angle_rad,
                camera_elev_rad,
                camera_y_rad,
                camera_z_rad,
                camera_distance,
                max_dist,
                width,
                height,
            )
        };

        // Draw each axis in subdivided segments for robustness
        let axis_subdiv_steps = 50;
        for (start, end) in &axes {
            draw_axis_subdiv(&mut img, *start, *end, axis_subdiv_steps, &do_transform, axis_color);
        }

        // Draw tick marks
        for (start_pt, end_pt) in &tick_segments {
            if let (Some((sx, sy)), Some((ex, ey))) = (do_transform(start_pt), do_transform(end_pt)) {
                draw_line(&mut img, (sx, sy), (ex, ey), tick_color);
            }
        }

        // Draw data points
        for pt in points {
            if let Some((px, py)) = do_transform(&pt.pos) {
                let rgb = pt.color.0;
                let rgba = Rgba([rgb[0], rgb[1], rgb[2], 255]);
                img.put_pixel(px, py, rgba);
            }
        }

        // Convert to raw buffer for GIF
        let raw_data = img.into_raw();
        frame_buffers.push(raw_data);
    }

    // Encode frames into a GIF
    let mut gif_data = Vec::new();
    {
        let mut encoder = GifEncoder::new(&mut gif_data, width as u16, height as u16, &[])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        encoder
            .set_repeat(Repeat::Infinite)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // 8 hundredths of a second per frame
        for mut raw in frame_buffers {
            let mut frame = GifFrame::from_rgba_speed(width as u16, height as u16, &mut raw, 10);
            frame.delay = 8;
            encoder
                .write_frame(&frame)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        }
    }

    // Display the resulting animation
    display_gif(&gif_data)?;

    Ok(())
}
