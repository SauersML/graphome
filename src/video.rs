use crate::display::{display_gif, DisplayError};
use crate::embed::Point3D;
use gif::{Encoder as GifEncoder, Frame as GifFrame, Repeat};
use image::{ImageBuffer, Rgba, RgbaImage};

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale3(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn norm3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn rotate_x(v: [f32; 3], angle: f32) -> [f32; 3] {
    let (sin_a, cos_a) = angle.sin_cos();
    [
        v[0],
        cos_a * v[1] - sin_a * v[2],
        sin_a * v[1] + cos_a * v[2],
    ]
}

fn rotate_y(v: [f32; 3], angle: f32) -> [f32; 3] {
    let (sin_a, cos_a) = angle.sin_cos();
    [
        cos_a * v[0] + sin_a * v[2],
        v[1],
        -sin_a * v[0] + cos_a * v[2],
    ]
}

fn rotate_z(v: [f32; 3], angle: f32) -> [f32; 3] {
    let (sin_a, cos_a) = angle.sin_cos();
    [
        cos_a * v[0] - sin_a * v[1],
        sin_a * v[0] + cos_a * v[1],
        v[2],
    ]
}

/// Draws a line in screen coordinates (x0, y0) -> (x1, y1) using Bresenham's algorithm.
fn draw_line(img: &mut RgbaImage, (x0, y0): (u32, u32), (x1, y1): (u32, u32), color: Rgba<u8>) {
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

/// Returns two 3D points that define a short line segment for a tick mark on an axis.
fn generate_tick_segment(axis: char, value: f32, tick_len: f32) -> ([f32; 3], [f32; 3]) {
    match axis {
        'x' => ([value, -tick_len * 0.5, 0.0], [value, tick_len * 0.5, 0.0]),
        'y' => ([-tick_len * 0.5, value, 0.0], [tick_len * 0.5, value, 0.0]),
        'z' => ([-tick_len * 0.5, 0.0, value], [tick_len * 0.5, 0.0, value]),
        _ => ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    }
}

/// A more robust method for drawing each axis:
/// Subdivide the axis into many small segments in 3D, then draw them piecewise
/// after transforming to screen coordinates. This ensures the axis remains fully
/// drawn even if perspective transforms or partial clipping occur.
fn draw_axis_subdiv(
    img: &mut RgbaImage,
    start_pt: [f32; 3],
    end_pt: [f32; 3],
    steps: usize,
    transform_fn: &dyn Fn(&[f32; 3]) -> Option<(u32, u32)>,
    color: Rgba<u8>,
) {
    // Parametric line: P(t) = start + t*(end-start), for t in [0..1]
    let stepf = steps as f32;
    let delta = sub3(end_pt, start_pt);
    for i in 0..steps {
        let t0 = i as f32 / stepf;
        let t1 = (i as f32 + 1.0) / stepf;
        let p0_3d = add3(start_pt, scale3(delta, t0));
        let p1_3d = add3(start_pt, scale3(delta, t1));
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
    point: &[f32; 3],
    angle_rad: f32,    // the "slow" rotation about Y for the animation
    camera_elev: f32,  // rotation about X
    camera_y_rad: f32, // fixed camera rotation about Y
    camera_z_rad: f32, // fixed camera rotation about Z
    camera_dist: f32,
    max_dist: f32,
    width: u32,
    height: u32,
) -> Option<(u32, u32)> {
    let rotated_y = rotate_y(*point, camera_y_rad);
    let rotated_fixed = rotate_z(rotated_y, camera_z_rad);
    let rotated_anim = rotate_y(rotated_fixed, angle_rad);
    let elevated = rotate_x(rotated_anim, camera_elev);

    // 4) Perspective projection (shift z by +camera_dist)
    let z_factor = (elevated[2] + camera_dist) / camera_dist;
    if z_factor <= 0.0 {
        return None;
    }
    let perspective_x = elevated[0] / z_factor;
    let perspective_y = elevated[1] / z_factor;

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
        .map(|p| norm3(p.pos))
        .fold(0.0_f32, f32::max)
        .max(1.0_f32);

    let num_frames = 1024;

    // Original camera parameters in the snippet:
    let camera_distance = 5.5 * max_dist; // Distance from origin
    let camera_elev_deg: f32 = -30.0; // Elevation in degrees (X-axis tilt)
    let camera_y_deg: f32 = 10.0; // Y-axis tilt
    let camera_z_deg: f32 = 5.0; // Z-axis tilt

    let camera_elev_rad = camera_elev_deg.to_radians();
    let camera_y_rad = camera_y_deg.to_radians();
    let camera_z_rad = camera_z_deg.to_radians();

    // We'll rotate about Y from 0..2π over all frames, but we can easily reduce to half-turn if desired.
    // The user asked for a single slow rotation around Y axis. Let's do a half-turn (180°) or full:
    // We'll do a full 360 for clarity, matching the snippet. Adjust if needed:
    let full_rotation = 2.0 * std::f32::consts::PI;

    // Main axes from -max_dist to +max_dist
    let axes = [
        ([-max_dist, 0.0, 0.0], [max_dist, 0.0, 0.0]), // X
        ([0.0, -max_dist, 0.0], [0.0, max_dist, 0.0]), // Y
        ([0.0, 0.0, -max_dist], [0.0, 0.0, max_dist]), // Z
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
        let do_transform = |p: &[f32; 3]| {
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
            draw_axis_subdiv(
                &mut img,
                *start,
                *end,
                axis_subdiv_steps,
                &do_transform,
                axis_color,
            );
        }

        // Draw tick marks
        for (start_pt, end_pt) in &tick_segments {
            if let (Some((sx, sy)), Some((ex, ey))) = (do_transform(start_pt), do_transform(end_pt))
            {
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
            .map_err(std::io::Error::other)?;

        encoder
            .set_repeat(Repeat::Infinite)
            .map_err(std::io::Error::other)?;

        // 1 hundredth of a second per frame
        for mut raw in frame_buffers {
            let mut frame = GifFrame::from_rgba_speed(width as u16, height as u16, &mut raw, 10);
            frame.delay = 1;
            encoder
                .write_frame(&frame)
                .map_err(std::io::Error::other)?;
        }
    }

    // Spawn thread to save GIF in background
    let gif_data_clone = gif_data.clone();
    std::thread::spawn(move || {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        if let Err(e) = std::fs::write(format!("graph_{}.gif", timestamp), gif_data_clone) {
            eprintln!("Failed to save GIF: {}", e);
        }
    });

    // Display the resulting animation
    display_gif(&gif_data)?;

    Ok(())
}
