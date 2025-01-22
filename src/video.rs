// src/video.rs

use crate::display::{display_gif, DisplayError};
use crate::embed::Point3D;
use gif::{Encoder as GifEncoder, Frame as GifFrame, Repeat};
use image::{ImageBuffer, Rgba, RgbaImage};
use nalgebra as na;
use std::io::Cursor;

/// Creates a rotating 3D plot of the given points, draws simple XYZ axes, encodes it as a GIF,
/// and displays it in the terminal. No ticks are drawn on the axes.
pub fn make_video(points: &[Point3D]) -> Result<(), DisplayError> {
    // Determine a bounding distance so all points (and axes) fit in view.
    // This will be used to scale and center the plot.
    let max_dist = points
        .iter()
        .map(|p| p.pos.coords.norm())
        .fold(0.0_f32, f32::max)
        .max(1.0_f32);

    // Number of frames for the 360-degree rotation around the Y-axis.
    // 36 frames => each frame is 10 degrees => a full rotation for a looping GIF.
    let num_frames = 36;

    // Dimensions of each GIF frame in pixels.
    let width = 400;
    let height = 400;

    // Helper to rotate a 3D point around the Y-axis by a given angle (degrees).
    fn rotate_y(point: &na::Point3<f32>, angle_deg: f32) -> na::Point3<f32> {
        let angle_rad = angle_deg.to_radians();
        let rotation = na::Rotation3::from_euler_angles(0.0, angle_rad, 0.0);
        rotation.transform_point(point)
    }

    // Helper to project a 3D point (after rotation) into 2D coordinates in the image.
    // Uses an orthographic projection and centers the image so origin is in the middle.
    fn project_to_image(
        p: &na::Point3<f32>,
        max_dist: f32,
        width: u32,
        height: u32,
    ) -> Option<(u32, u32)> {
        let half_w = width as f32 / 2.0;
        let half_h = height as f32 / 2.0;

        let x_img = half_w + (p.x / max_dist) * half_w;
        // Flip Y so positive is upward in the image (instead of downward).
        let y_img = half_h - (p.y / max_dist) * half_h;

        // Only draw if within the image bounds.
        if x_img < 0.0 || y_img < 0.0 || x_img >= width as f32 || y_img >= height as f32 {
            None
        } else {
            Some((x_img as u32, y_img as u32))
        }
    }

    // Build axes as sets of points. Each axis goes from -max_dist to +max_dist.
    // We'll sample each axis in small increments so it appears like a line.
    let axis_color = Rgba([255u8, 255u8, 255u8, 255u8]);
    let axis_resolution = 60;
    let mut axis_points: Vec<na::Point3<f32>> = Vec::new();

    // X-axis
    for i in 0..axis_resolution {
        let fraction = i as f32 / (axis_resolution - 1) as f32;
        let val = -max_dist + 2.0 * max_dist * fraction;
        axis_points.push(na::Point3::new(val, 0.0, 0.0));
    }

    // Y-axis
    for i in 0..axis_resolution {
        let fraction = i as f32 / (axis_resolution - 1) as f32;
        let val = -max_dist + 2.0 * max_dist * fraction;
        axis_points.push(na::Point3::new(0.0, val, 0.0));
    }

    // Z-axis
    for i in 0..axis_resolution {
        let fraction = i as f32 / (axis_resolution - 1) as f32;
        let val = -max_dist + 2.0 * max_dist * fraction;
        axis_points.push(na::Point3::new(0.0, 0.0, val));
    }

    // We will store the raw RGBA data for each frame here before encoding.
    let mut frame_buffers = Vec::with_capacity(num_frames);

    for frame_idx in 0..num_frames {
        let angle_deg = (frame_idx as f32) * (360.0 / num_frames as f32);

        // Create a blank image for this frame.
        let mut img: RgbaImage = ImageBuffer::new(width, height);

        // Fill background with black.
        for pixel in img.pixels_mut() {
            *pixel = Rgba([0, 0, 0, 255]);
        }

        // Draw the axes by rotating each axis point, then projecting.
        for axis_pt in &axis_points {
            let rotated_pt = rotate_y(axis_pt, angle_deg);
            if let Some((px, py)) = project_to_image(&rotated_pt, max_dist, width, height) {
                img.put_pixel(px, py, axis_color);
            }
        }

        // Draw each data point.
        for pt in points {
            let rotated_pt = rotate_y(&pt.pos, angle_deg);
            if let Some((px, py)) = project_to_image(&rotated_pt, max_dist, width, height) {
                let rgb = pt.color.0;
                let rgba = Rgba([rgb[0], rgb[1], rgb[2], 255]);
                img.put_pixel(px, py, rgba);
            }
        }

        // Convert image into raw RGBA data and store.
        let raw_data = img.into_raw();
        frame_buffers.push(raw_data);
    }

    // Encode all frames into a GIF using the 'gif' crate directly.
    let mut gif_data = Vec::new();
    {
        // We need to create a GIF encoder with the given width/height.
        let mut encoder = GifEncoder::new(&mut gif_data, width as u16, height as u16, &[])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Repeat infinitely.
        encoder
            .set_repeat(Repeat::Infinite)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Write each frame.
        for raw in frame_buffers {
            // Each frame is RGBA, so we can build a Frame using gif::Frame::from_rgba_speed.
            let mut frame = GifFrame::from_rgba_speed(width as u16, height as u16, raw);
            // Delay is in 1/100ths of a second. 5 => 50 ms => ~20 FPS.
            frame.delay = 5;
            encoder
                .write_frame(&frame)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        }
    }

    // Finally, display the GIF in the terminal.
    display_gif(&gif_data)?;

    Ok(())
}
