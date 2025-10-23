use image::Rgb;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use std::io;

pub struct Point3D {
    pub pos: [f32; 3],
    pub color: Rgb<u8>,
}

pub fn embed(start_node: usize, end_node: usize, _input: &str) -> io::Result<Vec<Point3D>> {
    // Create a normal distribution centered around 0.0 with a standard deviation of 1.0
    let normal = Normal::<f32>::new(0.0, 1.0).map_err(io::Error::other)?;

    let mut rng = thread_rng();

    // Calculate the number of points to generate based on the input range
    let num_points = if end_node >= start_node {
        end_node - start_node + 1
    } else {
        0
    };

    // Generate the 3D points
    let mut points = Vec::with_capacity(num_points);

    for _ in 0..num_points {
        // Sample random x, y, and z coordinates from the normal distribution
        let x = normal.sample(&mut rng);
        let y = normal.sample(&mut rng);
        let z = normal.sample(&mut rng);

        // Map the x, y, z coordinates to RGB values (scaled between 0 and 255)
        let r = ((x + 2.0_f32) * 127.5_f32).clamp(0.0_f32, 255.0_f32) as u8;
        let g = ((y + 2.0_f32) * 127.5_f32).clamp(0.0_f32, 255.0_f32) as u8;
        let b = ((z + 2.0_f32) * 127.5_f32).clamp(0.0_f32, 255.0_f32) as u8;

        // Create and return a Point3D object with the calculated position and color
        points.push(Point3D {
            pos: [x, y, z],
            color: Rgb([r, g, b]),
        });
    }

    // Return the generated points
    Ok(points)
}
