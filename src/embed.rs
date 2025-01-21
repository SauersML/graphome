use nalgebra as na;
use rand_distr::{Distribution, Normal};
use image::Rgb;
use rand::thread_rng;
use std::io;
use std::io::Write;

pub struct Point3D {
    pub pos: na::Point3<f32>,
    pub color: Rgb<u8>,
}

pub fn embed(start_node: usize, end_node: usize, _input: &str) -> io::Result<Vec<Point3D>> {
    println!("Initializing embed function...");

    // Create a normal distribution centered around 0.0 with a standard deviation of 1.0
    let normal = Normal::new(0.0, 1.0).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    println!("Normal distribution created.");

    let mut rng = thread_rng(); // Random number generator
    println!("Random number generator initialized.");

    // Calculate the number of points to generate based on the input range
    let num_points = if end_node >= start_node {
        end_node - start_node + 1
    } else {
        0
    };
    println!("Calculated number of points to generate: {}", num_points);

    // Flush stdout to make sure the messages are printed
    io::stdout().flush()?;

    // Generate the 3D points
    let points: Vec<Point3D> = (0..num_points)
        .map(|_| {
            println!("Generating point...");

            // Sample random x, y, and z coordinates from the normal distribution
            let x = normal.sample(&mut rng);
            let y = normal.sample(&mut rng);
            let z = normal.sample(&mut rng);

            // Print the generated 3D coordinates for debugging
            println!("Generated 3D coordinates: ({}, {}, {})", x, y, z);

            // Map the x, y, z coordinates to RGB values (scaled between 0 and 255)
            let r = ((x + 2.0_f32) * 127.5_f32).max(0.0_f32).min(255.0_f32) as u8;
            let g = ((y + 2.0_f32) * 127.5_f32).max(0.0_f32).min(255.0_f32) as u8;
            let b = ((z + 2.0_f32) * 127.5_f32).max(0.0_f32).min(255.0_f32) as u8;

            // Print the generated RGB color
            println!("Generated color: ({}, {}, {})", r, g, b);

            // Create and return a Point3D object with the calculated position and color
            Point3D {
                pos: na::Point3::new(x, y, z),
                color: Rgb([r, g, b]),
            }
        })
        .collect();

    // Return the generated points
    Ok(points)
}
