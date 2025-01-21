// embed.rs

use nalgebra as na;
use rand_distr::{Distribution, Normal};
use image::Rgb;
use rand::thread_rng;
use std::io;
use std::io::Write;

pub struct Point3D {
    pub pos: na::Point3<f32>,
    pub color: Rgb<u8>
}

pub fn embed(start_node: usize, end_node: usize, _input: &str) -> io::Result<Vec<Point3D>> {
    println!("Initializing embed function...");

    let normal = Normal::new(0.0, 1.0).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    println!("Normal distribution created.");

    let mut rng = thread_rng();
    println!("Random number generator initialized.");

    let num_points = if end_node >= start_node { end_node - start_node + 1 } else { 0 };
    println!("Calculated number of points to generate: {}", num_points);

    io::stdout().flush()?;

    Ok((0..num_points)
        .map(|_| {
            println!("Generating point...");

            let x = normal.sample(&mut rng);
            let y = normal.sample(&mut rng);
            let z = normal.sample(&mut rng);
            println!("Generated 3D coordinates: ({}, {}, {})", x, y, z);

            let r = ((x + 2.0_f32) * 127.5_f32).max(0.0_f32).min(255.0_f32) as u8;
            let g = ((y + 2.0_f32) * 127.5_f32).max(0.0_f32).min(255.0_f32) as u8;
            let b = ((z + 2.0_f32) * 127.5_f32).max(0.0_f32).min(255.0_f32) as u8;
            println!("Generated color: ({}, {}, {})", r, g, b);

            Point3D {
                pos: na::Point3::new(x, y, z),
                color: Rgb([r, g, b])
            }
        })
        .collect())
}
