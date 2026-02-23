use image::Rgb;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::io;
use std::sync::Arc;

pub struct Point3D {
    pub pos: [f32; 3],
    pub color: Rgb<u8>,
}

pub fn embed(start_node: usize, end_node: usize, input: &str) -> io::Result<Vec<Point3D>> {
    // Create a normal distribution centered around 0.0 with a standard deviation of 1.0
    let normal = Normal::<f32>::new(0.0, 1.0).map_err(io::Error::other)?;
    let normal = Arc::new(normal);

    // Calculate the number of points to generate based on the input range
    let num_points = if end_node >= start_node {
        end_node - start_node + 1
    } else {
        0
    };

    if num_points == 0 {
        return Ok(Vec::new());
    }

    // Seed used to derive independent RNGs per point deterministically.
    let input_seed = input.bytes().fold(0u64, |acc, byte| {
        acc.wrapping_mul(131).wrapping_add(byte as u64)
    });
    let mut global_seed = input_seed
        ^ (start_node as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (end_node as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    // Mix the seed to avoid weak low-entropy cases.
    global_seed ^= global_seed >> 30;
    global_seed = global_seed.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    global_seed ^= global_seed >> 27;
    global_seed = global_seed.wrapping_mul(0x94D0_49BB_1331_11EB);
    global_seed ^= global_seed >> 31;

    // Generate the 3D points in parallel
    let points: Vec<Point3D> = (0..num_points)
        .into_par_iter()
        .map(|i| {
            // Derive a unique seed for the thread-local RNG without contention.
            let seed = global_seed
                .wrapping_add(i as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut rng = StdRng::seed_from_u64(seed);

            // Sample random x, y, and z coordinates from the normal distribution
            let dist = normal.as_ref();
            let x = dist.sample(&mut rng);
            let y = dist.sample(&mut rng);
            let z = dist.sample(&mut rng);

            // Map the x, y, z coordinates to RGB values (scaled between 0 and 255)
            let r = ((x + 2.0_f32) * 127.5_f32).clamp(0.0_f32, 255.0_f32) as u8;
            let g = ((y + 2.0_f32) * 127.5_f32).clamp(0.0_f32, 255.0_f32) as u8;
            let b = ((z + 2.0_f32) * 127.5_f32).clamp(0.0_f32, 255.0_f32) as u8;

            Point3D {
                pos: [x, y, z],
                color: Rgb([r, g, b]),
            }
        })
        .collect();

    Ok(points)
}
