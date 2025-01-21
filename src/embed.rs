use nalgebra as na;
use rand_distr::{Distribution, Normal};
use image::Rgb;
use rand::thread_rng;

pub struct Point3D {
    pub pos: na::Point3<f32>,
    pub color: Rgb<u8>
}

pub fn embed() -> Vec<Point3D> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();
    let num_points = 1000;

    (0..num_points)
        .map(|_| {
            let x = normal.sample(&mut rng);
            let y = normal.sample(&mut rng); 
            let z = normal.sample(&mut rng);
            
            let r = ((x + 2.0) * 127.5) as u8;
            let g = ((y + 2.0) * 127.5) as u8;
            let b = ((z + 2.0) * 127.5) as u8;

            Point3D {
                pos: na::Point3::new(x, y, z),
                color: Rgb([r, g, b])
            }
        })
        .collect()
}
