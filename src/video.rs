pub fn render(points: Vec<Point3D>) -> io::Result<()> {
    let mut angle = 0.0f32;
    loop {
        let rotation = na::Rotation3::from_euler_angles(0.0, angle, 0.0);
        let width = 800;
        let height = 600;
        let mut img = ImageBuffer::new(width, height);

        // Loop through all points to project them
        for point in &points {
            let rotated = rotation * point.pos;

            // Naive perspective
            let z_factor = (rotated.z + 3.0) / 6.0;
            if z_factor > 0.0 {
                // Arbitrary screen mapping
                let screen_x = ((rotated.x / z_factor + 1.0) * width as f32 / 2.0) as u32;
                let screen_y = ((rotated.y / z_factor + 1.0) * height as f32 / 2.0) as u32;
                
                if screen_x < width && screen_y < height {
                    // Arbitrary point sizing and brightness
                    let size = (2.0 / z_factor) as u32;
                    let brightness = (1.0 / z_factor) as f32;

                    // Draw "square" point with no depth check
                    for dx in 0..size {
                        for dy in 0..size {
                            let px = screen_x.saturating_add(dx).min(width-1);
                            let py = screen_y.saturating_add(dy).min(height-1);
                            
                            let color = Rgb([
                                (point.color[0] as f32 * brightness) as u8,
                                (point.color[1] as f32 * brightness) as u8,
                                (point.color[2] as f32 * brightness) as u8,
                            ]);
                            
                            img.put_pixel(px, py, color);
                        }
                    }
                }
            }
        }

        // Encode as TGA and display
        let mut tga_data = Vec::new();
        TgaEncoder::new(&mut tga_data)
            .encode(
                img.as_raw(),
                width,
                height,
                ExtendedColorType::Rgb8
            )?;
        display_tga(&tga_data)?;

        // Infinite loop
        angle += 0.02;
        std::thread::sleep(std::time::Duration::from_millis(16));
    }
}
