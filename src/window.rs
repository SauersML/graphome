use ndarray::prelude::*;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use ndarray_npy::write_npy;
use rayon::prelude::*;
 use std::io::Seek;

/// Configuration for windowed extraction
pub struct WindowConfig {
    pub start: usize,      // Start of full range
    pub end: usize,        // End of full range
    pub window_size: usize,// Size of each window
    pub overlap: usize,    // Size of overlap between windows
}

impl WindowConfig {
    pub fn new(start: usize, end: usize, window_size: usize, overlap: usize) -> Self {
        assert!(end > start, "End must be greater than start");
        assert!(window_size > overlap, "Window size must be greater than overlap");
        assert!(window_size <= (end - start), "Window size must be less than or equal to range");
        
        WindowConfig {
            start,
            end,
            window_size,
            overlap,
        }
    }

    /// Generate all window ranges based on configuration
    pub fn generate_windows(&self) -> Vec<(usize, usize)> {
        let mut windows = Vec::new();
        let step_size = self.window_size - self.overlap;
        
        let mut window_start = self.start;
        while window_start + self.window_size <= self.end {
            windows.push((window_start, window_start + self.window_size));
            window_start += step_size;
        }
        
        // Handle last window if there's remaining space
        if window_start < self.end {
            windows.push((self.end - self.window_size, self.end));
        }
        
        windows
    }
}

/// Fast Laplacian construction for a single window
fn fast_laplacian_for_window(
    reader: &mut BufReader<File>,
    window_start: usize,
    window_end: usize,
) -> io::Result<Array2<f64>> {
    let dim = window_end - window_start;
    let mut laplacian = Array2::<f64>::zeros((dim, dim));
    let mut degrees = Array1::<f64>::zeros(dim);
    let mut buffer = [0u8; 8];

    // Seek to start of file
    reader.seek(std::io::SeekFrom::Start(0))?;

    // Single pass through file
    while let Ok(_) = reader.read_exact(&mut buffer) {
        let from = u32::from_le_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]) as usize;
        let to = u32::from_le_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]) as usize;

        if (window_start..window_end).contains(&from) && (window_start..window_end).contains(&to) {
            let i = from - window_start;
            let j = to - window_start;
            
            laplacian[[i, j]] = -1.0;
            degrees[i] += 1.0;
        }
    }

    // Fill diagonal with degrees
    for i in 0..dim {
        laplacian[[i, i]] = degrees[i];
    }

    Ok(laplacian)
}

/// Process a single window and save its Laplacian
fn process_window(
    file_path: &Path,
    output_dir: &Path,
    window_start: usize,
    window_end: usize,
) -> io::Result<()> {
    let file = File::open(file_path)?;
    let mut reader = BufReader::new(file);
    
    // Compute Laplacian for this window
    let laplacian = fast_laplacian_for_window(&mut reader, window_start, window_end)?;
    
    // Create output filename with window range
    let output_file = output_dir.join(format!(
        "laplacian_{:06}_{:06}.npy",
        window_start,
        window_end
    ));
    
    // Save to .npy file
    write_npy(&output_file, &laplacian)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    
    Ok(())
}

/// Extract and save Laplacian matrices for all windows in parallel
pub fn parallel_extract_windows<P: AsRef<Path>>(
    edge_list_path: P,
    output_dir: P,
    config: WindowConfig,
) -> io::Result<()> {
    let start_time = Instant::now();
    println!("🚀 Starting parallel window extraction");
    
    // Create output directory if it doesn't exist
    std::fs::create_dir_all(&output_dir)?;
    
    // Generate all window ranges
    let windows = config.generate_windows();
    println!("📊 Processing {} windows", windows.len());
    
    // Convert paths to Arc for thread safety
    let edge_list_path = Arc::new(edge_list_path.as_ref().to_path_buf());
    let output_dir = Arc::new(output_dir.as_ref().to_path_buf());
    
    // Process windows in parallel
    windows.par_iter().try_for_each(|(start, end)| {
        let edge_list = edge_list_path.clone();
        let out_dir = output_dir.clone();
        
        process_window(
            edge_list.as_ref(),
            out_dir.as_ref(),
            *start,
            *end,
        )
    })?;

    let duration = start_time.elapsed();
    println!("✨ Completed processing {} windows in {:.2?}", windows.len(), duration);
    
    Ok(())
}

// Example usage
pub fn main() -> io::Result<()> {
    let config = WindowConfig::new(
        0,      // Start of range
        1000,   // End of range
        100,    // Window size
        50,     // Overlap size
    );
    
    parallel_extract_windows(
        "path/to/input.gam",
        "output/directory",
        config,
    )?;
    
    Ok(())
}
