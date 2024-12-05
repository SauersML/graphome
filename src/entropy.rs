use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::io::{self, Write, BufReader};
use std::sync::Arc;
use rayon::prelude::*;
use ndarray::{Array1, ArrayView1};
use ndarray_npy::ReadNpyExt;
use csv::WriterBuilder;
use termcolor::{ColorChoice, ColorSpec, StandardStream, WriteColor, Color};
use std::collections::VecDeque;

#[derive(Debug)]
struct WindowResult {
    start_node: u32,
    end_node: u32,
    ngec: f64,
}

fn parse_window_name(name: &str) -> Option<(u32, u32)> {
    let parts: Vec<&str> = name.split('_').collect();
    if parts.len() >= 3 {
        let start = parts[1].parse().ok()?;
        let end = parts[2].parse().ok()?;
        Some((start, end))
    } else {
        None
    }
}

fn compute_ngec(eigenvalues: &Array1<f64>) -> io::Result<f64> {
    let epsilon = 1e-9;
    let m = eigenvalues.len();
    
    // Check for negative eigenvalues
    if eigenvalues.iter().any(|&x| x < -epsilon) {
        println!("Warning: Negative eigenvalues found in window");
    }
    
    // Calculate sum of eigenvalues (ignoring small negatives)
    let sum_eigen: f64 = eigenvalues
        .iter()
        .filter(|&&x| x >= -epsilon)
        .sum();
    
    // Normalize eigenvalues
    let normalized = eigenvalues.mapv(|x| if x >= -epsilon { x / sum_eigen } else { 0.0 });
    
    // Compute entropy
    let entropy: f64 = normalized
        .iter()
        .filter(|&&x| x > 0.0)
        .map(|&x| -x * x.ln())
        .sum();
    
    let log_m = (m as f64).ln();
    Ok(entropy / log_m)
}

fn print_eigenvalue_distribution(eigenvalues: ArrayView1<f64>) {
    let stdout = StandardStream::stdout(ColorChoice::Always);
    let mut stdout = stdout.lock();
    
    // Normalize eigenvalues for visualization
    let max_val = eigenvalues.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_val = eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min);
    
    // Print header
    println!("\nEigenvalue Distribution [min: {:.2e}, max: {:.2e}]:", min_val, max_val);
    
    // Print color gradient
    for &value in eigenvalues.iter() {
        let intensity = ((value - min_val) / (max_val - min_val)).clamp(0.0, 1.0);
        
        let mut color_spec = ColorSpec::new();
        color_spec.set_fg(Some(Color::Rgb(
            (intensity * 255.0) as u8,
            0,
            ((1.0 - intensity) * 255.0) as u8,
        )));
        
        let _ = stdout.set_color(&color_spec);
        let _ = write!(stdout, "██");
    }
    let _ = stdout.reset();
    let _ = writeln!(stdout);
}

fn process_window(window_path: PathBuf) -> io::Result<WindowResult> {
    // Extract window range from directory name
    let dirname = window_path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid directory name"
        ))?;
    
    let (start_node, end_node) = parse_window_name(dirname)
        .ok_or_else(|| io::Error::new(
            io::ErrorKind::InvalidData,
            "Failed to parse window range"
        ))?;
    
    // Load eigenvalues
    let eigenvalues_path = window_path.join("eigenvalues.npy");
    let file = File::open(&eigenvalues_path)?;
    let reader = BufReader::new(file);
    let eigenvalues = Array1::<f64>::read_npy(reader)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    
    // Compute NGEC
    let ngec = compute_ngec(&eigenvalues)?;
    
    // Visualize distribution
    println!("\nWindow {}-{}", start_node, end_node);
    print_eigenvalue_distribution(eigenvalues.view());
    
    Ok(WindowResult {
        start_node,
        end_node,
        ngec,
    })
}

pub fn analyze_windows(output_dir: &Path) -> io::Result<()> {
    // Find all window directories
    let window_dirs: Vec<PathBuf> = fs::read_dir(output_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_dir() && 
            path.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("window_"))
                .unwrap_or(false)
        })
        .collect();
    
    println!("Found {} window directories", window_dirs.len());
    
    // Process windows in parallel
    let window_dirs = Arc::new(window_dirs);
    let results: Vec<WindowResult> = window_dirs
        .par_iter()
        .filter_map(|dir| {
            match process_window(dir.clone()) {
                Ok(result) => Some(result),
                Err(e) => {
                    eprintln!("Error processing {:?}: {}", dir, e);
                    None
                }
            }
        })
        .collect();
    
    // Save results to CSV
    let csv_path = output_dir.join("ngec_results.csv");
    let mut writer = WriterBuilder::new()
        .has_headers(true)
        .from_path(&csv_path)?;
    
    writer.write_record(&["start_node", "end_node", "ngec"])?;
    
    for result in results {
        writer.write_record(&[
            result.start_node.to_string(),
            result.end_node.to_string(),
            result.ngec.to_string(),
        ])?;
    }
    
    println!("\nResults saved to: {:?}", csv_path);
    Ok(())
}
