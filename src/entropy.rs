use csv::WriterBuilder;
use ndarray::{Array1, ArrayView1};
use ndarray_npy::ReadNpyExt;
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::{self, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};
use crate::sparse_spectral::estimate_ngec_hutchinson;

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

fn parse_sparse_window_stem(name: &str) -> Option<(u32, u32)> {
    // laplacian_<start>_<end>_values.npy
    if !name.starts_with("laplacian_") || !name.ends_with("_values.npy") {
        return None;
    }
    let stem = name.strip_prefix("laplacian_")?.strip_suffix("_values.npy")?;
    let mut parts = stem.split('_');
    let start = parts.next()?.parse::<u32>().ok()?;
    let end = parts.next()?.parse::<u32>().ok()?;
    Some((start, end))
}

fn compute_ngec(eigenvalues: &Array1<f64>) -> io::Result<f64> {
    let epsilon = 1e-9;
    let m = eigenvalues.len();

    // Check for negative eigenvalues
    if eigenvalues.iter().any(|&x| x < -epsilon) {
        println!("Warning: Negative eigenvalues found in window");
    }

    let sum_eigen: f64 = eigenvalues.iter().filter(|&&x| x >= -epsilon).sum();

    if sum_eigen == 0.0 {
        return Ok(0.0);
    }

    let entropy: f64 = eigenvalues
        .iter()
        .filter_map(|&x| {
            if x >= -epsilon {
                let normalized = x / sum_eigen;
                if normalized > 0.0 {
                    Some(-normalized * normalized.ln())
                } else {
                    None
                }
            } else {
                None
            }
        })
        .sum();

    let log_m = (m as f64).ln();
    Ok(entropy / log_m)
}

fn print_eigenvalue_distribution(eigenvalues: ArrayView1<f64>) {
    let stdout = StandardStream::stdout(ColorChoice::Always);
    let mut stdout = stdout.lock();

    // Normalize eigenvalues for visualization
    let max_val = eigenvalues
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_val = eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min);

    // Print header
    println!(
        "\nEigenvalue Distribution [min: {:.2e}, max: {:.2e}]:",
        min_val, max_val
    );

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
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid directory name"))?;

    let (start_node, end_node) = parse_window_name(dirname).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "Failed to parse window range")
    })?;

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

fn process_sparse_window(output_dir: &Path, start_node: u32, end_node: u32) -> io::Result<WindowResult> {
    let stem = format!("laplacian_{:06}_{:06}", start_node, end_node);
    let values_path = output_dir.join(format!("{}_values.npy", stem));
    let col_indices_path = output_dir.join(format!("{}_col_indices.npy", stem));
    let row_ptr_path = output_dir.join(format!("{}_row_ptr.npy", stem));
    let shape_path = output_dir.join(format!("{}_shape.npy", stem));

    let values = Array1::<f64>::read_npy(BufReader::new(File::open(values_path)?))
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let col_indices = Array1::<u64>::read_npy(BufReader::new(File::open(col_indices_path)?))
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let row_ptr = Array1::<u64>::read_npy(BufReader::new(File::open(row_ptr_path)?))
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let shape = Array1::<u64>::read_npy(BufReader::new(File::open(shape_path)?))
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    if shape.len() != 2 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid CSR shape"));
    }
    let n = shape[0] as usize;
    if shape[1] as usize != n {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "CSR matrix must be square"));
    }

    let row_ptr_vec = row_ptr.to_vec();
    let col_vec = col_indices.to_vec();
    let val_vec = values.to_vec();
    if row_ptr_vec.len() != n + 1 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid row_ptr length"));
    }
    let trace_l: f64 = (0..n)
        .map(|i| {
            let rs = row_ptr_vec[i] as usize;
            let re = row_ptr_vec[i + 1] as usize;
            let mut diag = 0.0;
            for k in rs..re {
                if col_vec[k] as usize == i {
                    diag += val_vec[k];
                }
            }
            diag
        })
        .sum();

    // Conservative spectral bound for Laplacian rows.
    let mut lambda_max = 1.0f64;
    for i in 0..n {
        let rs = row_ptr_vec[i] as usize;
        let re = row_ptr_vec[i + 1] as usize;
        let mut abs_sum = 0.0;
        for k in rs..re {
            abs_sum += val_vec[k].abs();
        }
        if abs_sum > lambda_max {
            lambda_max = abs_sum;
        }
    }

    let ngec = estimate_ngec_hutchinson(n, trace_l, lambda_max, 64, 20, |x, y| {
        for i in 0..n {
            let rs = row_ptr_vec[i] as usize;
            let re = row_ptr_vec[i + 1] as usize;
            let mut acc = 0.0;
            for k in rs..re {
                acc += val_vec[k] * x[col_vec[k] as usize];
            }
            y[i] = acc;
        }
    })?;

    Ok(WindowResult {
        start_node,
        end_node,
        ngec,
    })
}

pub fn analyze_windows(output_dir: &Path) -> io::Result<()> {
    // Legacy mode: window_* directories containing eigenvalues.npy
    let window_dirs: Vec<PathBuf> = fs::read_dir(output_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("window_"))
                    .unwrap_or(false)
        })
        .collect();

    let results: Vec<WindowResult> = if !window_dirs.is_empty() {
        println!("Found {} window directories (legacy eigenvalue mode)", window_dirs.len());
        let window_dirs = Arc::new(window_dirs);
        window_dirs
            .par_iter()
            .filter_map(|dir| match process_window(dir.clone()) {
                Ok(result) => Some(result),
                Err(e) => {
                    eprintln!("Error processing {:?}: {}", dir, e);
                    None
                }
            })
            .collect()
    } else {
        // Sparse mode: flat CSR files from extract-windows.
        let stems: Vec<(u32, u32)> = fs::read_dir(output_dir)?
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| entry.file_name().to_str().map(|s| s.to_string()))
            .filter_map(|name| parse_sparse_window_stem(&name))
            .collect();
        println!("Found {} sparse window Laplacians (CSR mode)", stems.len());
        let stems = Arc::new(stems);
        stems
            .par_iter()
            .filter_map(|&(start_node, end_node)| {
                match process_sparse_window(output_dir, start_node, end_node) {
                    Ok(result) => Some(result),
                    Err(e) => {
                        eprintln!(
                            "Error processing sparse window {}-{}: {}",
                            start_node, end_node, e
                        );
                        None
                    }
                }
            })
            .collect()
    };

    // Save results to CSV
    let csv_path = output_dir.join("ngec_results.csv");
    let mut writer = WriterBuilder::new()
        .has_headers(true)
        .from_path(&csv_path)?;

    writer.write_record(["start_node", "end_node", "ngec"])?;

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
