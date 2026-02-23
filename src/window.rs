use indicatif::{ProgressBar, ProgressStyle};
use memmap2::MmapOptions;
use ndarray::prelude::*;
use ndarray_npy::write_npy;
use parking_lot::Mutex;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

// Buffer around requested range to retain edges that may connect into window boundaries.
const BUFFER_SIZE: usize = 10_000;

pub struct WindowConfig {
    pub start: usize,
    pub end: usize,
    pub window_size: usize,
    pub overlap: usize,
}

impl WindowConfig {
    pub fn new(start: usize, end: usize, window_size: usize, overlap: usize) -> Self {
        assert!(end > start, "End must be greater than start");
        assert!(
            window_size > overlap,
            "Window size must be greater than overlap"
        );
        assert!(
            window_size <= (end - start),
            "Window size must be less than or equal to range"
        );

        WindowConfig {
            start,
            end,
            window_size,
            overlap,
        }
    }

    pub fn generate_windows(&self) -> Vec<(usize, usize)> {
        let mut windows = Vec::with_capacity(
            ((self.end - self.start) / (self.window_size - self.overlap)).max(1),
        );
        let step_size = self.window_size - self.overlap;

        let mut window_start = self.start;
        while window_start + self.window_size <= self.end {
            windows.push((window_start, window_start + self.window_size));
            window_start += step_size;
        }

        if window_start < self.end {
            windows.push((self.end - self.window_size, self.end));
        }

        windows
    }
}

#[derive(Clone)]
struct EdgeList {
    data: Arc<Vec<(usize, usize)>>,
    node_index: Arc<Vec<Vec<usize>>>, // For each node in [index_start, index_end], edge indices touching it
    index_start: usize,
}

impl EdgeList {
    fn new(path: &Path, window_start: usize, window_end: usize) -> io::Result<Self> {
        let start_time = Instant::now();
        println!(
            "🔍 Loading edges for window {}-{}",
            window_start, window_end
        );

        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // Calculate seek positions with buffer zone
        let extended_start = window_start.saturating_sub(BUFFER_SIZE);
        let extended_end = window_end + BUFFER_SIZE;

        // Collect edges efficiently using pre-allocated vector
        let mut seen_edges = HashSet::new();

        // Scan full GAM because edge order is not guaranteed to be sorted by node id.
        let mut edges: Vec<(usize, usize)> = mmap
            .par_chunks_exact(8)
            .filter_map(|chunk| {
                let from = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as usize;
                let to = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]) as usize;

                // Keep edge if either node is in our extended window
                if (from >= extended_start && from <= extended_end)
                    || (to >= extended_start && to <= extended_end)
                {
                    Some((from, to))
                } else {
                    None
                }
            })
            .collect();

        // Remove duplicates while preserving order
        edges.retain(|&edge| seen_edges.insert(edge));
        edges.shrink_to_fit();

        println!(
            "📊 Loaded {} relevant edges in {:?}",
            edges.len(),
            start_time.elapsed()
        );

        // Build per-node edge index that does not assume any ordering in edge list.
        let index_size = extended_end - extended_start + 1;
        let mut node_index = vec![Vec::<usize>::new(); index_size];

        for (edge_idx, &(from, to)) in edges.iter().enumerate() {
            if (extended_start..=extended_end).contains(&from) {
                node_index[from - extended_start].push(edge_idx);
            }
            if to != from && (extended_start..=extended_end).contains(&to) {
                node_index[to - extended_start].push(edge_idx);
            }
        }

        println!(
            "✨ Indexed {} nodes in {:?}",
            index_size,
            start_time.elapsed()
        );

        Ok(Self {
            data: Arc::new(edges),
            node_index: Arc::new(node_index),
            index_start: extended_start,
        })
    }

    fn get_edges_for_window(&self, start: usize, end: usize) -> Vec<(usize, usize)> {
        // Collect candidate edges for each node in parallel, keeping per-node ordering.
        let per_node: Vec<Vec<(usize, usize)>> = (start..end)
            .into_par_iter()
            .map(|node| {
                if node < self.index_start || node - self.index_start >= self.node_index.len() {
                    return Vec::new();
                }

                let mut local_edges = Vec::new();

                for &edge_idx in &self.node_index[node - self.index_start] {
                    let edge = self.data[edge_idx];
                    if edge.0 < end && edge.1 < end && edge.0 >= start && edge.1 >= start {
                        local_edges.push(edge);
                    }
                }

                local_edges
            })
            .collect();

        let mut result: Vec<(usize, usize)> = per_node.into_iter().flatten().collect();

        // Deduplicate edges while preserving the first occurrence ordering.
        let mut seen = HashSet::new();
        result.retain(|&(from, to)| {
            let canonical = if from < to { (from, to) } else { (to, from) };
            seen.insert(canonical)
        });

        result
    }
}

fn compute_laplacian(
    edges: &[(usize, usize)],
    window_start: usize,
    window_size: usize,
) -> Array2<f64> {
    let mut laplacian = Array2::<f64>::zeros((window_size, window_size));
    let mut degrees = vec![0.0; window_size];

    // Process edges - each edge represents both directions
    for &(from, to) in edges {
        let i = from - window_start;
        let j = to - window_start;

        // Add edge in both directions
        laplacian[[i, j]] = -1.0;
        laplacian[[j, i]] = -1.0; // Symmetric

        // Count degrees for both nodes
        degrees[i] += 1.0;
        degrees[j] += 1.0;
    }

    // Fill diagonal with degrees
    laplacian.diag_mut().assign(&Array1::from(degrees));

    laplacian
}

pub fn parallel_extract_windows<P: AsRef<Path> + Sync>(
    edge_list_path: P,
    output_dir: P,
    config: WindowConfig,
) -> io::Result<()> {
    let start_time = Instant::now();
    println!("🚀 Starting parallel window extraction");
    println!("📊 Processing range: {} - {}", config.start, config.end);

    // Create output directory
    std::fs::create_dir_all(&output_dir)?;

    // Load edges with indexing - only for our range
    println!("📚 Loading and indexing edge list...");
    let edge_list = EdgeList::new(edge_list_path.as_ref(), config.start, config.end)?;

    // Convert output_dir to PathBuf once, before parallel processing
    let output_dir = PathBuf::from(output_dir.as_ref());
    let output_dir = Arc::new(output_dir);

    // Generate windows
    let windows = config.generate_windows();
    println!("📊 Processing {} windows in parallel", windows.len());

    // Setup progress bar
    let progress = Arc::new(Mutex::new(ProgressBar::new(windows.len() as u64)));
    progress.lock().set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} windows {msg}")
            .expect("Invalid progress bar template")
            .progress_chars("▰▱▱"),
    );

    // Process windows in parallel with optimized chunk size
    let chunk_size = (windows.len() / num_cpus::get().max(1)).max(1);
    windows.par_chunks(chunk_size).try_for_each(|chunk| {
        let edge_list = edge_list.clone();
        let output_dir = Arc::clone(&output_dir);
        let progress = Arc::clone(&progress);

        chunk.iter().try_for_each(|&(start, end)| {
            // Get relevant edges for this window
            let window_edges = edge_list.get_edges_for_window(start, end);

            // Compute Laplacian
            let window_size = end - start;
            let laplacian = compute_laplacian(&window_edges, start, window_size);

            // Save to NPY
            let output_file = output_dir.join(format!("laplacian_{:06}_{:06}.npy", start, end));
            write_npy(&output_file, &laplacian).map_err(io::Error::other)?;

            progress.lock().inc(1);
            Ok::<(), io::Error>(())
        })
    })?;

    progress.lock().finish_with_message("✨ Complete!");

    let duration = start_time.elapsed();
    println!("✨ Processed {} windows in {:.2?}", windows.len(), duration);

    Ok(())
}
