use ndarray::prelude::*;
use std::fs::File;
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use ndarray_npy::write_npy;
use rayon::prelude::*;
use memmap2::MmapOptions;
use indicatif::{ProgressBar, ProgressStyle};
use parking_lot::Mutex;

pub struct WindowConfig {
    pub start: usize,
    pub end: usize,
    pub window_size: usize,
    pub overlap: usize,
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

    pub fn generate_windows(&self) -> Vec<(usize, usize)> {
        let mut windows = Vec::with_capacity(
            ((self.end - self.start) / (self.window_size - self.overlap)).max(1)
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
    index: Arc<Vec<(usize, usize)>>, // (start_idx, end_idx) ranges for each node
}

impl EdgeList {
    fn new(path: &Path) -> io::Result<Self> {
        // Memory map the file for fastest possible reading
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        
        // Pre-allocate vectors with capacity
        let file_size = mmap.len();
        let edge_count = file_size / 8;
        let mut edges = Vec::with_capacity(edge_count);
        let mut max_node = 0usize;

        // Fast batch processing of memory mapped file
        for chunk in mmap.chunks_exact(8) {
            let from = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as usize;
            let to = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]) as usize;
            max_node = max_node.max(from).max(to);
            edges.push((from, to));
        }

        // Sort edges for better cache locality
        edges.par_sort_unstable();

        // Build index for O(1) node access
        let mut index = vec![(0, 0); max_node + 1];
        let mut current_node = 0;
        let mut start_idx = 0;

        for (i, &(from, _)) in edges.iter().enumerate() {
            while current_node < from {
                index[current_node] = (start_idx, i);
                current_node += 1;
                start_idx = i;
            }
        }
        while current_node <= max_node {
            index[current_node] = (start_idx, edges.len());
            current_node += 1;
        }

        Ok(Self {
            data: Arc::new(edges),
            index: Arc::new(index),
        })
    }

    fn get_edges_for_window(&self, start: usize, end: usize) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        
        // Use index to get relevant edge ranges
        for node in start..end {
            if node >= self.index.len() {
                break;
            }
            let (start_idx, end_idx) = self.index[node];
            for &edge in &self.data[start_idx..end_idx] {
                if edge.1 < end {
                    result.push(edge);
                }
            }
        }
        
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

    // Process edges
    for &(from, to) in edges {
        let i = from - window_start;
        let j = to - window_start;
        laplacian[[i, j]] = -1.0;
        degrees[i] += 1.0;
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

    // Create output directory
    std::fs::create_dir_all(&output_dir)?;

    // Load edges with indexing
    println!("📚 Loading and indexing edge list...");
    let edge_list = EdgeList::new(edge_list_path.as_ref())?;
    println!("✅ Loaded {} edges", edge_list.data.len());

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
    );

    // Process windows in parallel with chunk size optimization
    windows.par_chunks(num_cpus::get().max(1)).try_for_each(|chunk| {
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
            let output_file = output_dir.join(format!(
                "laplacian_{:06}_{:06}.npy",
                start, end
            ));
            write_npy(&output_file, &laplacian)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

            progress.lock().inc(1);
            Ok::<(), E>(())
        })
    })?;

    progress.lock().finish_with_message("✨ Complete!");
    
    let duration = start_time.elapsed();
    println!("✨ Processed {} windows in {:.2?}", windows.len(), duration);
    
    Ok::<(), E>(())
}
