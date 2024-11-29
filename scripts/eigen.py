import subprocess
import numpy as np
import os
import time
from pathlib import Path
import logging
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from typing import List, Tuple
import csv

def setup_logging():
    formatter = logging.Formatter('%(asctime)s - PID:%(process)d - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

def run_rust_extractor(
    matrix_path: str,
    output_dir: str,
    start_node: int,
    end_node: int,
    process_id: int
) -> str:
    """Run Rust binary to extract matrix window"""
    output_subdir = f"window_{start_node}_{end_node}"
    output_path = os.path.join(output_dir, output_subdir)
    os.makedirs(output_path, exist_ok=True)
    
    logger = logging.getLogger(f"PID:{process_id}")
    logger.info(f"Extracting window [{start_node}, {end_node}]")
    
    cmd = [
        "../target/release/graphome",
        "extract-matrices",  # Use ExtractMatrices command
        "--input", matrix_path,
        "--start-node", str(start_node),
        "--end-node", str(end_node),
        "--output", output_path
    ]
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time
        logger.info(f"Extraction complete in {elapsed:.2f}s")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Rust extractor failed: {e.stderr}")
        raise

def compute_eigen(npy_path: str, process_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute eigendecomposition using banded solver"""
    logger = logging.getLogger(f"PID:{process_id}")
    
    # Load matrix from NPY
    start_load = time.time()
    matrix = np.load(npy_path)
    load_time = time.time() - start_load
    logger.info(f"Loaded matrix {matrix.shape} in {load_time:.2f}s")
    
    # Extract bands
    N = matrix.shape[0]
    bandwidth = 0
    for i in range(N):
        for j in range(i+1, N):
            if abs(matrix[i,j]) > 1e-10:
                bandwidth = max(bandwidth, j-i)
    
    bands = np.zeros((bandwidth+1, N))
    for i in range(bandwidth+1):
        bands[i,:N-i] = np.diag(matrix, -i)
    
    logger.info(f"Extracted bands with bandwidth {bandwidth}")
    
    # Compute eigendecomposition
    from scipy.linalg import eig_banded
    start_eigen = time.time()
    vals, vecs = eig_banded(bands, lower=True, select='a')
    eigen_time = time.time() - start_eigen
    logger.info(f"Eigendecomposition complete in {eigen_time:.2f}s")
    
    # Sort eigenvalues
    idx = np.argsort(vals)
    return vals[idx], vecs[:, idx]

def process_range(args: Tuple[str, str, Tuple[int, int], int]) -> bool:
    """Process a single range: extract, compute, save"""
    matrix_path, output_dir, (start_node, end_node), process_id = args
    logger = logging.getLogger(f"PID:{process_id}")
    
    try:
        # Extract matrix using Rust
        output_path = run_rust_extractor(
            matrix_path, output_dir, start_node, end_node, process_id
        )
        
        # Find the laplacian.npy file
        laplacian_npy = os.path.join(output_path, "laplacian.npy")
        if not os.path.exists(laplacian_npy):
            raise FileNotFoundError(f"No laplacian.npy found in {output_path}")
        
        # Compute eigendecomposition
        vals, vecs = compute_eigen(laplacian_npy, process_id)
        
        # Save results in same directory
        np.save(os.path.join(output_path, "eigenvalues.npy"), vals)
        np.save(os.path.join(output_path, "eigenvectors.npy"), vecs)
        
        logger.info(f"Completed range [{start_node}, {end_node}]")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process range [{start_node}, {end_node}]: {str(e)}")
        return False

def parse_range(range_str: str) -> Tuple[int, int]:
    """Parse a range string like '0-1000' into a tuple"""
    try:
        start, end = map(int, range_str.split('-'))
        return (start, end)
    except:
        raise argparse.ArgumentTypeError("Range must be in format 'start-end'")

def read_ranges_file(file_path: str) -> List[Tuple[int, int]]:
    """Read ranges from a simple CSV file"""
    ranges = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                start, end = map(int, row)
                ranges.append((start, end))
            except:
                logging.warning(f"Skipping invalid range in CSV: {row}")
    return ranges

def main():
    parser = argparse.ArgumentParser(description="Parallel eigendecomposition")
    parser.add_argument("--matrix-path", required=True, help="Path to input matrix file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--ranges", nargs='+', type=parse_range, help="Space-separated ranges (e.g. '0-1000 1000-2000')")
    parser.add_argument("--ranges-file", help="CSV file with ranges (two columns: start, end)")
    parser.add_argument("--processes", type=int, default=max(1, mp.cpu_count() - 1),
                      help="Number of parallel processes")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger("main")
    
    try:
        # Get ranges from either CLI or file
        if args.ranges:
            ranges = args.ranges
        elif args.ranges_file:
            ranges = read_ranges_file(args.ranges_file)
        else:
            parser.error("Must specify either --ranges or --ranges-file")
        
        if not ranges:
            raise ValueError("No valid ranges provided")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Process ranges in parallel
        logger.info(f"Starting parallel processing with {args.processes} processes")
        logger.info(f"Processing {len(ranges)} ranges")
        
        with ProcessPoolExecutor(max_workers=args.processes) as executor:
            # Prepare arguments for each range
            process_args = [
                (args.matrix_path, args.output_dir, range_tuple, i)
                for i, range_tuple in enumerate(ranges)
            ]
            
            # Submit all tasks
            futures = [executor.submit(process_range, arg) for arg in process_args]
            
            # Process results as they complete
            completed = 0
            for future in as_completed(futures):
                if future.result():
                    completed += 1
                logger.info(f"Progress: {completed}/{len(ranges)} ranges completed")
        
        logger.info("Processing complete!")
        
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
