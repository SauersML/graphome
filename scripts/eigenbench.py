import subprocess
import numpy as np
import os
import time
from pathlib import Path
import random
from typing import List, Tuple, Dict, Optional
import argparse
from dataclasses import dataclass
import logging
from colorama import init, Fore, Style
import scipy.linalg
import scipy.sparse.linalg
import warnings
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
from queue import Queue
import signal
import atexit
import tempfile

# Initialize colorama
init(autoreset=True)

# Configure logging with thread safety
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global result queue for parallel processing
result_queue = Queue()
save_event = threading.Event()
SAVE_INTERVAL = 60  # Save every 60 seconds

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""
    matrix_path: str
    output_dir: str
    sizes: List[int]
    samples_per_size: int
    max_node_id: int = 110_000_000
    overlap_ratio: float = 0.4
    rust_binary: str = "../target/release/graphome"
    n_processes: int = mp.cpu_count()
    batch_size: int = 10

class EigenSolver:
    """Base class for eigensolvers"""
    def __init__(self, name: str):
        self.name = name
    
    def solve(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
    
    def check_symmetry(self, matrix: np.ndarray) -> bool:
        return np.allclose(matrix, matrix.T, rtol=1e-10)

class DenseEigenSolver(EigenSolver):
    """Numpy/Scipy dense eigensolvers"""
    def __init__(self, method, **kwargs):
        super().__init__(f"dense_{method.__name__}")
        self.method = method
        self.kwargs = kwargs
    
    def solve(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.check_symmetry(matrix):
            raise ValueError("Matrix must be symmetric")
        return self.method(matrix, **self.kwargs)

class SparseEigenSolver(EigenSolver):
    """Sparse matrix eigensolvers using scipy.sparse.linalg"""
    def __init__(self, method="eigsh", **kwargs):
        super().__init__(f"sparse_{method}")
        self.method = method
        self.kwargs = kwargs
    
    def to_sparse(self, matrix: np.ndarray):
        """Convert dense matrix to sparse format"""
        from scipy.sparse import csr_matrix
        return csr_matrix(matrix)
    
    def solve(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.check_symmetry(matrix):
            raise ValueError("Matrix must be symmetric")
        
        sparse_matrix = self.to_sparse(matrix)
        
        if self.method == "eigsh":
            from scipy.sparse.linalg import eigsh
            # Get all eigenvalues/vectors
            n = matrix.shape[0] - 1
            vals, vecs = eigsh(sparse_matrix, k=n, which='LM')
            # Sort them
            idx = np.argsort(vals)
            return vals[idx], vecs[:, idx]
        else:
            raise ValueError(f"Unknown sparse method: {self.method}")

class IterativeEigenSolver(EigenSolver):
    """Iterative eigensolvers for sparse matrices"""
    def __init__(self, method="lobpcg", **kwargs):
        super().__init__(f"iterative_{method}")
        self.method = method
        self.kwargs = kwargs
    
    def to_sparse(self, matrix: np.ndarray):
        from scipy.sparse import csr_matrix
        return csr_matrix(matrix)
    
    def solve(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.check_symmetry(matrix):
            raise ValueError("Matrix must be symmetric")
            
        sparse_matrix = self.to_sparse(matrix)
        
        if self.method == "lobpcg":
            from scipy.sparse.linalg import lobpcg
            n = matrix.shape[0]
            X = np.eye(n)  # Initial guess for all eigenvectors
            vals, vecs = lobpcg(sparse_matrix, X, largest=False)
            # Sort them
            idx = np.argsort(vals)
            return vals[idx], vecs[:, idx]
        else:
            raise ValueError(f"Unknown iterative method: {self.method}")

class BandedEigenSolver(EigenSolver):
    """Banded matrix eigensolvers"""
    def __init__(self, method, **kwargs):
        super().__init__(f"banded_{method.__name__}")
        self.method = method
        self.kwargs = kwargs
    
    def extract_bands(self, matrix: np.ndarray) -> Tuple[np.ndarray, int]:
        N = matrix.shape[0]
        bandwidth = 0
        for i in range(N):
            for j in range(i+1, N):
                if abs(matrix[i,j]) > 1e-10:
                    bandwidth = max(bandwidth, j-i)
        
        bands = np.zeros((bandwidth+1, N))
        for i in range(bandwidth+1):
            bands[i,:N-i] = np.diag(matrix, -i)
        return bands, bandwidth
    
    def solve(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.check_symmetry(matrix):
            raise ValueError("Matrix must be symmetric")
        bands, _ = self.extract_bands(matrix)
        return self.method(bands, **self.kwargs)

def get_solvers() -> List[EigenSolver]:
    """Configure all eigensolvers to benchmark"""
    solvers = [
        # Dense solvers from numpy
        DenseEigenSolver(np.linalg.eigh),
        DenseEigenSolver(np.linalg.eig),
        
        # Dense solvers from scipy
        DenseEigenSolver(scipy.linalg.eigh, driver='evr'),
        DenseEigenSolver(scipy.linalg.eigh, driver='evx'),
        DenseEigenSolver(scipy.linalg.eigh, subset_by_value=[-np.inf, np.inf]),
        DenseEigenSolver(scipy.linalg.eigh, lower=True, overwrite_a=True),
        
        # Banded solvers
        BandedEigenSolver(scipy.linalg.eig_banded, lower=True),
        
        # Sparse solvers
        SparseEigenSolver("eigsh"),
        IterativeEigenSolver("lobpcg"),
    ]
    return solvers

class ResultSaver(threading.Thread):
    """Background thread for periodic result saving"""
    def __init__(self, output_path: str, temp_dir: str):
        super().__init__()
        self.output_path = output_path
        self.temp_dir = temp_dir
        self.daemon = True
        self._stop_event = threading.Event()
        
    def stop(self):
        self._stop_event.set()
        
    def run(self):
        while not self._stop_event.is_set():
            if save_event.wait(timeout=SAVE_INTERVAL):
                self.save_results()
                save_event.clear()
    
    def save_results(self):
        try:
            temp_files = list(Path(self.temp_dir).glob("*.csv"))
            if not temp_files:
                return
                
            dfs = []
            for f in temp_files:
                try:
                    df = pd.read_csv(f)
                    dfs.append(df)
                    os.unlink(f)
                except Exception as e:
                    logger.error(f"Error reading {f}: {e}")
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                if os.path.exists(self.output_path):
                    existing_df = pd.read_csv(self.output_path)
                    combined_df = pd.concat([existing_df, combined_df], ignore_index=True)
                combined_df.to_csv(self.output_path, index=False)
                logger.info(f"Saved {len(dfs)} result batches")
        except Exception as e:
            logger.error(f"Error in save_results: {e}")

def run_rust_extractor(
    config: BenchmarkConfig,
    start_node: int,
    size: int
) -> str:
    """Run the Rust binary to extract a window"""
    output_subdir = f"window_{start_node}_{size}"
    output_path = os.path.join(config.output_dir, output_subdir)
    os.makedirs(output_path, exist_ok=True)
    
    overlap = int(size * config.overlap_ratio)
    
    cmd = [
        config.rust_binary,
        "extract-windows",
        "--input", config.matrix_path,
        "--start-node", str(start_node),
        "--end-node", str(start_node + size),
        "--window-size", str(size),
        "--overlap", str(overlap),
        "--output", output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running Rust extractor: {e.stderr}")
        raise

def load_laplacian(npy_path: str) -> np.ndarray:
    """Load a Laplacian matrix from NPY file"""
    return np.load(npy_path)

def benchmark_solver_batch(args: Tuple[EigenSolver, List[np.ndarray], str]) -> List[Dict]:
    """Benchmark a batch of matrices for a single solver"""
    solver, matrices, temp_dir = args
    results = []
    
    for matrix in matrices:
        try:
            # Warmup run
            solver.solve(matrix.copy())
            
            times = []
            for _ in range(3):
                start = time.perf_counter()
                vals, vecs = solver.solve(matrix.copy())
                end = time.perf_counter()
                times.append(end - start)
            
            result = {
                "solver": solver.name,
                "size": matrix.shape[0],
                "time": np.mean(times),
                "std": np.std(times),
                "eigenvalues": vals[:6] if vals is not None else None,
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            
        except Exception as e:
            logger.warning(f"Error in {solver.name}: {e}")
            continue
    
    if results:
        # Save batch results to temporary csv file
        df = pd.DataFrame(results)
        temp_file = os.path.join(temp_dir, f"{solver.name}_{time.time()}.csv")
        df.to_csv(temp_file)
        save_event.set()
    
    return results

def verify_results(results: List[Dict]) -> bool:
    """Verify eigendecomposition results across solvers"""
    if not results:
        return True
        
    reference = None
    for result in results:
        if result.get("eigenvalues") is None:
            continue
        if reference is None:
            reference = result
            continue
            
        ref_vals = np.sort(np.abs(reference["eigenvalues"]))
        curr_vals = np.sort(np.abs(result["eigenvalues"]))
        
        if not np.allclose(ref_vals, curr_vals, rtol=1e-5):
            logger.warning(
                f"Eigenvalues mismatch between {reference['solver']} and {result['solver']}"
            )
            return False
    return True

def run_benchmarks(config: BenchmarkConfig):
    """Main benchmark orchestration with parallel processing"""
    solvers = get_solvers()
    
    # Create temporary directory for intermediate results
    temp_dir = tempfile.mkdtemp()
    
    # Start result saver thread
    output_path = os.path.join(config.output_dir, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    saver = ResultSaver(output_path, temp_dir)
    saver.start()
    
    try:
        with ProcessPoolExecutor(max_workers=config.n_processes) as executor:
            for size in config.sizes:
                logger.info(f"Benchmarking matrices of size {size}x{size}")
                
                # Generate matrix batches
                matrix_batches = []
                for _ in range(0, config.samples_per_size, config.batch_size):
                    batch = []
                    for _ in range(config.batch_size):
                        start_node = random.randint(0, config.max_node_id - size)
                        output_path = run_rust_extractor(config, start_node, size)
                        npy_files = list(Path(output_path).glob("*.npy"))
                        if npy_files:
                            matrix = load_laplacian(str(npy_files[0]))
                            batch.append(matrix)
                    if batch:
                        matrix_batches.append(batch)
                
                # Submit solver tasks
                future_to_solver = {
                    executor.submit(benchmark_solver_batch, (solver, batch, temp_dir)): solver
                    for solver in solvers
                    for batch in matrix_batches
                }
                
                # Process results as they complete
                batch_results = []
                for future in tqdm(future_to_solver, desc="Processing batches"):
                    solver = future_to_solver[future]
                    try:
                        results = future.result()
                        batch_results.extend(results)
                    except Exception as e:
                        logger.error(f"Error in {solver.name}: {e}")
                
                # Verify results for this size
                if not verify_results(batch_results):
                    logger.warning(f"Result verification failed for size={size}")
    
    finally:
        # Stop result saver and clean up
        saver.stop()
        saver.join()
        
        # Final save
        saver.save_results()
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_dir)

def plot_results(df: pd.DataFrame, output_dir: str):
    """Generate plots from benchmark results"""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Time vs Matrix Size for each solver
    plt.figure(figsize=(15, 10))
    sns.lineplot(
        data=df,
        x="size",
        y="time",
        hue="solver",
        style="solver",
        markers=True,
        dashes=False
    )
    plt.yscale('log')
    plt.title("Solver Performance vs Matrix Size")
    plt.xlabel("Matrix Size")
    plt.ylabel("Time (seconds)")
    plt.xticks(df['size'].unique())
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "performance_vs_size.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Box plot of solver performance
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=df, x="solver", y="time", hue="size")
    plt.xticks(rotation=45, ha='right')
    plt.title("Performance Distribution by Solver")
    plt.xlabel("Solver")
    plt.ylabel("Time (seconds)")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "solver_distributions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Heatmap of solver performance
    pivot_df = df.pivot_table(
        values='time',
        index='solver',
        columns='size',
        aggfunc='mean'
    )
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt='.2e',
        cmap='viridis',
        cbar_kws={'label': 'Time (seconds)'}
    )
    plt.title("Average Solver Performance by Matrix Size")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "performance_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Performance stability (standard deviation)
    plt.figure(figsize=(15, 10))
    stability_df = df.groupby(['solver', 'size'])['time'].agg(['mean', 'std']).reset_index()
    stability_df['cv'] = stability_df['std'] / stability_df['mean']  # Coefficient of variation
    sns.barplot(data=stability_df, x='solver', y='cv', hue='size')
    plt.xticks(rotation=45, ha='right')
    plt.title("Solver Stability (Lower is Better)")
    plt.xlabel("Solver")
    plt.ylabel("Coefficient of Variation")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "solver_stability.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate summary statistics
    summary_stats = df.groupby('solver').agg({
        'time': ['mean', 'std', 'min', 'max'],
        'size': 'nunique'
    }).round(4)
    
    summary_path = os.path.join(output_dir, "summary_statistics.csv")
    summary_stats.to_csv(summary_path)



def main():
    parser = argparse.ArgumentParser(description="Benchmark eigendecomposition methods")
    parser.add_argument("--matrix-path", required=True, help="Path to input matrix file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 500, 1000],
        help="Matrix sizes to benchmark"
    )
    parser.add_argument(
        "--samples-per-size",
        type=int,
        default=3,
        help="Number of samples per size"
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure benchmark
    config = BenchmarkConfig(
        matrix_path=args.matrix_path,
        output_dir=args.output_dir,
        sizes=args.sizes,
        samples_per_size=args.samples_per_size
    )
    
    # Run benchmarks
    logger.info("Starting benchmark suite...")
    results_df = run_benchmarks(config)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.output_dir, f"results_{timestamp}.csv")
    results_df.to_csv(results_path, index=False)
    
    # Generate plots
    plot_results(results_df, args.output_dir)
    
    logger.info(f"Benchmarks complete. Results saved to {results_path}")
    logger.info(f"Plots saved in {os.path.join(args.output_dir, 'plots')}")

if __name__ == "__main__":
    main()
