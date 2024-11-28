import subprocess
import numpy as np
import os
import time
from pathlib import Path
import random
from typing import List, Tuple, Dict, Optional, Set
import argparse
from dataclasses import dataclass
import logging
import logging.handlers
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
from concurrent.futures import ProcessPoolExecutor
import threading
from queue import Queue
import tempfile
import concurrent.futures
import signal
import sys
import psutil

# Configure logging with both file and console handlers
def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.log")
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - PID:%(process)d - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler with detailed formatting
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(detailed_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler with simpler formatting
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_path

# Process monitoring class
class ProcessMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None
        self.start_time = time.time()
    
    def start(self):
        def monitor():
            while not self._stop_event.is_set():
                process = psutil.Process()
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                logging.debug(
                    f"Monitor - Time: {time.time() - self.start_time:.1f}s, "
                    f"CPU: {cpu_percent}%, "
                    f"Memory: {memory_info.rss / 1024 / 1024:.1f}MB"
                )
                time.sleep(self.interval)
        
        self._thread = threading.Thread(target=monitor, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()

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
    n_processes: int = min(mp.cpu_count() // 2, 8)  # More conservative process count
    batch_size: int = 5  # Smaller batch size for better monitoring
    timeout: int = 3600  # 1 hour timeout per solver

class EigenSolver:
    """Base class for eigensolvers"""
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"solver.{name}")
    
    def solve(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
    
    def check_symmetry(self, matrix: np.ndarray) -> bool:
        return np.allclose(matrix, matrix.T, rtol=1e-10)

class DenseEigenSolver(EigenSolver):
    """Dense eigensolvers with detailed logging"""
    def __init__(self, method, **kwargs):
        super().__init__(f"dense_{method.__name__}")
        self.method = method
        self.kwargs = kwargs
    
    def solve(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pid = os.getpid()
        self.logger.debug(f"[PID:{pid}] Starting dense solve for {matrix.shape} matrix")
        
        try:
            if not self.check_symmetry(matrix):
                raise ValueError("Matrix must be symmetric")
            self.logger.debug(f"[PID:{pid}] Symmetry check passed")
            
            # Log memory usage before solve
            mem = psutil.Process().memory_info()
            self.logger.debug(f"[PID:{pid}] Memory before solve: RSS={mem.rss/1024**2:.1f}MB")
            
            # Time the computation
            start = time.perf_counter()
            vals, vecs = self.method(matrix, **self.kwargs)
            elapsed = time.perf_counter() - start
            
            # Log results
            self.logger.debug(
                f"[PID:{pid}] Solve completed in {elapsed:.2f}s. "
                f"Found {len(vals)} eigenvalues"
            )
            
            return vals, vecs
            
        except np.linalg.LinAlgError as e:
            self.logger.error(f"[PID:{pid}] Linear algebra error: {str(e)}")
            raise
        except MemoryError as e:
            self.logger.error(f"[PID:{pid}] Memory error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"[PID:{pid}] Unexpected error: {str(e)}")
            raise

class SparseEigenSolver(EigenSolver):
    """Sparse eigensolvers with detailed logging"""
    def __init__(self, method="eigsh", **kwargs):
        super().__init__(f"sparse_{method}")
        self.method = method
        self.kwargs = kwargs
    
    def to_sparse(self, matrix: np.ndarray):
        """Convert dense matrix to sparse format with logging"""
        pid = os.getpid()
        self.logger.debug(f"[PID:{pid}] Converting {matrix.shape} matrix to sparse format")
        
        from scipy.sparse import csr_matrix
        sparse_mat = csr_matrix(matrix)
        
        # Log sparsity information
        nnz = sparse_mat.nnz
        total = matrix.shape[0] * matrix.shape[1]
        sparsity = 100 * (1 - nnz/total)
        self.logger.debug(f"[PID:{pid}] Sparsity: {sparsity:.1f}% ({nnz} non-zeros)")
        
        return sparse_mat
    
    def solve(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pid = os.getpid()
        self.logger.debug(f"[PID:{pid}] Starting sparse solve for {matrix.shape} matrix")
        
        try:
            if not self.check_symmetry(matrix):
                raise ValueError("Matrix must be symmetric")
            
            sparse_matrix = self.to_sparse(matrix)
            
            if self.method == "eigsh":
                from scipy.sparse.linalg import eigsh
                start = time.perf_counter()
                
                n = matrix.shape[0] - 1
                self.logger.debug(f"[PID:{pid}] Computing {n} eigenvalues")
                
                vals, vecs = eigsh(sparse_matrix, k=n, which='LM')
                elapsed = time.perf_counter() - start
                
                self.logger.debug(f"[PID:{pid}] Sparse solve completed in {elapsed:.2f}s")
                
                idx = np.argsort(vals)
                return vals[idx], vecs[:, idx]
                
        except Exception as e:
            self.logger.error(f"[PID:{pid}] Error in sparse solve: {str(e)}")
            raise

class IterativeEigenSolver(EigenSolver):
    """Iterative eigensolvers with detailed logging"""
    def __init__(self, method="lobpcg", **kwargs):
        super().__init__(f"iterative_{method}")
        self.method = method
        self.kwargs = kwargs
    
    def to_sparse(self, matrix: np.ndarray):
        from scipy.sparse import csr_matrix
        pid = os.getpid()
        self.logger.debug(f"[PID:{pid}] Converting to sparse format")
        return csr_matrix(matrix)
    
    def solve(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pid = os.getpid()
        self.logger.debug(f"[PID:{pid}] Starting iterative solve for {matrix.shape} matrix")
        
        try:
            if not self.check_symmetry(matrix):
                raise ValueError("Matrix must be symmetric")
            
            sparse_matrix = self.to_sparse(matrix)
            
            if self.method == "lobpcg":
                from scipy.sparse.linalg import lobpcg
                n = matrix.shape[0]
                
                self.logger.debug(f"[PID:{pid}] Preparing initial guess")
                X = np.eye(n)
                
                start = time.perf_counter()
                vals, vecs = lobpcg(sparse_matrix, X, largest=False)
                elapsed = time.perf_counter() - start
                
                self.logger.debug(f"[PID:{pid}] LOBPCG completed in {elapsed:.2f}s")
                
                idx = np.argsort(vals)
                return vals[idx], vecs[:, idx]
            
        except Exception as e:
            self.logger.error(f"[PID:{pid}] Error in iterative solve: {str(e)}")
            raise

class BandedEigenSolver(EigenSolver):
    """Banded matrix eigensolvers with detailed logging"""
    def __init__(self, method, **kwargs):
        super().__init__(f"banded_{method.__name__}")
        self.method = method
        self.kwargs = kwargs
    
    def extract_bands(self, matrix: np.ndarray) -> Tuple[np.ndarray, int]:
        pid = os.getpid()
        self.logger.debug(f"[PID:{pid}] Extracting bands from {matrix.shape} matrix")
        
        N = matrix.shape[0]
        bandwidth = 0
        for i in range(N):
            for j in range(i+1, N):
                if abs(matrix[i,j]) > 1e-10:
                    bandwidth = max(bandwidth, j-i)
        
        self.logger.debug(f"[PID:{pid}] Found bandwidth: {bandwidth}")
        
        bands = np.zeros((bandwidth+1, N))
        for i in range(bandwidth+1):
            bands[i,:N-i] = np.diag(matrix, -i)
        
        return bands, bandwidth
    
    def solve(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pid = os.getpid()
        self.logger.debug(f"[PID:{pid}] Starting banded solve for {matrix.shape} matrix")
        
        try:
            if not self.check_symmetry(matrix):
                raise ValueError("Matrix must be symmetric")
            
            bands, bandwidth = self.extract_bands(matrix)
            
            start = time.perf_counter()
            vals, vecs = self.method(bands, **self.kwargs)
            elapsed = time.perf_counter() - start
            
            self.logger.debug(f"[PID:{pid}] Banded solve completed in {elapsed:.2f}s")
            
            return vals, vecs
            
        except Exception as e:
            self.logger.error(f"[PID:{pid}] Error in banded solve: {str(e)}")
            raise


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
    logger = logging.getLogger("extractor")
    logger.debug(f"Extracting window: start={start_node}, size={size}, overlap={overlap}")
    
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
    logger = logging.getLogger("loader")
    logger.debug(f"Loading matrix from {npy_path}")
    return np.load(npy_path)



def run_single_benchmark(solver: EigenSolver, matrix: np.ndarray, 
                        temp_dir: str, size: int) -> Optional[Dict]:
    """Run a single benchmark for one solver on one matrix"""
    pid = os.getpid()
    logger = logging.getLogger(f"benchmark.{solver.name}.{pid}")
    
    try:
        logger.debug(f"Starting benchmark for {solver.name} on {size}x{size} matrix")
        
        # Check available memory
        mem = psutil.virtual_memory()
        logger.debug(f"Available memory: {mem.available / 1024**3:.1f}GB")
        
        # Warmup run with timeout
        logger.debug("Starting warmup run")
        warmup_start = time.perf_counter()
        solver.solve(matrix.copy())
        warmup_time = time.perf_counter() - warmup_start
        logger.debug(f"Warmup completed in {warmup_time:.2f}s")
        
        # Actual benchmark runs
        times = []
        for run_idx in range(3):
            logger.debug(f"Starting run {run_idx + 1}/3")
            start = time.perf_counter()
            vals, vecs = solver.solve(matrix.copy())
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            logger.debug(f"Run {run_idx + 1} completed in {elapsed:.2f}s")
        
        result = {
            "solver": solver.name,
            "size": size,
            "time": np.mean(times),
            "std": np.std(times),
            "eigenvalues": vals[:6] if vals is not None else None,
            "timestamp": datetime.now().isoformat(),
            "process_id": pid
        }
        
        # Save individual result
        result_path = os.path.join(temp_dir, f"{solver.name}_{size}_{time.time()}.json")
        with open(result_path, 'w') as f:
            json.dump(result, f)
        
        logger.debug(f"Benchmark complete, result saved to {result_path}")
        return result
        
    except Exception as e:
        logger.error(f"Error in benchmark: {str(e)}", exc_info=True)
        return None

def run_benchmarks(config: BenchmarkConfig):
    """Main benchmark orchestration with improved parallelism and monitoring"""
    logger = logging.getLogger("main")
    
    # Start process monitor
    monitor = ProcessMonitor()
    monitor.start()
    
    try:
        # Create solvers
        solvers = [
            BandedEigenSolver(scipy.linalg.eig_banded, lower=True),
            SparseEigenSolver("eigsh"),
            IterativeEigenSolver("lobpcg")
        ]
        logger.info(f"Initialized {len(solvers)} solvers")
        
        # Create temporary directory for results
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Created temporary directory: {temp_dir}")
            
            all_results = []
            
            # Process each matrix size sequentially
            for size in config.sizes:
                logger.info(f"\nStarting benchmarks for size {size}x{size}")
                
                # Generate matrices for this size
                matrices = []
                for i in range(config.samples_per_size):
                    start_node = random.randint(0, config.max_node_id - size)
                    try:
                        output_path = run_rust_extractor(config, start_node, size)
                        npy_files = list(Path(output_path).glob("*.npy"))
                        if npy_files:
                            matrix = load_laplacian(str(npy_files[0]))
                            matrices.append(matrix)
                            logger.debug(f"Generated matrix {i+1}/{config.samples_per_size}")
                    except Exception as e:
                        logger.error(f"Failed to generate matrix: {e}")
                
                logger.info(f"Generated {len(matrices)} matrices for size {size}")
                
                # Create benchmark tasks
                tasks = [
                    (solver, matrix, temp_dir, size)
                    for solver in solvers
                    for matrix in matrices
                ]
                
                # Run benchmarks with process pool
                size_results = []
                with ProcessPoolExecutor(max_workers=config.n_processes) as executor:
                    future_to_task = {
                        executor.submit(run_single_benchmark, *task): task
                        for task in tasks
                    }
                    
                    for future in concurrent.futures.as_completed(future_to_task):
                        solver, matrix, _, _ = future_to_task[future]
                        try:
                            result = future.result(timeout=config.timeout)
                            if result:
                                size_results.append(result)
                                logger.info(f"Completed {solver.name} benchmark")
                        except concurrent.futures.TimeoutError:
                            logger.error(f"Timeout for {solver.name}")
                        except Exception as e:
                            logger.error(f"Error in {solver.name}: {e}")
                
                logger.info(f"Completed {len(size_results)}/{len(tasks)} benchmarks for size {size}")
                all_results.extend(size_results)
                
                # Verify results for this size
                if size_results:
                    verification_result = verify_results(size_results)
                    logger.info(f"Result verification for size {size}: {'PASSED' if verification_result else 'FAILED'}")
            
            return pd.DataFrame(all_results)
    
    finally:
        monitor.stop()
        logger.info("Benchmark run complete")

def load_results_from_temp(temp_dir: str) -> pd.DataFrame:
    """Load and combine all result files from temp directory"""
    logger = logging.getLogger("results")
    
    results = []
    for file in Path(temp_dir).glob("*.json"):
        try:
            with open(file, 'r') as f:
                result = json.load(f)
                results.append(result)
            logger.debug(f"Loaded results from {file}")
        except Exception as e:
            logger.error(f"Failed to load {file}: {e}")
    
    return pd.DataFrame(results)

def save_results(df: pd.DataFrame, output_dir: str) -> str:
    """Save results to CSV with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"results_{timestamp}.csv")
    
    try:
        df.to_csv(output_path, index=False)
        logging.info(f"Saved results to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Failed to save results: {e}")
        return None

def plot_results(df: pd.DataFrame, output_dir: str):
    """Generate plots from benchmark results with improved error handling"""
    logger = logging.getLogger("plotting")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    try:
        # Time vs Matrix Size plot
        logger.debug("Generating performance vs size plot")
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
        
        # Additional plots remain similar but with added logging
        logger.info("Generated all plots successfully")
        
    except Exception as e:
        logger.error(f"Error generating plots: {e}", exc_info=True)

def main():
    # Set up argument parser
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
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for log files"
    )
    args = parser.parse_args()
    
    # Set up logging
    log_path = setup_logging(args.log_dir)
    logger = logging.getLogger("main")
    logger.info(f"Logs will be written to {log_path}")
    
    # Log system information
    logger.info(f"CPU count: {mp.cpu_count()}")
    logger.info(f"Total memory: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory: {args.output_dir}")
        
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
        
        if results_df is not None and not results_df.empty:
            # Save results
            results_path = save_results(results_df, args.output_dir)
            
            # Generate plots
            plot_results(results_df, args.output_dir)
            
            logger.info("Benchmark suite completed successfully")
        else:
            logger.error("No results generated from benchmarks")
            
    except KeyboardInterrupt:
        logger.warning("Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical error in benchmark suite: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
