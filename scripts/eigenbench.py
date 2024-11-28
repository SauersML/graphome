import subprocess
import numpy as np
import os
import time
from pathlib import Path
import random
from typing import List, Tuple, Dict
import argparse
from dataclasses import dataclass
import logging
from colorama import init, Fore, Style
import scipy.linalg
import scipy.sparse.linalg
from sklearn.utils.extmath import randomized_svd
import warnings
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""
    matrix_path: str
    output_dir: str
    sizes: List[int]
    samples_per_size: int
    max_node_id: int = 110_884_673  # From the Rust code
    overlap_ratio: float = 0.4
    rust_binary: str = "./target/release/graphome"

class EigenSolver:
    """Base class for eigensolvers"""
    def __init__(self, name: str):
        self.name = name
    
    def solve(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

class DenseEigenSolver(EigenSolver):
    """Numpy/Scipy dense eigensolvers"""
    def __init__(self, method, **kwargs):
        super().__init__(f"dense_{method.__name__}")
        self.method = method
        self.kwargs = kwargs
    
    def solve(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.method(matrix, **self.kwargs)

class SparseEigenSolver(EigenSolver):
    """Sparse matrix eigensolvers"""
    def __init__(self, name: str, method, **kwargs):
        super().__init__(f"sparse_{name}")
        self.method = method
        self.kwargs = kwargs
    
    def solve(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from scipy.sparse import csr_matrix
        sparse_matrix = csr_matrix(matrix)
        return self.method(sparse_matrix, **self.kwargs)

class RandomizedSolver(EigenSolver):
    """Randomized SVD-based solvers"""
    def __init__(self, n_components=None):
        super().__init__("randomized_svd")
        self.n_components = n_components
    
    def solve(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_components = self.n_components or matrix.shape[0] // 10
        U, S, Vt = randomized_svd(matrix, n_components=n_components)
        return S, U

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
        
        # Sparse solvers
        SparseEigenSolver(
            "eigsh_smallest",
            scipy.sparse.linalg.eigsh,
            k=6,
            which='SM'
        ),
        SparseEigenSolver(
            "eigsh_largest",
            scipy.sparse.linalg.eigsh,
            k=6,
            which='LM'
        ),
        
        # Randomized methods
        RandomizedSolver(),
    ]
    return solvers

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

def benchmark_solver(
    solver: EigenSolver,
    matrix: np.ndarray,
    warmup: bool = True
) -> Dict:
    """Benchmark a single solver"""
    if warmup:
        try:
            solver.solve(matrix.copy())
        except Exception as e:
            logger.warning(f"Warmup failed for {solver.name}: {e}")
            return None
    
    times = []
    for _ in range(3):  # 3 trials per solver
        try:
            start = time.perf_counter()
            vals, vecs = solver.solve(matrix.copy())
            end = time.perf_counter()
            times.append(end - start)
        except Exception as e:
            logger.warning(f"Trial failed for {solver.name}: {e}")
            return None
    
    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "eigenvalues": vals,
        "eigenvectors": vecs
    }

def verify_results(results: Dict) -> bool:
    """Verify eigendecomposition results across solvers"""
    reference = None
    for solver_name, result in results.items():
        if result is None:
            continue
        if reference is None:
            reference = (solver_name, result)
            continue
        
        ref_vals = np.sort(np.abs(reference[1]["eigenvalues"]))[:6]
        curr_vals = np.sort(np.abs(result["eigenvalues"]))[:6]
        
        if not np.allclose(ref_vals, curr_vals, rtol=1e-5):
            logger.warning(
                f"Eigenvalues mismatch between {reference[0]} and {solver_name}"
            )
            return False
    return True

def run_benchmarks(config: BenchmarkConfig):
    """Main benchmark orchestration"""
    results_data = []
    solvers = get_solvers()
    
    for size in config.sizes:
        logger.info(f"Benchmarking matrices of size {size}x{size}")
        
        for sample in range(config.samples_per_size):
            # Random starting point
            start_node = random.randint(0, config.max_node_id - size)
            
            # Extract window using Rust
            output_path = run_rust_extractor(config, start_node, size)
            
            # Find the Laplacian matrix
            npy_files = list(Path(output_path).glob("*.npy"))
            if not npy_files:
                logger.error(f"No NPY files found in {output_path}")
                continue
                
            matrix = load_laplacian(str(npy_files[0]))
            
            # Benchmark each solver
            solver_results = {}
            for solver in tqdm(solvers, desc="Testing solvers"):
                result = benchmark_solver(solver, matrix)
                solver_results[solver.name] = result
                
                if result:
                    results_data.append({
                        "size": size,
                        "sample": sample,
                        "solver": solver.name,
                        "time": result["mean_time"],
                        "std": result["std_time"]
                    })
            
            # Verify results
            if not verify_results(solver_results):
                logger.warning(
                    f"Result verification failed for size={size}, sample={sample}"
                )
    
    return pd.DataFrame(results_data)

def plot_results(df: pd.DataFrame, output_dir: str):
    """Generate plots from benchmark results"""
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Time vs Matrix Size for each solver
    plt.figure(figsize=(12, 8))
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
    plt.savefig(os.path.join(plots_dir, "performance_vs_size.png"))
    plt.close()
    
    # Plot 2: Box plot of solver performance
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x="solver", y="time", hue="size")
    plt.xticks(rotation=45)
    plt.title("Performance Distribution by Solver")
    plt.xlabel("Solver")
    plt.ylabel("Time (seconds)")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "solver_distributions.png"))
    plt.close()

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
