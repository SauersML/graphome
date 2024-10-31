import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import glob
import imageio.v2 as imageio
from time import time
import argparse

def load_matrix(filepath):
    """Load matrix from CSV, coercing to square by cleaning and dropping as needed."""
    print(f"\nLoading matrix from {filepath}...")
    try:
        # Read the CSV
        df = pd.read_csv(filepath)
        
        # Remove any completely empty rows and columns
        df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        
        # Get numeric data only, starting from row 1, col 1
        matrix = df.iloc[1:, 1:].apply(pd.to_numeric, errors='coerce')
        
        # Drop any rows/cols with non-numeric data
        matrix = matrix.dropna(how='any', axis=0).dropna(how='any', axis=1)
        
        print(f"Initial matrix shape: {matrix.shape}")
        
        # Make square by taking the minimum dimension
        min_dim = min(matrix.shape[0], matrix.shape[1])
        if matrix.shape[0] > min_dim:
            dropped_rows = matrix.shape[0] - min_dim
            print(f"Dropping {dropped_rows} rows to make matrix square")
            matrix = matrix.iloc[:min_dim, :]
        if matrix.shape[1] > min_dim:
            dropped_cols = matrix.shape[1] - min_dim
            print(f"Dropping {dropped_cols} columns to make matrix square")
            matrix = matrix.iloc[:, :min_dim]
            
        # Convert to numpy array
        matrix = matrix.astype(float).values
        print(f"Final square matrix shape: {matrix.shape}")
        
        return matrix
    except Exception as e:
        print(f"Error loading matrix: {e}")
        sys.exit(1)

def compute_laplacian(matrix):
    """Compute the Laplacian matrix."""
    print("\nComputing Laplacian matrix...")
    D = np.diag(np.sum(matrix, axis=1))
    L = D - matrix
    print(f"Laplacian matrix computed with shape {L.shape}")
    return L

def compute_weighted_importance(eigenvectors, eigenvalues):
    """
    Compute weighted importance scores for each component.
    Each component's score is the sum of its eigenvector elements weighted by eigenvalues.
    """
    # Normalize eigenvalues to [0, 1] range for weighting
    normalized_eigenvalues = (eigenvalues - eigenvalues.min()) / (eigenvalues.max() - eigenvalues.min())
    
    # Compute weighted sum for each component
    weighted_sums = np.zeros(eigenvectors.shape[0])
    for i in range(len(eigenvalues)):
        weighted_sums += np.abs(eigenvectors[:, i]) * normalized_eigenvalues[i]
    
    return weighted_sums

def eigendecompose_and_sort(matrix):
    """Perform eigendecomposition and sort by weighted importance."""
    print("\nPerforming eigendecomposition...")
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        print(f"Eigendecomposition complete: {len(eigenvalues)} eigenvalues found")

        # Compute weighted importance scores
        weighted_scores = compute_weighted_importance(eigenvectors, eigenvalues)
        
        # Get sorting indices based on weighted importance
        sorted_indices = np.argsort(weighted_scores)[::-1]  # Sort in descending order
        
        # Apply the sorted indices to the eigenvectors
        eigenvectors = eigenvectors[sorted_indices]
        
        return eigenvalues, eigenvectors, sorted_indices
    except Exception as e:
        print(f"Error during eigendecomposition: {e}")
        sys.exit(1)

def create_output_dirs(base_name):
    """Create output directories for visualizations."""
    dirs = {
        '2d': f'output_{base_name}_2d',
        '3d': f'output_{base_name}_3d',
        'frames': f'output_{base_name}_frames'
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return dirs

def normalize_data(data):
    """Normalize non-zero data for visualization."""
    non_zero = data[data != 0]
    if non_zero.size == 0:
        return None, None, None
    mean = np.mean(non_zero)
    std = np.std(non_zero)
    intensity = (data - (mean - 2 * std)) / (4 * std)
    return np.clip(intensity, 0, 1), mean, std

def plot_2d(eigenvalues, eigenvectors, output_dir, suffix=''):
    """Create 2D visualization."""
    print(f"\nCreating 2D visualization{suffix}...")
    start_time = time()
    
    intensity, mean, std = normalize_data(eigenvectors)
    if intensity is None:
        print("Skipping 2D plot: all zero values")
        return
    
    plt.figure(figsize=(12, 10), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')
    
    cmap = mcolors.LinearSegmentedColormap.from_list('bwr_custom', ['blue', 'black', 'red'], N=256)
    img = ax.imshow(intensity, cmap=cmap, aspect='auto', interpolation='nearest')
    
    # Configure axes and labels
    ax.invert_yaxis()
    plt.title(f'Eigenvector Visualization (Weighted Sort){suffix}', color='white', fontsize=16)
    plt.xlabel('Eigenvector Index', color='white', fontsize=14)
    plt.ylabel('Component Index (Sorted by Weighted Importance)', color='white', fontsize=14)
    
    # Configure colorbar
    cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Eigenvector Value', color='white', fontsize=14)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    # Style adjustments
    ax.tick_params(axis='both', colors='white', labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f'eigenplot_2d{suffix}.png')
    plt.savefig(filename, dpi=300, facecolor='black')
    plt.close()
    
    print(f"2D visualization completed in {time() - start_time:.2f} seconds")

def plot_3d(eigenvalues, eigenvectors, output_dir, suffix=''):
    """Create 3D visualization with multiple views."""
    print(f"\nCreating 3D visualization{suffix}...")
    start_time = time()
    
    # Prepare data
    num_components, num_eigenvectors = eigenvectors.shape
    X, Y = np.meshgrid(np.arange(num_components), np.arange(num_eigenvectors))
    Z = np.outer(np.ones(num_components), eigenvalues)
    colors, mean, std = normalize_data(eigenvectors)
    
    if colors is None:
        print("Skipping 3D plot: all zero values")
        return
    
    # Define viewing angles
    views = [(30, angle) for angle in range(0, 360, 45)] + [(45, angle) for angle in range(45, 360, 90)]
    
    for i, (elev, azim) in enumerate(views):
        fig = plt.figure(figsize=(16, 12), facecolor='black')
        ax = fig.add_subplot(111, projection='3d', facecolor='black')
        
        scatter = ax.scatter(X.flatten(), Y.flatten(), Z.flatten(),
                           c=colors.flatten(),
                           cmap='bwr',
                           s=20)
        
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'3D Eigenvector Visualization (Weighted Sort){suffix}', color='white', fontsize=20)
        
        # Style the plot
        ax.set_facecolor('black')
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')
        
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.set_facecolor((0, 0, 0, 1.0))
            axis.label.set_color('white')
        
        filename = os.path.join(output_dir, f'eigenplot_3d{suffix}_view_{i}.png')
        plt.savefig(filename, dpi=300, facecolor='black')
        plt.close()
    
    print(f"3D visualization completed in {time() - start_time:.2f} seconds")

def create_animation_frames(eigenvalues, eigenvectors, output_dir, suffix=''):
    """Create animation frames."""
    print(f"\nCreating animation frames{suffix}...")
    start_time = time()
    
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    intensity, mean, std = normalize_data(eigenvectors)
    if intensity is None:
        print("Skipping animation: all zero values")
        return
    
    for i in range(eigenvectors.shape[1]):
        plt.figure(figsize=(12, 10), facecolor='black')
        ax = plt.gca()
        ax.set_facecolor('black')
        
        current_data = intensity[:, :i+1] if i > 0 else intensity[:, :1]
        img = ax.imshow(current_data, cmap='bwr', aspect='auto')
        
        plt.title(f'Eigenvector Animation Frame {i+1} (Weighted Sort){suffix}', color='white')
        plt.savefig(os.path.join(frames_dir, f'frame_{i:04d}{suffix}.png'),
                   dpi=200, facecolor='black')
        plt.close()
    
    # Create animation
    frames = []
    frame_files = sorted(glob.glob(os.path.join(frames_dir, f'frame_*{suffix}.png')))
    for frame_file in frame_files:
        frames.append(imageio.imread(frame_file))
    
    output_file = os.path.join(output_dir, f'animation{suffix}.mp4')
    imageio.mimsave(output_file, frames, fps=10)
    
    print(f"Animation created in {time() - start_time:.2f} seconds")

def process_matrix(matrix, base_name, is_laplacian=False):
    """Process matrix and create all visualizations."""
    suffix = '_laplacian' if is_laplacian else ''
    print(f"\nProcessing {'Laplacian' if is_laplacian else 'original'} matrix...")
    
    # Create output directories
    output_dirs = create_output_dirs(f"{base_name}{suffix}")
    
    # Perform eigendecomposition and sort by weighted importance
    eigenvalues, eigenvectors, sorted_indices = eigendecompose_and_sort(matrix)
    
    # Create visualizations
    plot_2d(eigenvalues, eigenvectors, output_dirs['2d'], suffix)
    plot_3d(eigenvalues, eigenvectors, output_dirs['3d'], suffix)
    create_animation_frames(eigenvalues, eigenvectors, output_dirs['frames'], suffix)

def main():
    parser = argparse.ArgumentParser(description="Matrix Analysis and Visualization")
    parser.add_argument('filepath', help='Path to input CSV file')
    args = parser.parse_args()
    
    start_time = time()
    print("=== Matrix Analysis and Visualization Script ===")
    
    # Load and process original matrix
    matrix = load_matrix(args.filepath)
    base_name = os.path.splitext(os.path.basename(args.filepath))[0]
    process_matrix(matrix, base_name)
    
    # Process Laplacian
    laplacian = compute_laplacian(matrix)
    process_matrix(laplacian, base_name, is_laplacian=True)
    
    print(f"\nTotal processing time: {time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
