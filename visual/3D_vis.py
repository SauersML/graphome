import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
from time import time

def compute_alpha(deviation):
    """
    Compute alpha transparency based on absolute deviation from the mean.
    Closer to the mean -> more transparent (lower alpha).
    Further from the mean -> more opaque (higher alpha).

    Parameters:
    - deviation: numpy array of deviations in SD units, already clipped to [-2, 2].

    Returns:
    - alpha: numpy array of alpha values in [0, 1].
    """
    # Compute absolute deviation
    abs_deviation = np.abs(deviation)
    
    # Normalize absolute deviation to [0, 1] based on the clipping at 2 SDs
    alpha = abs_deviation / 2.0  # Since deviation is clipped at [-2, 2]
    
    return alpha

def assign_colors(data, deviation):
    N = len(data)
    colors = np.zeros((N, 4))  # Initialize RGBA array
    
    # Compute alpha transparency based on absolute deviation
    alpha = compute_alpha(deviation)
    
    # Base color: blue
    colors[:, 0] = 0  # R
    colors[:, 1] = 0  # G
    colors[:, 2] = 1  # B
    colors[:, 3] = alpha  # Alpha based on deviation
    
    # Assign red to positive deviations
    mask_pos = deviation >= 0
    colors[mask_pos, 0] = 1  # R
    colors[mask_pos, 1] = 0  # G
    colors[mask_pos, 2] = 0  # B
    
    # Assign black to zero values
    mask_zero = data == 0
    colors[mask_zero] = [0, 0, 0, 1]  # Black with full opacity
    
    return colors

def create_3d_scatter(eigenvalues, eigenvectors, sorted_data=False, filename="eigenplot_3d.png", view_angles=None):
    plot_start_time = time()
    plot_type = "Sorted" if sorted_data else "Unsorted"
    print(f"=== Creating {plot_type} 3D Visualization ===\n")
    
    if sorted_data:
        idx_sorted = np.argsort(eigenvalues)
        eigenvalues_sorted = eigenvalues[idx_sorted]
        eigenvectors_sorted = eigenvectors[:, idx_sorted]
    else:
        eigenvalues_sorted = eigenvalues
        eigenvectors_sorted = eigenvectors
    
    num_eigenvectors = eigenvectors_sorted.shape[1]
    num_components = eigenvectors_sorted.shape[0]
    X, Y = np.meshgrid(np.arange(num_components), np.arange(num_eigenvectors))
    X = X.flatten()
    Y = Y.flatten()
    Z = eigenvalues_sorted[Y]
    data = eigenvectors_sorted.flatten()
    
    non_zero_data = data[data != 0]
    if non_zero_data.size == 0:
        print("All eigenvector values are zero. Skipping plot creation.\n")
        return
    mean = np.mean(non_zero_data)
    std = np.std(non_zero_data)
    deviation = (data - mean) / std
    deviation_clipped = np.clip(deviation, -2, 2)
    
    colors = assign_colors(data, deviation_clipped)
    
    fig = plt.figure(figsize=(16, 12), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
    scatter = ax.scatter(
        X, Y, Z,
        facecolors=colors,  # Updated here
        marker='s',
        s=20,
        depthshade=False      # Updated here
    )
    
    if view_angles:
        elev, azim = view_angles
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'Eigenvalue-Eigenvector 3D Visualization ({plot_type})', color='white', fontsize=20)
    else:
        ax.set_title(f'Eigenvalue-Eigenvector 3D Visualization ({plot_type})', color='white', fontsize=20)
    
    ax.set_xlabel('Component Index', color='white', fontsize=14)
    ax.set_ylabel('Eigenvector Index', color='white', fontsize=14)
    ax.set_zlabel('Eigenvalue', color='white', fontsize=14)
    
    ax.xaxis.pane.set_facecolor((0, 0, 0, 1.0))
    ax.yaxis.pane.set_facecolor((0, 0, 0, 1.0))
    ax.zaxis.pane.set_facecolor((0, 0, 0, 1.0))
    
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')
    
    cmap = mcolors.LinearSegmentedColormap.from_list('blue_red', ['blue', 'red'], N=256)
    norm = mcolors.Normalize(vmin=np.min(non_zero_data), vmax=np.max(non_zero_data))
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(non_zero_data)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=20, pad=0.1)
    cbar.set_label('Eigenvector Value (SD from Mean)', color='white', fontsize=14)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    fig.patch.set_facecolor('black')
    ax.patch.set_facecolor('black')
    
    plt.tight_layout()
    
    try:
        plt.savefig(filename, dpi=300, facecolor='black')
        plt.close()
        print(f"Plot saved successfully as '{filename}' in {time() - plot_start_time:.4f} seconds.\n")
    except Exception as e:
        print(f"Error saving plot: {e}\n")
        return
    
    print(f"{plot_type} visualization creation completed in {time() - plot_start_time:.4f} seconds.\n")

def create_sorted_plots(eigenvalues, eigenvectors, view_angles, filename_prefix="eigenplot_sorted_3d_view"):
    print("=== Creating Sorted 3D Visualizations with Multiple Views ===")
    for idx, (elev, azim) in enumerate(view_angles, start=1):
        filename = f"{filename_prefix}_{idx}.png"
        print(f"--- Creating view {idx}: Elevation={elev}, Azimuth={azim} ---")
        create_3d_scatter(
            eigenvalues,
            eigenvectors,
            sorted_data=True,
            filename=filename,
            view_angles=(elev, azim)
        )

def main():
    start_time = time()
    print("=== Eigenvalue-Eigenvector 3D Visualization Script ===\n")
    
    required_files = ['submatrix.eigenvalues.csv', 'submatrix.eigenvectors.csv']
    missing_files = [file for file in required_files if not os.path.isfile(file)]
    if missing_files:
        print(f"Error: The following required file(s) are missing: {', '.join(missing_files)}")
        sys.exit(1)
    print("All required files are present.\n")
    
    try:
        eigenvalues = pd.read_csv('submatrix.eigenvalues.csv', header=None).values.flatten()
        eigenvectors = pd.read_csv('submatrix.eigenvectors.csv', header=None).values
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        sys.exit(1)
    print(f"Loaded {len(eigenvalues)} eigenvalues and eigenvectors with shape {eigenvectors.shape}.\n")
    
    if eigenvectors.shape[1] != len(eigenvalues):
        print("Error: The number of eigenvalues does not match the number of eigenvectors.")
        sys.exit(1)
    
    create_3d_scatter(
        eigenvalues,
        eigenvectors,
        sorted_data=False,
        filename="eigenplot_original_3d.png"
    )
    
    view_angles_sorted = [
        (30, 45),
        (30, 90),
        (30, 135),
        (30, 180),
        (30, 225),
        (30, 270),
        (30, 315),
        (45, 45),
        (45, 135),
        (45, 225)
    ]
    
    create_sorted_plots(
        eigenvalues,
        eigenvectors,
        view_angles_sorted,
        filename_prefix="eigenplot_sorted_3d_view"
    )
    
    total_time = time() - start_time
    print(f"All visualizations have been generated successfully in {total_time:.4f} seconds.")

if __name__ == "__main__":
    main()
