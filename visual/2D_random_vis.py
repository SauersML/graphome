import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
from time import time

def generate_random_data(num_eigenvalues, num_components):
    """
    Generates completely random eigenvalues and eigenvectors.
    Eigenvalues and eigenvectors are drawn from a normal distribution.
    """
    eigenvalues = np.random.normal(loc=0.0, scale=1.0, size=num_eigenvalues)
    eigenvectors = np.random.normal(loc=0.0, scale=1.0, size=(num_components, num_eigenvalues))
    return eigenvalues, eigenvectors

def generate_null_data(num_eigenvalues, num_components):
    """
    Generates null (zero) eigenvalues and eigenvectors.
    """
    eigenvalues = np.zeros(num_eigenvalues)
    eigenvectors = np.zeros((num_components, num_eigenvalues))
    return eigenvalues, eigenvectors

def generate_sparse_random_data(num_eigenvalues, num_components, sparsity_level=0.1):
    """
    Generates sparse random eigenvalues and eigenvectors.
    Eigenvalues are drawn from a uniform distribution.
    Eigenvectors have a specified sparsity level (percentage of non-zero elements).
    """
    eigenvalues = np.random.uniform(low=0.0, high=10.0, size=num_eigenvalues)
    eigenvectors = np.zeros((num_components, num_eigenvalues))
    num_nonzeros = int(sparsity_level * num_components * num_eigenvalues)
    
    # Randomly choose indices to be non-zero
    indices = np.unravel_index(
        np.random.choice(num_components * num_eigenvalues, num_nonzeros, replace=False),
        (num_components, num_eigenvalues)
    )
    eigenvectors[indices] = np.random.normal(loc=0.0, scale=1.0, size=num_nonzeros)
    return eigenvalues, eigenvectors

def generate_mixed_data(num_eigenvalues, num_components, sparsity_level=0.1):
    """
    Generates mixed data by combining different data generation strategies.
    For example, half random and half sparse.
    """
    half = num_eigenvalues // 2
    # Generate first half random
    eigenvalues_random, eigenvectors_random = generate_random_data(half, num_components)
    # Generate second half sparse
    eigenvalues_sparse, eigenvectors_sparse = generate_sparse_random_data(num_eigenvalues - half, num_components, sparsity_level)
    # Concatenate the data
    eigenvalues = np.concatenate((eigenvalues_random, eigenvalues_sparse))
    eigenvectors = np.hstack((eigenvectors_random, eigenvectors_sparse))
    return eigenvalues, eigenvectors

def sort_data(eigenvalues, eigenvectors):
    """
    Sorts eigenvalues in ascending order and rearranges eigenvectors accordingly.
    """
    idx = np.argsort(eigenvalues)
    eigenvalues_sorted = eigenvalues[idx]
    eigenvectors_sorted = eigenvectors[:, idx]
    return eigenvalues_sorted, eigenvectors_sorted

def compute_z_normalization(data):
    """
    Computes Z-normalization for the data excluding zeros.
    Maps values <= (mean - 2*std) to 0 and >= (mean + 2*std) to 1.
    """
    non_zero_values = data[data != 0]
    if non_zero_values.size == 0:
        return None  # Indicates all data is zero
    mean = np.mean(non_zero_values)
    std = np.std(non_zero_values)
    intensity = (data - (mean - 2 * std)) / (4 * std)
    intensity = np.clip(intensity, 0, 1)
    return intensity

def create_2d_plot(eigenvalues, eigenvectors, method_name, filename):
    """
    Creates and saves a 2D scatter plot based on eigenvalues and eigenvectors.
    """
    plot_start_time = time()
    print(f"=== Creating 2D Visualization for Method: {method_name} ===")
    print("Starting plot creation...\n")
    
    # Sort the data
    eigenvalues_sorted, eigenvectors_sorted = sort_data(eigenvalues, eigenvectors)
    print("Data sorted based on eigenvalues.\n")
    
    # Prepare grid positions
    num_eigenvectors = eigenvectors_sorted.shape[1]
    num_components = eigenvectors_sorted.shape[0]
    data = eigenvectors_sorted  # Shape: (num_components, num_eigenvectors)
    print(f"Prepared data grid with shape: {data.shape}\n")
    
    # Compute Z-normalization
    intensity = compute_z_normalization(data)
    if intensity is None:
        print("All eigenvector values are zero. Skipping plot creation.\n")
        return
    print("Z-normalization completed.\n")
    
    print(f"Intensity range: min={intensity.min()}, max={intensity.max()}\n")
    
    # Define custom colormap
    cmap = mcolors.LinearSegmentedColormap.from_list('bwr_custom', ['blue', 'black', 'red'], N=256)
    
    # Create the plot
    try:
        plt.figure(figsize=(20, 18), facecolor='black')
        ax = plt.gca()
        ax.set_facecolor('black')
        
        # Mask zero values
        masked_intensity = np.ma.masked_where(data == 0, intensity)
        
        # Plot using imshow
        img = ax.imshow(masked_intensity, cmap=cmap, aspect='auto', interpolation='nearest')
        
        # Set ticks
        num_ticks = 10
        xticks = np.linspace(0, num_eigenvectors - 1, num_ticks)
        yticks = np.linspace(0, num_components - 1, num_ticks)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([int(x) for x in xticks])
        ax.set_yticklabels([int(y) for y in yticks])
        
        # Invert y-axis
        ax.invert_yaxis()
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Set titles and labels
        plt.title(f'Eigenvalue-Eigenvector 2D Visualization ({method_name})', color='white', fontsize=24)
        plt.xlabel('Eigenvector Index', color='white', fontsize=20)
        plt.ylabel('Component Index', color='white', fontsize=20)
        
        # Configure colorbar
        cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Eigenvector Value', color='white', fontsize=18)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Adjust tick colors
        ax.tick_params(axis='x', colors='white', labelsize=14)
        ax.tick_params(axis='y', colors='white', labelsize=14)
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(filename, dpi=300, facecolor='black')
        plt.close()
        print(f"Plot saved successfully as '{filename}'.\n")
    except Exception as e:
        print(f"Error creating scatter plot for method '{method_name}': {e}\n")
        return
    
    plot_end_time = time()
    print(f"Visualization for method '{method_name}' created in {plot_end_time - plot_start_time:.2f} seconds.\n")

def main():
    start_time = time()
    print("=== Eigenvalue-Eigenvector 2D Visualization Script ===\n")
    
    # Define parameters
    num_eigenvalues = 5001
    num_components = 5001
    sparsity_level = 0.1  # 10% non-zero
    
    # Define data generation methods
    data_methods = {
        'Random Data': generate_random_data,
        'Null Data': generate_null_data,
        'Sparse Random Data': lambda n, c: generate_sparse_random_data(n, c, sparsity_level),
        'Mixed Data': lambda n, c: generate_mixed_data(n, c, sparsity_level)
    }
    
    # Iterate through each data generation method
    for method_name, data_gen_function in data_methods.items():
        print(f"--- Processing Method: {method_name} ---\n")
        method_start_time = time()
        
        # Generate data
        eigenvalues, eigenvectors = data_gen_function(num_eigenvalues, num_components)
        print(f"Generated eigenvalues and eigenvectors for method '{method_name}'.\n")
        
        safe_method_name = method_name.lower().replace(' ', '_')
        filename = f"eigenplot_{safe_method_name}_2d.png"
        
        create_2d_plot(eigenvalues, eigenvectors, method_name, filename)
        
        method_end_time = time()
        print(f"Completed Method: {method_name} in {method_end_time - method_start_time:.2f} seconds.\n")
    
    total_time = time() - start_time
    print(f"All visualizations have been generated successfully in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()
