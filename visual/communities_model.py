# not complete

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from time import time

def generate_block_circulant_graph(n, k, block_size, perturbation_prob=0.1):
    """
    Generates a graph with distinct diagonal, reverse diagonal, and block structures.
    Each block is circulant-ish, and some edges are perturbed randomly.
    
    n: Number of nodes
    k: Number of nearest neighbors to connect to on each side within blocks
    block_size: Size of each block
    perturbation_prob: Probability of random edge addition/removal
    """
    G = nx.Graph()

    # Add nodes
    G.add_nodes_from(range(n))
    
    # Create circulant structure within each block
    for b in range(0, n, block_size):
        for i in range(b, b + block_size):
            for j in range(1, k + 1):
                G.add_edge(i, (i + j) % block_size + b)  # Forward neighbor
                G.add_edge(i, (i - j) % block_size + b)  # Backward neighbor

    # Create reverse diagonal connections between blocks
    for b in range(0, n - block_size, block_size):
        for i in range(block_size):
            G.add_edge(b + i, b + block_size - i - 1)

    # Randomly perturb some edges
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < perturbation_prob:
                if G.has_edge(i, j):
                    G.remove_edge(i, j)  # Randomly remove edge
                else:
                    G.add_edge(i, j)  # Randomly add edge

    return G

def compute_laplacian_eigenvectors(G):
    """
    Computes the Laplacian of the graph and returns its eigenvalues and eigenvectors.
    """
    L = nx.laplacian_matrix(G).toarray()  # Convert to dense matrix
    eigenvalues, eigenvectors = np.linalg.eigh(L)  # Compute eigenvalues and eigenvectors
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
    """
    non_zero_values = data[data != 0]
    if non_zero_values.size == 0:
        return None  # Indicates all data is zero
    mean = np.mean(non_zero_values)
    std = np.std(non_zero_values)
    intensity = (data - (mean - 2 * std)) / (4 * std)
    intensity = np.clip(intensity, 0, 1)
    return intensity

def create_2d_plot(eigenvectors, method_name, filename):
    """
    Creates and saves a 2D scatter plot of eigenvectors.
    """
    plot_start_time = time()
    print(f"=== Creating 2D Visualization for Method: {method_name} ===")

    # Z-normalize eigenvectors
    intensity = compute_z_normalization(eigenvectors)
    if intensity is None:
        print("All eigenvector values are zero. Skipping plot creation.\n")
        return

    cmap = mcolors.LinearSegmentedColormap.from_list('bwr_custom', ['blue', 'black', 'red'], N=256)
    
    plt.figure(figsize=(20, 18), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')
    
    masked_intensity = np.ma.masked_where(eigenvectors == 0, intensity)
    
    img = ax.imshow(masked_intensity, cmap=cmap, aspect='auto', interpolation='nearest')
    
    plt.title(f'Eigenvector 2D Visualization ({method_name})', color='white', fontsize=24)
    plt.xlabel('Eigenvector Index', color='white', fontsize=20)
    plt.ylabel('Component Index', color='white', fontsize=20)
    
    cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Eigenvector Value', color='white', fontsize=18)
    
    ax.tick_params(axis='x', colors='white', labelsize=14)
    ax.tick_params(axis='y', colors='white', labelsize=14)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, facecolor='black')
    plt.close()
    print(f"Plot saved successfully as '{filename}'.")

def main():
    start_time = time()
    print("=== Laplacian Eigenvector Visualization Script ===\n")

    # Parameters for block circulant graph
    n = 1001  # Number of nodes
    k = 2     # Number of nearest neighbors on each side within blocks
    block_size = 50  # Size of each block
    perturbation_prob = 0.05  # Probability of perturbing edges
    
    # Generate block circulant graph
    G = generate_block_circulant_graph(n, k, block_size, perturbation_prob)
    print(f"Generated block circulant graph with {n} nodes.\n")
    
    # Compute Laplacian and its eigenvectors
    eigenvalues, eigenvectors = compute_laplacian_eigenvectors(G)
    print("Computed Laplacian matrix and its eigenvalues/eigenvectors.\n")
    
    # Sort eigenvectors by eigenvalues
    eigenvalues_sorted, eigenvectors_sorted = sort_data(eigenvalues, eigenvectors)
    
    # Save plot of sorted eigenvectors
    create_2d_plot(eigenvectors_sorted, "Block Circulant Graph Eigenvectors", "block_circulant_eigenvectors_2d.png")

    total_time = time() - start_time
    print(f"Visualization generated successfully in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()
