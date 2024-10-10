import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from time import time
import os
import concurrent.futures

def generate_path_community_graph(n, num_communities, inter_community_prob):
    """
    n: Total number of nodes
    num_communities: Number of communities (blocks)
    inter_community_prob: Probability of connecting nodes between different communities
    """
    G = nx.Graph()

    # Calculate size of each community
    community_size = n // num_communities

    # Create a path graph for each community
    for c in range(num_communities):
        start = c * community_size
        end = start + community_size
        path = nx.path_graph(range(start, end))  # Create a path graph for the community
        G.add_edges_from(path.edges())

    # Add inter-community connections
    for i in range(num_communities):
        for j in range(i + 1, num_communities):
            start_i = i * community_size
            start_j = j * community_size
            for a in range(community_size):
                if np.random.rand() < inter_community_prob:
                    G.add_edge(start_i + a, start_j + a)  # Connecting corresponding nodes between communities

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

    # Parameters for the community graph
    n = 1001  # Total number of nodes
    variable_to_modify = 'inter_community_prob'  # Could be 'num_communities' or 'inter_community_prob'
    num_steps = 10  # Number of steps
    steps = np.linspace(0.0, 0.4, num_steps)  # Range of values for inter_community_prob (or num_communities)

    def generate_graph_data(num_communities, inter_community_prob):
        G = generate_path_community_graph(n, num_communities, inter_community_prob)
        eigenvalues, eigenvectors = compute_laplacian_eigenvectors(G)
        eigenvalues_sorted, eigenvectors_sorted = sort_data(eigenvalues, eigenvectors)
        return eigenvectors_sorted, num_communities, inter_community_prob

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        # Modify either num_communities or inter_community_prob
        if variable_to_modify == 'num_communities':
            for step in steps:
                futures.append(executor.submit(generate_graph_data, int(step), 0.2))  # inter_community_prob fixed
        elif variable_to_modify == 'inter_community_prob':
            for step in steps:
                futures.append(executor.submit(generate_graph_data, 3, step))  # num_communities fixed

        # Collect results and plot in the main thread
        for future in concurrent.futures.as_completed(futures):
            eigenvectors_sorted, num_communities, inter_community_prob = future.result()
            filename = f"community_eigenvectors_2d_nc{num_communities}_p{inter_community_prob:.2f}.png"
            create_2d_plot(eigenvectors_sorted, f"nc={num_communities}, p={inter_community_prob:.2f}", filename)
            print(f"Plot saved: {filename}")

    total_time = time() - start_time
    print(f"Visualizations generated successfully in {total_time:.2f} seconds.")


if __name__ == "__main__":
    main()
