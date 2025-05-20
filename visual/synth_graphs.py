import os
import networkx as nx
import numpy as np
import time
import random
import pickle # Added for direct pickling

# Imports from sibling modules.
from communities_model import generate_community_graph
from bipartite_model import generate_approx_bipartite_graph
from path_model import generate_path_community_graph


# --- Helper functions (can be used for later processing) ---
def compute_laplacian_eigenvectors(G):
    """
    Computes the Laplacian of the graph and returns its eigenvalues and eigenvectors.
    """
    if not G or not G.nodes:
        print("Warning: Graph is empty or has no nodes. Cannot compute Laplacian.")
        return np.array([]), np.array([])
    
    # nx.laplacian_matrix handles isolated nodes correctly.
    L = nx.laplacian_matrix(G).toarray()
    if L.shape[0] == 0: # Empty matrix
        return np.array([]), np.array([])
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    return eigenvalues, eigenvectors

def sort_data(eigenvalues, eigenvectors):
    """
    Sorts eigenvalues in ascending order and rearranges eigenvectors accordingly.
    """
    if eigenvalues.size == 0:
        return eigenvalues, eigenvectors
    idx = np.argsort(eigenvalues)
    eigenvalues_sorted = eigenvalues[idx]
    eigenvectors_sorted = eigenvectors[:, idx]
    return eigenvalues_sorted, eigenvectors_sorted

# --- Main Data Generation Logic ---

def generate_and_save_graphs():
    """
    Generates a diverse set of synthetic graphs using the imported models
    and saves them using Python's pickle.
    """
    base_output_dir = "synthetic_graph_outputs_pickle" # Changed suffix
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"Saving generated graphs to: {os.path.abspath(base_output_dir)}")

    graph_counter = 0
    total_start_time = time.time()
    num_graphs_per_model_type = 80 # Number of graphs to generate for each model type

    # --- Parameter Sampling Functions ---
    def sample_n_nodes():
        return random.randint(20, 160) # Wider range for node counts

    def sample_probability(low=0.0, high=0.3): # Wider general probability range
        return random.uniform(low, high)

    # 1. Bipartite Model
    print("\n--- Generating Approximate Bipartite Graphs ---")
    model_name = "bipartite"
    model_output_dir = os.path.join(base_output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    for i in range(num_graphs_per_model_type):
        n_bp_original = sample_n_nodes()
        # Bipartite model's initial setup benefits from an even number of nodes.
        n_bp = n_bp_original if n_bp_original % 2 == 0 else n_bp_original + 1
        
        perturb_p_bp = sample_probability(0.0, 0.25) # Perturbation probability

        G = generate_approx_bipartite_graph(n_bp, perturb_p_bp)
        filename = f"{model_name}_n{n_bp}_perturb{perturb_p_bp:.4f}_id{i:03d}.pkl" # Changed extension
        filepath = os.path.join(model_output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        graph_counter += 1
        if (i + 1) % 10 == 0 or i == num_graphs_per_model_type -1:
            print(f"Generated {model_name} ({i+1}/{num_graphs_per_model_type}): {filename} (N: {G.number_of_nodes()}, E: {G.number_of_edges()})")


    # 2. Path Model
    print("\n--- Generating Path Community Graphs ---")
    model_name = "path_community"
    model_output_dir = os.path.join(base_output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    for i in range(num_graphs_per_model_type):
        n_pc_original = sample_n_nodes()
        
        min_nodes_per_community_path = 1 
        max_c_pc = n_pc_original // min_nodes_per_community_path
        if max_c_pc < 1: max_c_pc = 1 
        num_c_pc = random.randint(1, max_c_pc)
        
        n_pc = (n_pc_original // num_c_pc) * num_c_pc
        if n_pc == 0 : 
            if n_pc_original > 0: 
                num_c_pc = n_pc_original 
                n_pc = n_pc_original
            else: 
                continue 

        inter_p_pc = sample_probability(0.0, 0.35) 

        G = generate_path_community_graph(n_pc, num_c_pc, inter_p_pc)
        filename = f"{model_name}_n{n_pc}_numc{num_c_pc}_interp{inter_p_pc:.4f}_id{i:03d}.pkl" # Changed extension
        filepath = os.path.join(model_output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        graph_counter += 1
        if (i + 1) % 10 == 0 or i == num_graphs_per_model_type -1:
            print(f"Generated {model_name} ({i+1}/{num_graphs_per_model_type}): {filename} (N: {G.number_of_nodes()}, E: {G.number_of_edges()})")

    # 3. Communities Model (most complex parameters)
    print("\n--- Generating General Community Graphs ---")
    model_name = "general_community"
    model_output_dir = os.path.join(base_output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    for i in range(num_graphs_per_model_type * 2): 
        n_gc_original = sample_n_nodes()
        
        min_nodes_per_community_gc = 1 
        max_c_gc = n_gc_original // min_nodes_per_community_gc
        if max_c_gc < 1: max_c_gc = 1
        num_c_gc = random.randint(1, max_c_gc)

        n_gc = (n_gc_original // num_c_gc) * num_c_gc
        if n_gc == 0:
            if n_gc_original > 0:
                num_c_gc = n_gc_original
                n_gc = n_gc_original
            else:
                continue
        
        community_size_gc = n_gc // num_c_gc
        
        if community_size_gc <= 1:
            k_gc = 0 
        elif community_size_gc == 2:
            k_gc = random.randint(0,1)
        else: 
            max_k_val = (community_size_gc -1) // 2 
            k_gc = random.randint(0, max_k_val) 

        inter_p_gc = sample_probability(0.0, 0.20) 
        intra_p_gc = sample_probability(0.0, 0.30) 
        
        G = generate_community_graph(n_gc, num_c_gc, k_gc, inter_p_gc, intra_p_gc)
        filename = (f"{model_name}_n{n_gc}_numc{num_c_gc}_k{k_gc}_"
                    f"interp{inter_p_gc:.4f}_intrap{intra_p_gc:.4f}_id{i:03d}.pkl") # Changed extension
        filepath = os.path.join(model_output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        graph_counter += 1
        if (i + 1) % 10 == 0 or i == (num_graphs_per_model_type * 2) -1:
             print(f"Generated {model_name} ({i+1}/{num_graphs_per_model_type*2}): {filename} (N: {G.number_of_nodes()}, E: {G.number_of_edges()})")

    total_end_time = time.time()
    print(f"\n--- Generation Complete ---")
    print(f"Generated a total of {graph_counter} graphs.")
    print(f"Total time taken: {total_end_time - total_start_time:.2f} seconds.")
    print(f"Graphs saved in: {os.path.abspath(base_output_dir)}")


if __name__ == "__main__":
    print("Starting synthetic graph generation process...")
    generate_and_save_graphs()
