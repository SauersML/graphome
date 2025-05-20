import os
import networkx as nx
import numpy as np
import time
import random
import pickle

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
    base_output_dir = "synthetic_graph_outputs_pickle_10k" 
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"Saving generated graphs to: {os.path.abspath(base_output_dir)}")

    graph_counter = 0
    total_start_time = time.time()
    # Target ~10,000 graphs: Bipartite (2500) + Path (2500) + Communities (2*2500=5000)
    num_graphs_per_model_base = 2500 

    # --- Parameter Sampling Functions ---
    def sample_n_nodes():
        # Allow for smaller and somewhat larger graphs
        if random.random() < 0.7: # 70% chance for smaller graphs
            return random.randint(10, 80) 
        else: # 30% chance for larger graphs
            return random.randint(81, 250)

    def sample_probability(low=0.0, high=0.5): # Wider general probability range
        # Skew towards lower probabilities for sparser graphs often found in real world
        # but allow higher probabilities for density.
        # Using a power law or beta distribution could be more sophisticated here,
        # but uniform provides a simple way to get varied values.
        if random.random() < 0.6: # 60% chance for sparser connections
            return random.uniform(low, high * 0.3)
        elif random.random() < 0.9: # 30% chance for medium connections
            return random.uniform(low, high * 0.6)
        else: # 10% chance for denser connections
            return random.uniform(low, high)


    # 1. Bipartite Model
    print(f"\n--- Generating {num_graphs_per_model_base} Approximate Bipartite Graphs ---")
    model_name = "bipartite"
    model_output_dir = os.path.join(base_output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    print_interval_bipartite = max(1, num_graphs_per_model_base // 20) # Print progress about 20 times

    for i in range(num_graphs_per_model_base):
        n_bp_original = sample_n_nodes()
        # Bipartite model's initial setup benefits from an even number of nodes.
        n_bp = n_bp_original if n_bp_original % 2 == 0 else n_bp_original + 1
        if n_bp < 2: n_bp = 2 # Smallest possible bipartite graph with an edge
        
        perturb_p_bp = sample_probability(0.0, 0.4) # Perturbation probability

        G = generate_approx_bipartite_graph(n_bp, perturb_p_bp)
        filename = f"{model_name}_n{n_bp}_perturb{perturb_p_bp:.4f}_id{i:04d}.pkl" 
        filepath = os.path.join(model_output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        graph_counter += 1
        if (i + 1) % print_interval_bipartite == 0 or i == num_graphs_per_model_base -1:
            print(f"Generated {model_name} ({i+1}/{num_graphs_per_model_base}): {filename} (N: {G.number_of_nodes()}, E: {G.number_of_edges()})")


    # 2. Path Model
    print(f"\n--- Generating {num_graphs_per_model_base} Path Community Graphs ---")
    model_name = "path_community"
    model_output_dir = os.path.join(base_output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    print_interval_path = max(1, num_graphs_per_model_base // 20)

    for i in range(num_graphs_per_model_base):
        n_pc_original = sample_n_nodes()
        if n_pc_original < 1: n_pc_original = 1
        
        # Allow for 1 to n communities
        num_c_pc = random.randint(1, n_pc_original) 
        
        # Adjust n_pc so each community has at least one node
        n_pc = (n_pc_original // num_c_pc) * num_c_pc
        if n_pc == 0 : 
            n_pc = num_c_pc # each community gets 1 node if original n was too small

        inter_p_pc = sample_probability(0.0, 0.6) # Inter-community connection probability

        G = generate_path_community_graph(n_pc, num_c_pc, inter_p_pc)
        filename = f"{model_name}_n{n_pc}_numc{num_c_pc}_interp{inter_p_pc:.4f}_id{i:04d}.pkl"
        filepath = os.path.join(model_output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        graph_counter += 1
        if (i + 1) % print_interval_path == 0 or i == num_graphs_per_model_base -1:
            print(f"Generated {model_name} ({i+1}/{num_graphs_per_model_base}): {filename} (N: {G.number_of_nodes()}, E: {G.number_of_edges()})")

    # 3. Communities Model (most complex parameters)
    num_graphs_communities = num_graphs_per_model_base * 2
    print(f"\n--- Generating {num_graphs_communities} General Community Graphs ---")
    model_name = "general_community"
    model_output_dir = os.path.join(base_output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    print_interval_communities = max(1, num_graphs_communities // 20)


    for i in range(num_graphs_communities): 
        n_gc_original = sample_n_nodes()
        if n_gc_original < 1: n_gc_original = 1
        
        num_c_gc = random.randint(1, n_gc_original)

        n_gc = (n_gc_original // num_c_gc) * num_c_gc
        if n_gc == 0:
            n_gc = num_c_gc
        
        community_size_gc = n_gc // num_c_gc
        
        # k for intra-community circulant structure
        if community_size_gc <= 1: # Handles single-node communities
            k_gc = 0 
        else: # community_size_gc >= 2
            # Max k is such that each node connects to k distinct others on each side without overlap.
            # For a cycle of size m, max k is (m-1)//2.
            max_k_val = (community_size_gc -1) // 2 
            k_gc = random.randint(0, max_k_val) # k can be 0 (no initial k-neighbor intra-edges)

        inter_p_gc = sample_probability(0.0, 0.4) 
        intra_p_gc = sample_probability(0.0, 0.5) 
        
        G = generate_community_graph(n_gc, num_c_gc, k_gc, inter_p_gc, intra_p_gc)
        filename = (f"{model_name}_n{n_gc}_numc{num_c_gc}_k{k_gc}_"
                    f"interp{inter_p_gc:.4f}_intrap{intra_p_gc:.4f}_id{i:04d}.pkl")
        filepath = os.path.join(model_output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        graph_counter += 1
        if (i + 1) % print_interval_communities == 0 or i == num_graphs_communities -1:
             print(f"Generated {model_name} ({i+1}/{num_graphs_communities}): {filename} (N: {G.number_of_nodes()}, E: {G.number_of_edges()})")

    total_end_time = time.time()
    print(f"\n--- Generation Complete ---")
    print(f"Generated a total of {graph_counter} graphs.")
    print(f"Total time taken: {total_end_time - total_start_time:.2f} seconds.")
    print(f"Graphs saved in: {os.path.abspath(base_output_dir)}")


if __name__ == "__main__":
    print("Starting synthetic graph generation process...")
    generate_and_save_graphs()
