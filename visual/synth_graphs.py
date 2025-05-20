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

# --- Parameter Sampling Functions (used by both main loop and composite generator) ---
def sample_n_nodes_for_component(min_n=10, max_n=150, short_graph_bias=0.7, short_graph_max_n=70):
    """Samples number of nodes, with a bias towards smaller graphs."""
    if random.random() < short_graph_bias:
        return random.randint(min_n, short_graph_max_n) 
    else:
        return random.randint(short_graph_max_n + 1, max_n)

def sample_probability_for_component(low=0.0, high=0.5, sparse_bias=0.6, medium_bias=0.3):
    """Samples a probability, with a bias towards sparser connections."""
    rand_val = random.random()
    if rand_val < sparse_bias:
        return random.uniform(low, high * 0.3)
    elif rand_val < sparse_bias + medium_bias:
        return random.uniform(high * 0.3, high * 0.6)
    else:
        return random.uniform(high * 0.6, high)

# --- Composite Graph Generation Function ---
def generate_composite_graph(
    base_model_generators, 
    num_base_components_range=(2, 5),
    node_count_per_component_func=lambda: sample_n_nodes_for_component(10, 80), # Use main sampler
    connection_attempts_factor_range=(0.4, 2.2),
    connection_probability_func=lambda: sample_probability_for_component(0.005, 0.3) # Use main sampler
):
    """
    Generates a composite graph by creating and stitching together multiple base graphs.
    Base components will use the internal parameter sampling.
    """
    if not base_model_generators:
        return nx.Graph()

    num_components = random.randint(num_base_components_range[0], num_base_components_range[1])
    components_list = []
    
    for _ in range(num_components):
        model_generator_func = random.choice(base_model_generators)
        n_component = node_count_per_component_func()
        component_graph = nx.Graph()

        if model_generator_func.__name__ == "generate_approx_bipartite_graph":
            n_bp = n_component if n_component % 2 == 0 else n_component + 1
            if n_bp < 2: n_bp = 2
            perturb_p_bp = sample_probability_for_component(0.0, 0.35)
            component_graph = model_generator_func(n_bp, perturb_p_bp)

        elif model_generator_func.__name__ == "generate_path_community_graph":
            if n_component < 1: n_component = 1
            num_c_pc = random.randint(1, max(1, n_component // 1)) # Each node can be a community
            n_pc = (n_component // num_c_pc) * num_c_pc
            if n_pc == 0: n_pc = num_c_pc if num_c_pc > 0 else 1
            
            inter_p_pc = sample_probability_for_component(0.0, 0.5)
            component_graph = model_generator_func(n_pc, num_c_pc, inter_p_pc)

        elif model_generator_func.__name__ == "generate_community_graph":
            if n_component < 1: n_component = 1
            num_c_gc = random.randint(1, max(1, n_component // 1))
            n_gc = (n_component // num_c_gc) * num_c_gc
            if n_gc == 0: n_gc = num_c_gc if num_c_gc > 0 else 1
            
            community_size_gc = n_gc // num_c_gc if num_c_gc > 0 else 0
            
            k_gc = 0
            if community_size_gc > 1:
                 max_k_val = (community_size_gc - 1) // 2
                 k_gc = random.randint(0, max(0,max_k_val))
            
            inter_p_gc = sample_probability_for_component(0.0, 0.3)
            intra_p_gc = sample_probability_for_component(0.0, 0.4)
            component_graph = model_generator_func(n_gc, num_c_gc, k_gc, inter_p_gc, intra_p_gc)
        
        if component_graph.number_of_nodes() > 0:
            components_list.append(component_graph)
        elif num_components == 1:
             return nx.Graph()

    if not components_list:
        return nx.Graph()

    composite_graph = nx.disjoint_union_all(components_list)
    
    if composite_graph.number_of_nodes() == 0:
        return composite_graph

    connection_attempts_factor = random.uniform(connection_attempts_factor_range[0], connection_attempts_factor_range[1])
    num_connection_attempts = int(connection_attempts_factor * composite_graph.number_of_nodes())
    # at least num_components-1 attempts to connect all, but not excessively many
    min_attempts = max(1, num_components - 1) if num_components > 1 else 0
    num_connection_attempts = max(min_attempts, min(num_connection_attempts, composite_graph.number_of_nodes() * 2 + num_components))


    for _ in range(num_connection_attempts):
        list_of_ccs_tuples = list(enumerate(nx.connected_components(composite_graph)))
        
        if len(list_of_ccs_tuples) < 2:
            break 
        
        # Select two different connected components to bridge
        # we pick from the original indices of components before any merging
        comp_indices = random.sample(range(len(list_of_ccs_tuples)), 2)
        
        component1_nodes = list(list_of_ccs_tuples[comp_indices[0]][1]) # Get node set from tuple
        component2_nodes = list(list_of_ccs_tuples[comp_indices[1]][1]) # Get node set from tuple

        if not component1_nodes or not component2_nodes: continue

        node_from_c1 = random.choice(component1_nodes)
        node_from_c2 = random.choice(component2_nodes)
        
        p_connect = connection_probability_func()
        
        if random.random() < p_connect and not composite_graph.has_edge(node_from_c1, node_from_c2):
            composite_graph.add_edge(node_from_c1, node_from_c2)
            
    return composite_graph

# --- Main Data Generation Logic ---

def generate_and_save_graphs():
    """
    Generates a diverse set of synthetic graphs using imported and composite models,
    and saves them using Python's pickle.
    """
    base_output_dir = "synthetic_graph_outputs_pickle_10k_plus_composite" 
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"Saving generated graphs to: {os.path.abspath(base_output_dir)}")

    graph_counter = 0
    total_start_time = time.time()
    
    # Target ~10,000 base graphs + additional composite graphs
    # Bipartite (2000) + Path (2000) + Communities (2*2000=4000) + Composite (2000) = 10000
    num_graphs_per_model_base = 2000
    num_graphs_communities_model = num_graphs_per_model_base * 2
    num_graphs_composite_model = 2000


    # 1. Bipartite Model
    print(f"\n--- Generating {num_graphs_per_model_base} Approximate Bipartite Graphs ---")
    model_name = "bipartite"
    model_output_dir = os.path.join(base_output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    print_interval = max(1, num_graphs_per_model_base // 20) 

    for i in range(num_graphs_per_model_base):
        n_bp_original = sample_n_nodes_for_component()
        n_bp = n_bp_original if n_bp_original % 2 == 0 else n_bp_original + 1
        if n_bp < 2: n_bp = 2
        perturb_p_bp = sample_probability_for_component(0.0, 0.4)
        G = generate_approx_bipartite_graph(n_bp, perturb_p_bp)
        filename = f"{model_name}_n{n_bp}_perturb{perturb_p_bp:.4f}_id{i:04d}.pkl" 
        filepath = os.path.join(model_output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        graph_counter += 1
        if (i + 1) % print_interval == 0 or i == num_graphs_per_model_base -1:
            print(f"Generated {model_name} ({i+1}/{num_graphs_per_model_base}): {filename} (N: {G.number_of_nodes()}, E: {G.number_of_edges()})")

    # 2. Path Model
    print(f"\n--- Generating {num_graphs_per_model_base} Path Community Graphs ---")
    model_name = "path_community"
    model_output_dir = os.path.join(base_output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    print_interval = max(1, num_graphs_per_model_base // 20)

    for i in range(num_graphs_per_model_base):
        n_pc_original = sample_n_nodes_for_component()
        if n_pc_original < 1: n_pc_original = 1
        num_c_pc = random.randint(1, n_pc_original) 
        n_pc = (n_pc_original // num_c_pc) * num_c_pc
        if n_pc == 0 : n_pc = num_c_pc if num_c_pc > 0 else 1
        inter_p_pc = sample_probability_for_component(0.0, 0.6)
        G = generate_path_community_graph(n_pc, num_c_pc, inter_p_pc)
        filename = f"{model_name}_n{n_pc}_numc{num_c_pc}_interp{inter_p_pc:.4f}_id{i:04d}.pkl"
        filepath = os.path.join(model_output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        graph_counter += 1
        if (i + 1) % print_interval == 0 or i == num_graphs_per_model_base -1:
            print(f"Generated {model_name} ({i+1}/{num_graphs_per_model_base}): {filename} (N: {G.number_of_nodes()}, E: {G.number_of_edges()})")

    # 3. Communities Model
    print(f"\n--- Generating {num_graphs_communities_model} General Community Graphs ---")
    model_name = "general_community"
    model_output_dir = os.path.join(base_output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    print_interval = max(1, num_graphs_communities_model // 20)

    for i in range(num_graphs_communities_model): 
        n_gc_original = sample_n_nodes_for_component()
        if n_gc_original < 1: n_gc_original = 1
        num_c_gc = random.randint(1, n_gc_original)
        n_gc = (n_gc_original // num_c_gc) * num_c_gc
        if n_gc == 0: n_gc = num_c_gc if num_c_gc > 0 else 1
        community_size_gc = n_gc // num_c_gc if num_c_gc > 0 else 0
        k_gc = 0
        if community_size_gc > 1:
            max_k_val = (community_size_gc -1) // 2 
            k_gc = random.randint(0, max(0,max_k_val))
        inter_p_gc = sample_probability_for_component(0.0, 0.4) 
        intra_p_gc = sample_probability_for_component(0.0, 0.5) 
        G = generate_community_graph(n_gc, num_c_gc, k_gc, inter_p_gc, intra_p_gc)
        filename = (f"{model_name}_n{n_gc}_numc{num_c_gc}_k{k_gc}_"
                    f"interp{inter_p_gc:.4f}_intrap{intra_p_gc:.4f}_id{i:04d}.pkl")
        filepath = os.path.join(model_output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        graph_counter += 1
        if (i + 1) % print_interval == 0 or i == num_graphs_communities_model -1:
             print(f"Generated {model_name} ({i+1}/{num_graphs_communities_model}): {filename} (N: {G.number_of_nodes()}, E: {G.number_of_edges()})")

    # 4. Composite Model
    available_base_generators = [
        generate_community_graph,
        generate_approx_bipartite_graph,
        generate_path_community_graph
    ]
    print(f"\n--- Generating {num_graphs_composite_model} Composite Graphs ---")
    model_name = "composite"
    model_output_dir = os.path.join(base_output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    print_interval = max(1, num_graphs_composite_model // 20)

    for i in range(num_graphs_composite_model):
        num_components_min = random.randint(2, 3)
        num_components_max = random.randint(num_components_min, 6)
        
        # Composite components can be smaller on average
        comp_node_sampler = lambda: sample_n_nodes_for_component(5, 60, 0.8, 40) 

        conn_attempts_min_factor = random.uniform(0.2, 0.9)
        conn_attempts_max_factor = random.uniform(conn_attempts_min_factor + 0.1, 2.8)

        comp_stitch_prob_sampler = lambda: sample_probability_for_component(0.001, 0.4)

        G = generate_composite_graph(
            base_model_generators=available_base_generators,
            num_base_components_range=(num_components_min, num_components_max),
            node_count_per_component_func=comp_node_sampler,
            connection_attempts_factor_range=(conn_attempts_min_factor, conn_attempts_max_factor),
            connection_probability_func=comp_stitch_prob_sampler
        )

        if G.number_of_nodes() == 0:
            print(f"Skipped saving empty composite graph id {i:04d}")
            continue

        # Extract some high-level parameters for filename
        filename = f"{model_name}_id{i:04d}_nodes{G.number_of_nodes()}.pkl"
        filepath = os.path.join(model_output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        graph_counter += 1
        if (i + 1) % print_interval == 0 or i == num_graphs_composite_model - 1:
            print(f"Generated {model_name} ({i+1}/{num_graphs_composite_model}): {filename} (N: {G.number_of_nodes()}, E: {G.number_of_edges()})")


    total_end_time = time.time()
    print(f"\n--- Generation Complete ---")
    print(f"Generated a total of {graph_counter} graphs.")
    print(f"Total time taken: {total_end_time - total_start_time:.2f} seconds.")
    print(f"Graphs saved in: {os.path.abspath(base_output_dir)}")


if __name__ == "__main__":
    print("Starting synthetic graph generation process...")
    generate_and_save_graphs()
