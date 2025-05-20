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

# --- Parameter Sampling Functions ---
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

# --- Modified Composite Graph Generation Function ---
def generate_composite_graph_with_details(
    base_model_generators,
    num_base_components_range=(2, 5),
    node_count_per_component_func=lambda: sample_n_nodes_for_component(10, 80),
    connection_attempts_factor_range=(0.4, 2.2),
    connection_probability_func=lambda: sample_probability_for_component(0.005, 0.3)
):
    """
    Generates a composite graph and returns the graph along with detailed generation parameters.
    """
    instance_params = {
        'num_base_components_range_cfg': num_base_components_range, # Cfg suffix for configured range
        'components_details': [],
        'conn_attempts_factor_range_cfg': connection_attempts_factor_range,
    }

    if not base_model_generators:
        return nx.Graph(), instance_params

    num_components_actual = random.randint(num_base_components_range[0], num_base_components_range[1])
    instance_params['num_components_actual_val'] = num_components_actual # Val suffix for actual value
    components_graph_list = [] 

    for i in range(num_components_actual):
        model_generator_func = random.choice(base_model_generators)
        n_component_requested_val = node_count_per_component_func()
        component_graph_obj = nx.Graph() 
        # Store parameters used for *this specific component's generation*
        comp_gen_params = {'type_str': model_generator_func.__name__, 'n_requested_val': n_component_requested_val}

        if model_generator_func.__name__ == "generate_approx_bipartite_graph":
            n_bp_for_gen = n_component_requested_val if n_component_requested_val % 2 == 0 else n_component_requested_val + 1
            if n_bp_for_gen < 2: n_bp_for_gen = 2
            perturb_p_for_gen = sample_probability_for_component(0.0, 0.35)
            component_graph_obj = model_generator_func(n_bp_for_gen, perturb_p_for_gen)
            comp_gen_params.update({'perturb_p_val': perturb_p_for_gen}) # Store only specific params

        elif model_generator_func.__name__ == "generate_path_community_graph":
            n_pc_target = n_component_requested_val
            if n_pc_target < 1: n_pc_target = 1
            num_c_pc_for_gen = random.randint(1, max(1, n_pc_target))
            n_pc_for_gen = (n_pc_target // num_c_pc_for_gen) * num_c_pc_for_gen if num_c_pc_for_gen > 0 else 0
            if n_pc_for_gen == 0: n_pc_for_gen = num_c_pc_for_gen if num_c_pc_for_gen > 0 else 1
            inter_p_pc_for_gen = sample_probability_for_component(0.0, 0.5)
            component_graph_obj = model_generator_func(n_pc_for_gen, num_c_pc_for_gen, inter_p_pc_for_gen)
            comp_gen_params.update({'num_c_val': num_c_pc_for_gen, 'inter_p_val': inter_p_pc_for_gen})

        elif model_generator_func.__name__ == "generate_community_graph":
            n_gc_target = n_component_requested_val
            if n_gc_target < 1: n_gc_target = 1
            num_c_gc_for_gen = random.randint(1, max(1, n_gc_target))
            n_gc_for_gen = (n_gc_target // num_c_gc_for_gen) * num_c_gc_for_gen if num_c_gc_for_gen > 0 else 0
            if n_gc_for_gen == 0: n_gc_for_gen = num_c_gc_for_gen if num_c_gc_for_gen > 0 else 1
            community_size_gc = n_gc_for_gen // num_c_gc_for_gen if num_c_gc_for_gen > 0 else 0
            k_gc_for_gen = 0
            if community_size_gc > 1:
                max_k_val = (community_size_gc - 1) // 2
                k_gc_for_gen = random.randint(0, max(0, max_k_val))
            inter_p_gc_for_gen = sample_probability_for_component(0.0, 0.3)
            intra_p_gc_for_gen = sample_probability_for_component(0.0, 0.4)
            component_graph_obj = model_generator_func(n_gc_for_gen, num_c_gc_for_gen, k_gc_for_gen, inter_p_gc_for_gen, intra_p_gc_for_gen)
            comp_gen_params.update({'num_c_val': num_c_gc_for_gen, 'k_val': k_gc_for_gen, 'inter_p_val': inter_p_gc_for_gen, 'intra_p_val': intra_p_gc_for_gen})
        
        # Store the actual number of nodes of the generated sub-component
        comp_gen_params['n_nodes_final_val'] = component_graph_obj.number_of_nodes()
        instance_params['components_details'].append(comp_gen_params)

        if component_graph_obj.number_of_nodes() > 0:
            components_graph_list.append(component_graph_obj)
        elif num_components_actual == 1: # Only one component requested and it was empty
            return nx.Graph(), instance_params

    if not components_graph_list: # All components were empty
        return nx.Graph(), instance_params

    composite_graph = nx.disjoint_union_all(components_graph_list)
    # instance_params['nodes_after_disjoint_union_val'] = composite_graph.number_of_nodes() # Removed as redundant

    if composite_graph.number_of_nodes() == 0:
        return composite_graph, instance_params

    conn_attempts_factor_actual_val = random.uniform(connection_attempts_factor_range[0], connection_attempts_factor_range[1])
    num_conn_attempts_calculated = int(conn_attempts_factor_actual_val * composite_graph.number_of_nodes())
    effective_num_components_to_connect = len(components_graph_list)
    min_attempts = max(1, effective_num_components_to_connect - 1) if effective_num_components_to_connect > 1 else 0
    max_sensible_attempts = composite_graph.number_of_nodes() * 2 + effective_num_components_to_connect
    num_conn_attempts_actual_val = max(min_attempts, min(num_conn_attempts_calculated, max_sensible_attempts))

    instance_params['conn_attempts_factor_actual_val'] = conn_attempts_factor_actual_val
    instance_params['num_conn_attempts_actual_val'] = num_conn_attempts_actual_val

    edges_added_stitching_val = 0
    for _ in range(num_conn_attempts_actual_val):
        current_ccs = list(nx.connected_components(composite_graph))
        if len(current_ccs) < 2:
            break
        idx1, idx2 = random.sample(range(len(current_ccs)), 2)
        component1_nodes = list(current_ccs[idx1])
        component2_nodes = list(current_ccs[idx2])
        if not component1_nodes or not component2_nodes: continue
        node_from_c1 = random.choice(component1_nodes)
        node_from_c2 = random.choice(component2_nodes)
        p_connect_val = connection_probability_func() # This is sampled per attempt
        if random.random() < p_connect_val:
            if not composite_graph.has_edge(node_from_c1, node_from_c2):
                composite_graph.add_edge(node_from_c1, node_from_c2)
                edges_added_stitching_val +=1
    instance_params['edges_added_stitching_val'] = edges_added_stitching_val

    return composite_graph, instance_params

# --- Helper function to format composite parameters for filename (ultra-abbreviated) ---
def _format_composite_params_for_filename(params_dict, cns_desc_str, csps_desc_str):
    parts = []
    parts.append(f"CS({cns_desc_str})") # CNS = Component Node Sampler
    parts.append(f"CP({csps_desc_str})") # CPS = Component Stitching Probability Sampler
    
    ncr_cfg = params_dict.get('num_base_components_range_cfg', ('?','?'))
    parts.append(f"CR({ncr_cfg[0]}-{ncr_cfg[1]})") # CR = Components Range
    parts.append(f"CA{params_dict.get('num_components_actual_val', 0)}") # CA = Components Actual
    
    comp_fn_strs = []
    for i, comp_data in enumerate(params_dict.get('components_details', [])):
        type_name_full = comp_data.get('type_str','uk')
        # Abbreviate type: generate_approx_bipartite_graph -> ab, path_community_graph -> pc, community_graph -> cg
        type_abbrev = "".join(word[0] for word in type_name_full.replace("generate_","").replace("_graph","").split("_")).lower()
        
        s = f"{i}{type_abbrev}" # Index and type
        s += f"nr{comp_data.get('n_requested_val',0)}" # nr = n_requested
        
        # Type-specific parameters (use single letter keys if possible)
        if type_name_full == "generate_approx_bipartite_graph":
            s += f"p{comp_data.get('perturb_p_val',0):.2f}" # p = perturb_p
        elif type_name_full == "generate_path_community_graph":
            s += f"c{comp_data.get('num_c_val',0)}" # c = num_communities
            s += f"i{comp_data.get('inter_p_val',0):.2f}" # i = inter_p
        elif type_name_full == "generate_community_graph":
            s += f"c{comp_data.get('num_c_val',0)}"
            s += f"k{comp_data.get('k_val',0)}" # k = k_value
            s += f"i{comp_data.get('inter_p_val',0):.2f}"
            s += f"t{comp_data.get('intra_p_val',0):.2f}" # t = intra_p (t for inTra)
        
        s += f"NF{comp_data.get('n_nodes_final_val',0)}" # NF = Nodes Final (for this component)
        comp_fn_strs.append(s)
    
    if comp_fn_strs:
        parts.append('Cmps[' + '_'.join(comp_fn_strs) + ']') # Cmps = Components List

    caf_cfg = params_dict.get('conn_attempts_factor_range_cfg', ('?','?'))
    parts.append(f"AR({caf_cfg[0]:.1f}-{caf_cfg[1]:.1f})") # AR = Attempts Range (factor)
    parts.append(f"AA{params_dict.get('conn_attempts_factor_actual_val', 0.0):.2f}") # AA = Attempts Actual (factor)
    parts.append(f"NA{params_dict.get('num_conn_attempts_actual_val', 0)}") # NA = Num Attempts (actual count)
    parts.append(f"ES{params_dict.get('edges_added_stitching_val',0)}") # ES = Edges Stitched
    
    return "_" + "_".join(filter(None, parts))


# --- Main Data Generation Logic ---
def generate_and_save_graphs():
    base_output_dir = "/content/g_data" 
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"Saving generated graphs to: {os.path.abspath(base_output_dir)}")

    graph_counter = 0
    total_start_time = time.time()

    num_graphs_per_model_base = 10000
    num_graphs_communities_model = num_graphs_per_model_base * 2
    num_graphs_composite_model = 10000

    # 1. Bipartite Model
    print(f"\n--- Generating {num_graphs_per_model_base} Approximate Bipartite Graphs ---")
    model_name_bp = "bipartite"
    model_output_dir_bp = os.path.join(base_output_dir, model_name_bp)
    os.makedirs(model_output_dir_bp, exist_ok=True)
    print_interval_bp = max(1, num_graphs_per_model_base // 20)
    for i in range(num_graphs_per_model_base):
        n_bp_original = sample_n_nodes_for_component()
        n_bp = n_bp_original if n_bp_original % 2 == 0 else n_bp_original + 1
        if n_bp < 2: n_bp = 2
        perturb_p_bp = sample_probability_for_component(0.0, 0.4)
        G_bp = generate_approx_bipartite_graph(n_bp, perturb_p_bp)
        filename_bp = f"{model_name_bp}_n{n_bp}_perturb{perturb_p_bp:.4f}_id{i:04d}.pkl"
        filepath_bp = os.path.join(model_output_dir_bp, filename_bp)
        with open(filepath_bp, 'wb') as f:
            pickle.dump(G_bp, f, pickle.HIGHEST_PROTOCOL)
        graph_counter += 1
        if (i + 1) % print_interval_bp == 0 or i == num_graphs_per_model_base - 1:
            print(f"Generated {model_name_bp} ({i+1}/{num_graphs_per_model_base}): {filename_bp} (N: {G_bp.number_of_nodes()}, E: {G_bp.number_of_edges()})")

    # 2. Path Model
    print(f"\n--- Generating {num_graphs_per_model_base} Path Community Graphs ---")
    model_name_pc = "path_community"
    model_output_dir_pc = os.path.join(base_output_dir, model_name_pc)
    os.makedirs(model_output_dir_pc, exist_ok=True)
    print_interval_pc = max(1, num_graphs_per_model_base // 20)
    for i in range(num_graphs_per_model_base):
        n_pc_original = sample_n_nodes_for_component()
        if n_pc_original < 1: n_pc_original = 1
        num_c_pc = random.randint(1, n_pc_original)
        n_pc = (n_pc_original // num_c_pc) * num_c_pc if num_c_pc > 0 else 0
        if n_pc == 0 : n_pc = num_c_pc if num_c_pc > 0 else 1
        inter_p_pc = sample_probability_for_component(0.0, 0.6)
        G_pc = generate_path_community_graph(n_pc, num_c_pc, inter_p_pc)
        filename_pc = f"{model_name_pc}_n{n_pc}_numc{num_c_pc}_interp{inter_p_pc:.4f}_id{i:04d}.pkl"
        filepath_pc = os.path.join(model_output_dir_pc, filename_pc)
        with open(filepath_pc, 'wb') as f:
            pickle.dump(G_pc, f, pickle.HIGHEST_PROTOCOL)
        graph_counter += 1
        if (i + 1) % print_interval_pc == 0 or i == num_graphs_per_model_base - 1:
            print(f"Generated {model_name_pc} ({i+1}/{num_graphs_per_model_base}): {filename_pc} (N: {G_pc.number_of_nodes()}, E: {G_pc.number_of_edges()})")

    # 3. Communities Model
    print(f"\n--- Generating {num_graphs_communities_model} General Community Graphs ---")
    model_name_gc = "general_community"
    model_output_dir_gc = os.path.join(base_output_dir, model_name_gc)
    os.makedirs(model_output_dir_gc, exist_ok=True)
    print_interval_gc = max(1, num_graphs_communities_model // 20)
    for i in range(num_graphs_communities_model):
        n_gc_original = sample_n_nodes_for_component()
        if n_gc_original < 1: n_gc_original = 1
        num_c_gc = random.randint(1, n_gc_original)
        n_gc = (n_gc_original // num_c_gc) * num_c_gc if num_c_gc > 0 else 0
        if n_gc == 0: n_gc = num_c_gc if num_c_gc > 0 else 1
        community_size_gc = n_gc // num_c_gc if num_c_gc > 0 else 0
        k_gc = 0
        if community_size_gc > 1:
            max_k_val = (community_size_gc - 1) // 2
            k_gc = random.randint(0, max(0,max_k_val))
        inter_p_gc = sample_probability_for_component(0.0, 0.4)
        intra_p_gc = sample_probability_for_component(0.0, 0.5)
        G_gc = generate_community_graph(n_gc, num_c_gc, k_gc, inter_p_gc, intra_p_gc)
        filename_gc = (f"{model_name_gc}_n{n_gc}_numc{num_c_gc}_k{k_gc}_"
                       f"interp{inter_p_gc:.4f}_intrap{intra_p_gc:.4f}_id{i:04d}.pkl")
        filepath_gc = os.path.join(model_output_dir_gc, filename_gc)
        with open(filepath_gc, 'wb') as f:
            pickle.dump(G_gc, f, pickle.HIGHEST_PROTOCOL)
        graph_counter += 1
        if (i + 1) % print_interval_gc == 0 or i == num_graphs_communities_model - 1:
            print(f"Generated {model_name_gc} ({i+1}/{num_graphs_communities_model}): {filename_gc} (N: {G_gc.number_of_nodes()}, E: {G_gc.number_of_edges()})")

    # 4. Composite Model
    available_base_generators_comp = [
        generate_community_graph,
        generate_approx_bipartite_graph,
        generate_path_community_graph
    ]
    print(f"\n--- Generating {num_graphs_composite_model} Composite Graphs (Ultra-Abbreviated Detailed Filenames) ---")
    model_name_comp = "composite"
    model_output_dir_comp = os.path.join(base_output_dir, model_name_comp)
    os.makedirs(model_output_dir_comp, exist_ok=True)
    print_interval_comp = max(1, num_graphs_composite_model // 20)

    # Parameters for the lambda functions, to be included in filename descriptions
    CNS_MIN_CFG, CNS_MAX_CFG, CNS_BIAS_CFG, CNS_SMAX_CFG = 5, 60, 0.8, 40
    CSPS_LOW_CFG, CSPS_HIGH_CFG = 0.001, 0.4
    
    # Create descriptive strings for these lambda configurations
    cns_desc_str_for_fn = f"M{CNS_MIN_CFG}X{CNS_MAX_CFG}B{CNS_BIAS_CFG:.1f}S{CNS_SMAX_CFG}"
    csps_desc_str_for_fn = f"L{CSPS_LOW_CFG:.2f}H{CSPS_HIGH_CFG:.1f}" # .2f for low prob, .1f for high
    
    # Define the sampler functions using these configurations
    comp_node_sampler = lambda: sample_n_nodes_for_component(CNS_MIN_CFG, CNS_MAX_CFG, CNS_BIAS_CFG, CNS_SMAX_CFG)
    comp_stitch_prob_sampler = lambda: sample_probability_for_component(CSPS_LOW_CFG, CSPS_HIGH_CFG) # Uses default biases in sample_probability_for_component

    for i in range(num_graphs_composite_model):
        # These ranges are sampled per graph instance for variety
        num_components_min_inst = random.randint(2, 3)
        num_components_max_inst = random.randint(num_components_min_inst, 6)
        
        conn_attempts_min_factor_inst = random.uniform(0.2, 0.9)
        conn_attempts_max_factor_inst = random.uniform(conn_attempts_min_factor_inst + 0.1, 2.8)

        G_comp, instance_gen_params = generate_composite_graph_with_details(
            base_model_generators=available_base_generators_comp,
            num_base_components_range=(num_components_min_inst, num_components_max_inst),
            node_count_per_component_func=comp_node_sampler, # Uses the globally configured sampler
            connection_attempts_factor_range=(conn_attempts_min_factor_inst, conn_attempts_max_factor_inst),
            connection_probability_func=comp_stitch_prob_sampler # Uses the globally configured sampler
        )

        if G_comp.number_of_nodes() == 0:
            print(f"Skipped saving empty composite graph id {i:04d}")
            continue

        # Pass the descriptive strings of the *configurations* of the samplers
        detailed_param_string = _format_composite_params_for_filename(
            instance_gen_params, 
            cns_desc_str_for_fn, 
            csps_desc_str_for_fn
        )
        
        filename_comp = f"{model_name_comp}_id{i:04d}_N{G_comp.number_of_nodes()}{detailed_param_string}.pkl" # N instead of nodes
        filepath_comp = os.path.join(model_output_dir_comp, filename_comp)

        with open(filepath_comp, 'wb') as f:
            pickle.dump(G_comp, f, pickle.HIGHEST_PROTOCOL)
        graph_counter += 1
        if (i + 1) % print_interval_comp == 0 or i == num_graphs_composite_model - 1:
            # Display a portion of the filename for console readability
            short_fn_display = filename_comp[:150] + "..." if len(filename_comp) > 150 else filename_comp
            print(f"Generated {model_name_comp} ({i+1}/{num_graphs_composite_model}): {short_fn_display} (N: {G_comp.number_of_nodes()}, E: {G_comp.number_of_edges()})")

    total_end_time = time.time()
    print(f"\n--- Generation Complete ---")
    print(f"Generated a total of {graph_counter} graphs.")
    print(f"Total time taken: {total_end_time - total_start_time:.2f} seconds.")
    print(f"Graphs saved in: {os.path.abspath(base_output_dir)}")

if __name__ == "__main__":
    print("Starting synthetic graph generation process...")
    generate_and_save_graphs()
