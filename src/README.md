# src

## File Overview

### `convert.rs`

This module handles converting GFA files into adjacency matrices in edge list format. It processes GFA files in two main steps:

- **Segment Parsing**: Extracts all segments (nodes) from the GFA file and assigns them unique indices.
- **Link Parsing**: Reads the links (edges) from the GFA file and write edges.

```
./target/release/graphome convert --input hprc-v1.0-pggb.gfa --output edges.gam
```

01/23/2025 benchmark 32 core 128 GB: Completed in 1301.39s seconds.

#### Key Functions:
- `convert_gfa_to_edge_list`: Orchestrates the GFA conversion into an adjacency matrix.
- `parse_segments`: Parses GFA segments and assigns indices.
- `parse_links_and_write_edges`: Extracts links and writes edges in a bidirectional format.

### `eigen_print.rs`

This module performs eigendecomposition on the Laplacian matrix derived from adjacency matrices.

#### Key Functions:
- `call_eigendecomp`: Computes the eigendecomposition of dense Laplacians with faer's self-adjoint solver.
- `compute_eigenvalues_and_vectors_sym`: Uses faer to obtain eigenpairs for symmetric matrices.

### `extract.rs`

This module extracts adjacency submatrices from an edge list and performs subsequent analysis, including the computation of the Laplacian matrix and eigendecomposition.

```
./target/release/graphome extract --input edges.gam --start-node 5 --end-node 100 --output output_dir
```

Issue: performance depends heavily on the size of the .gam file. Example with large .gam: ⏰ Completed in 193.04s seconds. (Completes in a few seconds for small .gam.)

#### Key Functions:
- `extract_and_analyze_submatrix`: Extracts a submatrix, computes the Laplacian, and performs eigendecomposition.
- `load_adjacency_matrix`: Loads an adjacency matrix from a binary edge list file.

### `main.rs`

The entry point for the source code. It sets up a command-line interface (CLI) using `clap` for interacting with the conversion and extraction functionalities. This file defines the CLI commands and arguments for running the `convert` and `extract` processes.

### `map.rs`
Parse a GFA and untangle PAF for node↔hg38 lookups.
Usage examples:

  ```
  ./target/release/graphome map --gfa ../hprc-v1.0-pggb.gfa --paf ../hprc-v1.0-pggb.all.vs.grch38.untangle-m10000-s0-j0.paf node2coord 10127854
```
  
  ```
  ./target/release/graphome map --gfa ../hprc-v1.0-pggb.gfa --paf ../hprc-v1.0-pggb.all.vs.grch38.untangle-m10000-s0-j0.paf coord2node "grch38#chr1:228557148-228557149"
```

### `make_sequence.rs`
Write the FASTA-format DNA sequence corresponding to a coordinate range in a .gfa sample.
```
./target/release/graphome make-sequence --gfa ../hprc-v1.0-pggb.gfa --paf ../hprc-v1.0-pggb.all.vs.grch38.untangle-m10000-s0-j0.paf --region "grch38#chr19:44905791-44909393" --sample HG00438 --output sequence
```


### `viz.rs`
Visualize a pangenome node range.
```
./target/release/graphome viz --gfa hprc-v1.0-pggb.gfa --start-node "211000" --end-node "211005" --output-tga visualization_211000_211005.tga
```

### `embed.rs`
```
./target/release/graphome embed --input adjacency_matrix.gam --start-node 0 --end-node 100
```
