# src

## File Overview

### `convert.rs`

This module handles converting GFA files into adjacency matrices in edge list format. It processes GFA files in two main steps:

- **Segment Parsing**: Extracts all segments (nodes) from the GFA file and assigns them unique indices.
- **Link Parsing**: Reads the links (edges) from the GFA file and writes both forward and backward edges to ensure bidirectionality.

#### Key Functions:
- `convert_gfa_to_edge_list`: Orchestrates the GFA conversion into an adjacency matrix.
- `parse_segments`: Parses GFA segments and assigns indices.
- `parse_links_and_write_edges`: Extracts links and writes edges in a bidirectional format.

### `eigen.rs`

This module performs eigendecomposition on the Laplacian matrix derived from adjacency matrices. It uses either LAPACK's `dsbevd` function or nalgebra's `SymmetricEigen` depending on the structure of the matrix.

#### Key Functions:
- `call_eigendecomp`: Selects between LAPACK and nalgebra eigendecomposition based on the matrix bandedness.
- `max_band`: Determines the bandedness of the Laplacian matrix.
- `compute_eigenvalues_and_vectors_sym_band`: Uses LAPACK for eigendecomposition of banded matrices.
- `compute_eigenvalues_and_vectors_sym`: Uses nalgebra for more general eigendecomposition.

### `extract.rs`

This module extracts adjacency submatrices from an edge list and performs subsequent analysis, including the computation of the Laplacian matrix and eigendecomposition.

#### Key Functions:
- `extract_and_analyze_submatrix`: Extracts a submatrix, computes the Laplacian, and performs eigendecomposition.
- `load_adjacency_matrix`: Loads an adjacency matrix from a binary edge list file.

### `main.rs`

The entry point for the source code. It sets up a command-line interface (CLI) using `clap` for interacting with the conversion and extraction functionalities. This file defines the CLI commands and arguments for running the `convert` and `extract` processes.

### `map.rs`
Parse a GFA and untangle PAF for node↔hg38 lookups.
Usage examples:

  ```
  ./target/release/graphome map --gfa hprc-v1.0-pggb.gfa --paf ../hprc-v1.0-pggb.all.vs.grch38.untangle-m10000-s0-j0.paf node2coord 10127854
```
  
  ```
  ./target/release/graphome map --gfa hprc-v1.0-pggb.gfa --paf ../hprc-v1.0-pggb.all.vs.grch38.untangle-m10000-s0-j0.paf coord2node "grch38#chr1:228557148-228557149"
```

### `viz.rs`
Visualize a pangenome node range.
```
./target/release/graphome viz --gfa hprc-v1.0-pggb.gfa --start-node "211000" --end-node "211005" --output-tga visualization_211000_211005.tga
```
