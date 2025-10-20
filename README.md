# Graphome

Graphome is a Rust toolchain for converting large Graphical Fragment Assembly (GFA) files into compact binary adjacency data and for exploring spectral properties of the resulting graphs. The utilities are written to operate on very large genome graphs (tested on inputs larger than 80 GB) while keeping memory use manageable by streaming the source file and writing binary edge data.

After conversion, Graphome exposes commands for extracting subgraphs, producing Laplacian windows, running eigenvalue analyses, aligning manifolds, mapping genomic coordinates, and generating visualizations. The same tooling is applicable to other graph-like GFAs provided they follow the standard record structure.

---

## Overview

Graphome provides a streamlined pipeline for turning raw GFA data into compact adjacency data that are ready for downstream analysis. By leveraging Rust's performance and memory safety guarantees, Graphome enables iteration over very large graph genomes (validated on 89 GB+ GFA files) without requiring oversized compute nodes.

## Capabilities

- High-throughput conversion from GFA to a binary edge list (`.gam`) using streaming I/O and parallel parsing.
- Extraction of node ranges, Laplacian matrices, and eigenvectors for targeted windows.
- Sliding-window analysis and NGEC statistics to trace local spectral changes.
- Coordinate mapping between GFA nodes and reference positions through paired GFA/PAF inputs.
- Visualization helpers that render node subsets, build 3D embeddings, and assemble animations.

## Statistics & Analytics Modules

Graphome can calculate various statistics of a graph.

- Node connectivities.
- Eigenvalues and eigenvectors of a graph as a whole or for specific node ranges.
- Entropy of a graph, or a specific region of the graph.
- Clustering of paths in a graph (in progress).
- Magnitude of differences between paths (in progress).
- Path-level convex hull overlap (in progress).

Eigendecomposition can be run in sliding windows, and manifold alignment modules allow comparisons between adjacent windows.

## Installation

Ensure you have a recent version of [Rust](https://www.rust-lang.org/tools/install) and `cargo` installed.

Build from source:

```bash
git clone https://github.com/SauersML/graphome
cd graphome
cargo build --release
```

Graphome performs eigendecomposition with [faer](https://crates.io/crates/faer), a pure Rust linear algebra library. No external LAPACK or BLAS dependencies are required.

## Quick Start

```bash
# Convert a GFA file to a binary edge list (pairs of node indices)
graphome convert --input input.gfa --output graphome.gam

# Extract a subgraph, compute eigenvectors, and save the window to disk
graphome extract --input graphome.gam --start-node 100 --end-node 500 --output chr20_window.gam

# Generate overlapping Laplacian windows and inspect their spectra
graphome extract-windows --input graphome.gam --start-node 0 --end-node 1000 \
  --window-size 200 --overlap 50 --output windows/
graphome analyze-windows --input windows/
```

## Usage

### Basic Conversion

Convert a GFA file to the binary edge list format used by Graphome:

```bash
graphome convert --input input.gfa --output output.gam
```

### View Matrix Statistics (planned)

The `stats` command is under active development. Refer to the entropy and window utilities (and supporting scripts) for current experimental workflows.

### Extract Subgraph

```bash
graphome extract --input input.gam --start-node 1000000 --end-node 2000000 --output chr20_window.gam
```

### Command Reference

Graphome exposes multiple subcommands tailored to specific analysis workflows:

- `convert`: Convert a GFA file to an adjacency edge list stored as binary pairs of `u32` indices.
- `extract`: Extract a node range from a `.gam` file and perform eigenanalysis.
- `extract-matrices`: Save adjacency and Laplacian matrices for a node window as `.npy` files.
- `extract-windows`: Generate overlapping windows of Laplacian matrices in parallel for large-scale spectral analysis.
- `analyze-windows`: Post-process extracted windows and compute NGEC statistics.
- `map`: Translate between graph node IDs and genomic coordinates using paired GFA/PAF inputs.
  - `node2coord`: Map a node ID to its genomic coordinates.
  - `coord2node`: Find nodes covering a genomic region such as `grch38#chr1:100000-200000`.
- `viz`: Render a range of nodes as a colored TGA, optionally running a force-directed refinement step.
- `embed`: Produce a 3D embedding for a submatrix and generate a rotating visualization.
- `make-sequence`: Extract nucleotide sequence for a region based on aligned coordinates.
- `gfa2gbz`: Convert a GFA file to GBZ format for interoperable graph tooling.

Refer to `graphome --help` or `graphome <command> --help` for argument descriptions and defaults.

## File Format Specification

### GAM Format (Genome/Graph Adjacency Matrix)

Graphome writes `.gam` files as a binary edge list:

- Records are written sequentially with no textual delimiters.
- Each record is exactly 8 bytes: a little-endian `u32` source node index followed by a little-endian `u32` target node index.
- Every undirected adjacency is written twice—once for each direction—to simplify downstream spectral operations.
- Node indices correspond to the sorted order of `S` records in the source GFA.

## Working with GAM Files

To inspect the raw edge pairs in a `.gam` file, read the binary data as little-endian `u32` integers:

```
python - <<'PY'
import struct

with open("submatrix.gam", "rb") as handle:
    for _ in range(10):
        chunk = handle.read(8)
        if len(chunk) < 8:
            break
        src_idx, dst_idx = struct.unpack("<II", chunk)
        print(src_idx, dst_idx)
PY
```

Count the number of edges like this (example output: 43080809):

```
num_L=$(grep -c '^L[[:space:]]' hprc-v1.0-pggb.gfa)
num_C=$(grep -c '^C[[:space:]]' hprc-v1.0-pggb.gfa)
num_P=$(
  grep '^P[[:space:]]' hprc-v1.0-pggb.gfa \
    | awk '{
        n=split($3, segs, ",");
        total += (n-1);
      }
      END { print total }
    '
)
echo $((num_L + num_C + num_P))
```

This should match the number of 8-byte records in the edge list:

```
size_in_bytes=$(wc -c < edge_list.gam) && echo $((size_in_bytes / 8))
```

## Project Layout

- `src/`: Rust source code implementing conversion, extraction, visualization, and analysis modules.
- `tests/`: Integration tests for key workflows.
- `scripts/`: Utility scripts for benchmarking, visualization, and data preparation.
- `audio/`, `visual/`: Example assets produced by Graphome tooling.

## Development

- Format the codebase before committing:

  ```bash
  cargo fmt
  ```

- Run the test suite to verify changes:

  ```bash
  cargo test
  ```

Contributions via issues and pull requests are welcome! Feel free to propose new analysis pipelines, visualization improvements, or optimizations for large graph genomes.
