# Graphome 💠 🧬
Graphome is a graph analysis program. It is designed with graph genomes in mind, but can work with other graph structures. It can convert genome graph assembly files (GFA) to efficient binary adjacency matrices (GAM), and work with GAM files. It also performs eigendecomposition on Laplacian matrices.

## Overview

Graphome is a high-performance Rust tool that converts Graphical Fragment Assembly (GFA) files into binary adjacency matrices, creating `.gam` ({Genome||Graph} Adjacency Matrix) format files. This conversion allows for efficient graph analysis and manipulation while drastically reducing memory footprint.

### Key Features

- 🚀 **High Performance**: Optimized for processing large-scale genome graphs (tested on 89GB+ GFA files)
- 💾 **Memory Efficient**: Processes data in streams without loading entire graphs into memory
- 📊 **Binary Format**: Compact `.gam` format storing only connectivity information (0s and 1s)
- 🧮 **Matrix Operations**: Efficient adjacency matrix operations for graph analysis

### Statistics

Graphome can calculate various statistics of a graph.

- Node connectivities
- Eigenvalues and eigenvectors of a graph as a whole, or a region of the graph
- Entropy of a graph, or a specific region of the graph
- Clustering of paths in a graph (in progress)
- Mangitude of differences between paths (in progress)
- Path-level convex hull overlap (in progress)

Eigendecomposition can be done in windows. Manifold alignment can be used to align features between windows.



## Installation

Build from source:

```bash
git clone https://github.com/SauersML/graphome
cd graphome
cargo build --release
```

Graphome performs eigendecomposition with [faer](https://crates.io/crates/faer), a pure Rust linear algebra library. No external LAPACK or BLAS dependencies are required.

## Usage

### Basic Conversion

Convert a GFA file to GAM format:

```bash
graphome convert input.gfa output.gam
```

### View Matrix Statistics (tbd)

```bash
graphome stats matrix.gam
```

### Extract Subgraph

```bash
graphome extract --region chr20:1000000-2000000 input.gam output.gam
```

## File Format Specification

### GAM Format (Genome (or Graph) Adjacency Matrix)

The `.gam` format is a binary format representing genome graph connectivity:

- **Header** (16 bytes):
  - Magic number (4 bytes): "GAM\0" (tbd)
  - Version (4 bytes): u32 (tbd)
  - Node count (4 bytes): u32
  - Flags (4 bytes): u32 (tbd)

- **Matrix Data**:
  - Bit-packed adjacency matrix
  - Row-major order (tbd)
  - Each bit represents an edge (1) or no edge (0)

To quicky view a .gam file in human-readable format (with under 1000 nodes), you can use:
```
N=$(for n in $(seq 1 1000); do \
      k=$(( (n +7)/8 )); \
      if [ $((n * k )) -eq $(stat -c%s submatrix.gam) ]; then \
          echo $n; \
          break; \
      fi; \
    done); \
xxd -b -c2 submatrix.gam | \
cut -d ' ' -f2,3 | \
awk '{ 
    byte1 = substr($1,1,8); 
    byte2 = substr($2,1,8); 
    reversed1 = ""; 
    reversed2 = ""; 
    for(i=8;i>=1;i--) { 
        reversed1 = reversed1 substr(byte1,i,1); 
    } 
    for(i=8;i>=1;i--) { 
        reversed2 = reversed2 substr(byte2,i,1); 
    } 
    bits = reversed1 reversed2; 
    row = substr(bits,1,12); 
    for(j=1;j<=12;j++) { 
        printf "%s ", substr(row,j,1); 
    } 
    print "" 
}'; \
echo
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

This should be the same number as the amount of edges in the edge list:
```
size_in_bytes=$(wc -c < edge_list.gam) && echo $((size_in_bytes / 8))
```
