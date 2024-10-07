# Graphome 🧬

Convert genome graph assembly files (GFA) to efficient binary adjacency matrices (GAM), and work with GAM files.

## Overview

Graphome is a high-performance Rust tool that converts Graphical Fragment Assembly (GFA) files into binary adjacency matrices, creating `.gam` (Genome Adjacency Matrix) format files. This conversion allows for efficient graph analysis and manipulation while drastically reducing memory footprint.

### Key Features

- 🚀 **High Performance**: Optimized for processing large-scale genome graphs (tested on 89GB+ GFA files)
- 💾 **Memory Efficient**: Processes data in streams without loading entire graphs into memory
- 📊 **Binary Format**: Compact `.gam` format storing only connectivity information (0s and 1s)
- 🧮 **Matrix Operations**: Efficient adjacency matrix operations for graph analysis

## Installation

Build from source:

```bash
git clone https://github.com/ScottSauers/graphome
cd graphome
cargo build --release
```

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

### Extract Subgraph (tbd)

```bash
graphome extract --region chr20:1000000-2000000 input.gam output.gam
```

## File Format Specification

### GAM Format (Genome (or Graph) Adjacency Matrix)

The `.gam` format is a binary format representing genome graph connectivity:

- **Header** (16 bytes):
  - Magic number (4 bytes): "GAM\0"
  - Version (4 bytes): u32 (tbd)
  - Node count (4 bytes): u32
  - Flags (4 bytes): u32 (tbd)

- **Matrix Data**:
  - Bit-packed adjacency matrix
  - Row-major order (tbd)
  - Each bit represents an edge (1) or no edge (0)
