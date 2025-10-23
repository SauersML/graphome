# Graphome

Graphome is a Rust toolchain for turning huge Graphical Fragment Assembly (GFA) or GBZ pangenome graphs into compact binary edge data and for running a collection of spectral and coordinate analyses on top of those data. The utilities are built to stream multi‑gigabyte genomes (validated on inputs over 80 GB) without keeping the whole file in memory, allowing detailed exploration on commodity hardware.

## Highlights

- Streaming conversion from GFA or GBZ graphs into a binary edge list that fits neatly in downstream pipelines.
- Sliding-window Laplacian extraction and NGEC (Normalized Global Eigen-Complexity) summaries to study local spectral structure.
- Coordinate mapping helpers that tie graph nodes back to reference coordinates using GFA+PAF pairs.
- Visualization commands that render static 2D TGA plots, animated GIFs, and synthetic 3D embeddings for presentations.
- Sequence extraction utilities that reconstruct FASTA segments for arbitrary genomic intervals routed through the graph.

## Installation

Graphome builds with stable Rust. Install the toolchain with [rustup](https://www.rust-lang.org/tools/install) if you have not already.

```bash
git clone https://github.com/SauersML/graphome
cd graphome
cargo build --release
```

The compiled binary will be at `target/release/graphome`. All third-party math is handled through pure-Rust crates (for example, [`faer`](https://crates.io/crates/faer)), so no external BLAS/LAPACK installation is required.

## Basic Pipeline

1. Convert a GFA or GBZ into Graphome's edge-list format.
2. Run one or more analysis commands (window extraction, eigenanalysis, mapping, etc.).
3. Inspect the generated CSV/NPY outputs, images, or animated GIFs.

A minimal end-to-end example:

```bash
# 1. Convert the source graph (GFA or GBZ) into an edge list.
graphome convert --input pangenome.gfa --output pangenome.gam

# 2. Extract a 10k-node slice and run Laplacian eigendecomposition.
graphome extract --input pangenome.gam --start-node 100000 --end-node 110000

# 3. Emit overlapping Laplacian windows for later downstream analysis.
graphome extract-windows --input pangenome.gam --start-node 0 --end-node 200000 \
  --window-size 2000 --overlap 500 --output windows

# Follow up with custom eigendecomposition (Python/NumPy, etc.) and
# place the resulting eigenvalues into per-window folders so that
# `graphome analyze-windows` can read them when summarising NGEC.
```

## Command-Line Guide

Run `graphome --help` or `graphome <command> --help` to see every flag. The summaries below capture the current behaviour of each subcommand and show concrete invocations.

### `convert`

Convert a GFA or GBZ graph into a binary edge list where each 8-byte record stores a little-endian pair of `u32` node indices. Undirected edges are written twice (once per direction). The command auto-detects GBZ input and re-uses deterministic node indexing across runs.

```bash
graphome convert --input hprc-v1.0-pggb.gfa --output hprc-v1.0-pggb.gam
# or convert an existing GBZ index
graphome convert --input hprc-v1.0-pggb.gbz --output hprc-v1.0-pggb.gam
```

The generated `.gam` file can be inspected with `python -m struct` snippets or any binary reader; the record count must equal `filesize_bytes / 8`.

### `extract`

Load a node-inclusive range from a `.gam` file, compute the dense adjacency, Laplacian, eigendecomposition, NGEC score, and print ASCII heatmaps. Outputs are placed next to the `.gam` file: `laplacian.csv`, `eigenvalues.csv`, and `eigenvectors.csv`.

```bash
graphome extract --input hprc-v1.0-pggb.gam --start-node 250000 --end-node 252000
```

### `extract-matrices`

Build a Laplacian for a node range and persist it as an `.npy` file for Python-friendly consumption. The `.npy` lives in the directory supplied through `--output`.

```bash
graphome extract-matrices --input hprc-v1.0-pggb.gam \
  --start-node 250000 --end-node 252000 --output matrices/window_250k_252k
# writes matrices/window_250k_252k/laplacian.npy
```

### `extract-windows`

Generate overlapping Laplacian windows across a node span. The current implementation writes one `.npy` file per window named `laplacian_<start>_<end>.npy` directly into the chosen output directory. (Older pipelines produced per-window folders with eigenvalues; see the `analyze-windows` note for how to bridge that gap.) Extraction parallelises via memory-mapped reads.

```bash
graphome extract-windows --input hprc-v1.0-pggb.gam \
  --start-node 0 --end-node 100000 \
  --window-size 5000 --overlap 1000 \
  --output windows
# Produces files like windows/laplacian_000000_005000.npy
```

### `analyze-windows`

Summarise NGEC statistics from previously computed eigenvalues. The command expects an input directory that contains subdirectories named `window_<start>_<end>/` with an `eigenvalues.npy` file inside each. If you generated Laplacians with `extract-windows`, run a follow-up script to compute eigenvalues and arrange them into that layout before invoking this command.

```bash
graphome analyze-windows --input windows_with_eigenvalues
```

### `map`

Bridge between node IDs and reference coordinates by pairing a graph (GFA or GBZ) with a corresponding untangle PAF. The command will produce or re-use a GBZ index internally so repeated lookups are fast.

```bash
# Map a node to every reference interval it spans
graphome map --gfa hprc-v1.0-pggb.gfa --paf hprc-v1.0-pggb.untangle.paf \
  node2coord 123456

# Discover all nodes overlapping a genomic interval (hg38-style coordinates)
graphome map --gfa hprc-v1.0-pggb.gfa --paf hprc-v1.0-pggb.untangle.paf \
  coord2node grch38#chr1:100000-105000
```

Results are printed to stdout, including merged interval summaries and counts of unique mappings. The `--paf` flag may be omitted when the GBZ already embeds the necessary mappings.

### `viz`

Render a subgraph as a coloured TGA image. If a `.gam` file with the same stem as the input graph does not exist, the command calls `convert` automatically and caches the result. Force-directed refinement can be toggled with `--force-directed`.

```bash
graphome viz --gfa hprc-v1.0-pggb.gfa \
  --start-node 250000 --end-node 251000 \
  --output-tga viz/chr1_window.tga \
  --force-directed
```

The renderer refuses to draw more than 5 000 nodes at a time for performance reasons.

### `embed`

Produce a synthetic 3D embedding for a node range and immediately render it as an animated GIF. The current implementation generates random Gaussian points (useful for smoke testing the video pipeline), attempts to display the animation inline, and asynchronously saves it as `graph_<timestamp>.gif` in the working directory.

```bash
graphome embed --input hprc-v1.0-pggb.gam --start-node 100 --end-node 600
# Displays the animation inline and saves graph_<timestamp>.gif in the working directory
```

### `make-sequence`

Extract FASTA sequences corresponding to a coordinate interval. The command queries the GBZ index (building one if needed) and stitches node sequences, falling back to the source GFA when GBZ data are incomplete.

```bash
graphome make-sequence --gfa hprc-v1.0-pggb.gfa --paf hprc-v1.0-pggb.untangle.paf \
  --region grch38#chr1:100000-120000 \
  --sample HG002 --output sequences/
# Produces sequences/grch38_chr1_100000-120000_HG002.fa
```

### `gfa2gbz`

Convert a GFA into a GBZ archive that bundles a GBWT index, node translation tables, and metadata describing samples/paths. The resulting file is written alongside the input (`<graph>.gbz`).

```bash
graphome gfa2gbz --input hprc-v1.0-pggb.gfa
```

## Edge List Format

Graphome writes `.gam` files as a raw stream of edges:

- Each record is 8 bytes: `u32` source node index followed by `u32` destination node index (both little-endian).
- The converter emits both directions of an undirected edge. Counting edges therefore requires dividing the record count by two to obtain the number of unique adjacencies.
- Node indices follow the sorted order of the original `S` records (or the canonical GBZ node numbering when starting from GBZ).

You can quickly verify the edge count with:

```bash
size_in_bytes=$(wc -c < pangenome.gam)
record_count=$((size_in_bytes / 8))
unique_edges=$((record_count / 2))
```

## Repository Layout

- `src/` – Rust implementation of conversion, extraction, mapping, visualisation, and utility modules.
- `scripts/` – Helper scripts for benchmarking or auxiliary analysis.
- `tests/` – Integration-level Rust tests.
- `audio/`, `visual/` – Example artefacts generated by the toolchain.

## Development

- Format the Rust code before committing:

  ```bash
  cargo fmt
  ```

- Run the test suite to ensure regressions are caught:

  ```bash
  cargo test
  ```

Contributions are welcome! Open an issue to discuss ideas for new analyses, optimisations, or visualisations.
