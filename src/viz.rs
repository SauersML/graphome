// src/viz.rs

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufWriter, Write};

use crate::map::{GlobalData, parse_gfa_memmap, NodeInfo};

/// Renders a visualization (uncompressed TGA in BGR24) of nodes in the GFA whose
/// IDs fall in the inclusive lexicographical range [start_node .. end_node].
///
/// - `gfa_path`: path to the GFA file.
/// - `start_node`: the lowest node ID (string comparison) to include.
/// - `end_node`: the highest node ID (string comparison) to include.
/// - `output_tga`: path to the TGA file to write.
///
/// The image:
///   - Is `width = 600` pixels wide (arbitrary; can be adjusted).
///   - Has one horizontal row per selected node.  (Hence, image height = #nodes.)
///   - Each row is colored deterministically by a hash of node ID.
///   - The color brightness is modulated by the node’s length (if known).
///
/// The TGA is written uncompressed, 24 bits/pixel, BGR order. 
/// If no nodes match, an error is returned instead of writing an empty image.
pub fn run_viz(
    gfa_path: &str,
    start_node: &str,
    end_node: &str,
    output_tga: &str
) -> Result<(), Box<dyn Error>> {
    eprintln!("[viz] Loading GFA from: {}", gfa_path);

    // Prepare a fresh data container from map.rs
    let mut global = GlobalData {
        node_map: HashMap::new(),
        path_map: HashMap::new(),
        node_to_paths: HashMap::new(),
        alignment_by_path: HashMap::new(),
        ref_trees: HashMap::new(),
    };

    // Call parse from map.rs
    parse_gfa_memmap(gfa_path, &mut global);
    let total_nodes = global.node_map.len();
    eprintln!("[viz] Parsed {} nodes from GFA.", total_nodes);

    // If the GFA is empty or no node lines found, bail out
    if total_nodes == 0 {
        return Err(format!(
            "GFA contains no nodes; cannot produce visualization."
        ).into());
    }

    // Collect all node IDs in a vector, along with their lengths
    // We'll rely on NodeInfo.length if present, or 0 if unknown.
    // We do not assume numeric IDs. We do a standard string range check.
    let mut nodes: Vec<(String, usize)> = global
        .node_map
        .iter()
        .map(|(nid, info)| (nid.clone(), info.length))
        .collect();

    // Filter to only those in [start_node..end_node] (string comparison).
    nodes.retain(|(nid, _len)| {
        nid >= start_node && nid <= end_node
    });

    // Sort lexicographically by node ID
    nodes.sort_by(|a, b| a.0.cmp(&b.0));

    // If no nodes remain after filtering, return an error
    if nodes.is_empty() {
        return Err(format!(
            "No nodes found in range [{} .. {}]. Nothing to visualize.",
            start_node, end_node
        ).into());
    }

    // Decide on image dimensions:
    //   width: fixed value (arbitrary)
    //   height: number of selected nodes
    let width = 600u16;
    let height = nodes.len() as u16;

    // Each row = one node => row i in [0..height-1].
    // We'll store pixels in row-major order, BGR, bottom->top as TGA requires.
    // So row 0 in memory is the bottom row of the final image.
    // We can either invert the iteration or write a top-down approach but
    // reorder the buffer at the end. Here, we’ll fill from the bottom up directly.
    let mut buffer = vec![0u8; width as usize * height as usize * 3];

    // Fill the buffer with a color for each node
    for (i, (node_id, length)) in nodes.iter().enumerate() {
        // row i means the bottom row is i in the TGA sense
        let row_index = i as u16;

        // Generate a stable color from the node ID (hash).
        // We also incorporate 'length' to vary brightness.
        let (b, g, r) = color_from_node(node_id, *length);

        // Fill all pixels in this row with that color
        let row_offset = (row_index as usize) * (width as usize) * 3;
        for px_i in 0..width {
            let px_offset = row_offset + (px_i as usize) * 3;
            buffer[px_offset + 0] = b;
            buffer[px_offset + 1] = g;
            buffer[px_offset + 2] = r;
        }
    }

    // Write the TGA
    write_uncompressed_tga(width, height, &buffer, output_tga)?;

    eprintln!(
        "[viz] Wrote visualization of {} nodes to {} ({}x{} TGA)",
        nodes.len(),
        output_tga,
        width,
        height
    );

    Ok(())
}

/// Produce a stable BGR color for the given node ID, modulated by the node length.
/// This is one possible strategy; you can adapt it.
fn color_from_node(node_id: &str, length: usize) -> (u8, u8, u8) {
    use std::collections::hash_map::DefaultHasher;
    // Compute a 64-bit hash of the node_id
    let mut hasher = DefaultHasher::new();
    node_id.hash(&mut hasher);
    let h = hasher.finish(); // 64-bit

    // We'll break the hash into 3 bytes for B, G, R
    // Then incorporate length to shift brightness
    let hb = (h & 0xFF) as u8;
    let hg = ((h >> 8) & 0xFF) as u8;
    let hr = ((h >> 16) & 0xFF) as u8;

    // Some factor from length to modulate brightness. 
    // We'll do a simple clamp to [0..255].
    let length_factor = (length % 256) as u8;

    // Combine them: 
    //   b = (hb + length_factor) mod 256
    //   g = (hg + length_factor/2) mod 256
    //   r = hr
    let b = hb.wrapping_add(length_factor);
    let g = hg.wrapping_add(length_factor / 2);
    let r = hr;

    (b, g, r)
}

/// Writes an uncompressed TGA file in 24‐bit BGR.
///
/// # Arguments
///
/// - `width`, `height`: image dimensions
/// - `buffer`: must be `width * height * 3` bytes, in **BGR** order, bottom row first
/// - `output_path`: filesystem path to write
///
/// # Errors
/// Returns `Err(...)` if writing fails.
fn write_uncompressed_tga(
    width: u16,
    height: u16,
    buffer: &[u8],
    output_path: &str
) -> Result<(), Box<dyn Error>> {
    if buffer.len() != (width as usize) * (height as usize) * 3 {
        return Err(format!(
            "Buffer size {} does not match TGA size {} x {} x 3",
            buffer.len(),
            width,
            height
        ).into());
    }

    // 18-byte TGA header for uncompressed 24-bit:
    //   - ID length       : 0
    //   - Color map type  : 0
    //   - Image type      : 2 (uncompressed true color)
    //   - Color map spec  : 5 zero bytes
    //   - X origin        : 0 (2 bytes)
    //   - Y origin        : 0 (2 bytes)
    //   - Width           : 2 bytes (LE)
    //   - Height          : 2 bytes (LE)
    //   - Bits per pixel  : 24
    //   - Image descriptor: 0 (no alpha bits, origin at lower left)

    let mut header = [0u8; 18];
    header[2] = 2; // uncompressed RGB
    header[12] = (width & 0xFF) as u8;
    header[13] = ((width >> 8) & 0xFF) as u8;
    header[14] = (height & 0xFF) as u8;
    header[15] = ((height >> 8) & 0xFF) as u8;
    header[16] = 24; // bits per pixel

    let f = File::create(output_path)?;
    let mut writer = BufWriter::new(f);
    writer.write_all(&header)?;

    // Write the pixel data
    writer.write_all(buffer)?;

    // TGA footer can be omitted for basic usage.
    writer.flush()?;
    Ok(())
}
