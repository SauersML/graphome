// src/viz.rs

use std::fs::File;
use std::io::{Write, BufWriter};
use std::error::Error;
use std::collections::HashMap;

// Import needed items from map.rs
use crate::map::{
    GlobalData,
    parse_gfa_memmap,
};

/// Main entry point for visualization:
/// - `gfa_path`: path to the GFA file
/// - `node_start`: the lowest node ID (assuming numeric) to include
/// - `node_end`: the highest node ID (assuming numeric) to include
/// - `output_tga`: where to write the uncompressed TGA image
pub fn run_viz(
    gfa_path: &str,
    node_start: u64,
    node_end: u64,
    output_tga: &str
) -> Result<(), Box<dyn Error>> 
{
    eprintln!("[viz] Parsing GFA from: {}", gfa_path);

    let mut global = GlobalData {
        node_map: HashMap::new(),
        path_map: HashMap::new(),
        node_to_paths: HashMap::new(),
        alignment_by_path: HashMap::new(),
        ref_trees: HashMap::new(),
    };

    // Use the memory-mapped parser from map.rs
    parse_gfa_memmap(gfa_path, &mut global);
    eprintln!(
        "[viz] GFA parse done. Total nodes read = {}",
        global.node_map.len()
    );

    // Collect node IDs in the requested range
    let mut selected_nodes = Vec::new();
    for (node_id, info) in &global.node_map {
        // Attempt to parse node_id as u64
        if let Ok(num_id) = node_id.parse::<u64>() {
            if num_id >= node_start && num_id <= node_end {
                selected_nodes.push((num_id, info.length));
            }
        }
    }

    // Sort the selected nodes by numeric ID
    selected_nodes.sort_by_key(|&(id, _)| id);

    if selected_nodes.is_empty() {
        eprintln!(
            "[viz] No nodes found in range {}..{}; nothing to visualize.",
            node_start, node_end
        );
        return Ok(());
    }

    // For this simple example, define an image width
    // and let the height = number of nodes in range.
    let width = 400u16; 
    let height = selected_nodes.len() as u16;

    // We'll build a small pixel buffer for the final TGA.
    // TGA uncompressed 24-bit = BGR for each pixel
    let mut buffer = vec![0u8; (width as usize) * (height as usize) * 3];

    // Fill each row with some color that depends on the node's length
    // We'll do bottom-to-top in TGA format. The bottom row is index 0.
    // The top row is index (height-1).
    // We'll iterate from 0 to height-1, but assign buffer lines
    // so that row 0 is the lowest row in the final TGA.
    for (row_idx, &(node_id, node_len)) in selected_nodes.iter().enumerate() {
        // The row in the final TGA that we fill is row_idx from the bottom
        let row_bottom = row_idx as u16;
        let color = make_color_from_length(node_len);

        // Each row has width pixels, each pixel is 3 bytes (B,G,R).
        let row_offset = (row_bottom as usize) * (width as usize) * 3;

        for x in 0..width {
            let px_offset = row_offset + (x as usize) * 3;

            // color is (b, g, r)
            buffer[px_offset + 0] = color.0; // B
            buffer[px_offset + 1] = color.1; // G
            buffer[px_offset + 2] = color.2; // R
        }
    }

    // Write out TGA (uncompressed, 24-bits, BGR) to output_tga
    write_uncompressed_tga(width, height, &buffer, output_tga)?;

    eprintln!(
        "[viz] Done! Wrote {}x{} TGA image to {}",
        width, height, output_tga
    );
    Ok(())
}

/// A simple function to convert a node length to some color (in B,G,R).
fn make_color_from_length(length: usize) -> (u8, u8, u8) {
    // Let's just do a gradient based on length
    // We'll clamp length to some range for demonstration
    let val = (length % 255) as u8;
    // We'll produce a color in BGR order
    let b = val;        // blue component
    let g = 255 - val;  // green component
    let r = val / 2;    // red component
    (b, g, r)
}

/// Write a 24-bit uncompressed TGA in BGR format.
/// - `width`, `height`: image dimensions
/// - `buffer`: must be `width * height * 3` bytes, in BGR for each pixel
/// - `path`: output filename
fn write_uncompressed_tga(
    width: u16,
    height: u16,
    buffer: &[u8],
    path: &str
) -> Result<(), Box<dyn Error>> 
{
    // TGA header is 18 bytes
    // For an uncompressed truecolor TGA:
    //   Offset 0:  ID length = 0
    //   Offset 1:  Color map type = 0
    //   Offset 2:  Image type = 2 (uncompressed true color)
    //   Offset 3-7:  Color map specification (5 bytes) = 0
    //   Offset 8-9:  X origin (2 bytes) = 0
    //   Offset 10-11:Y origin (2 bytes) = 0
    //   Offset 12-13: width (2 bytes, little-endian)
    //   Offset 14-15: height (2 bytes, little-endian)
    //   Offset 16:  bits per pixel = 24
    //   Offset 17:  image descriptor = 0 (no alpha, origin at lower-left)

    let mut header = [0u8; 18];
    header[2] = 2; // uncompressed truecolor
    // width
    header[12] = (width & 0x00FF) as u8;
    header[13] = ((width & 0xFF00) >> 8) as u8;
    // height
    header[14] = (height & 0x00FF) as u8;
    header[15] = ((height & 0xFF00) >> 8) as u8;
    // bits per pixel
    header[16] = 24; 

    let f = File::create(path)?;
    let mut writer = BufWriter::new(f);

    // Write header
    writer.write_all(&header)?;
    // Write pixel data
    // TGA expects the bottom row first, top row last, so the data in `buffer`
    // should already be arranged that way.
    writer.write_all(buffer)?;

    writer.flush()?;
    Ok(())
}
