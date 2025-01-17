// src/viz.rs

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufWriter, Write, BufRead};
use std::path::Path;

use std::f32::consts::PI;

/// A simple data structure to hold node info: length, adjacency, etc.
#[derive(Debug)]
struct NodeData {
    length: usize,
    // We'll store edges in an adjacency list: node_id -> set of neighbors
    neighbors: HashSet<String>,
}

/// Renders a real 2D graph layout of all nodes in the GFA whose IDs
/// fall in the inclusive lexicographical range [start_node..end_node].
///
/// Steps:
///  1) Parse GFA (S-lines and L-lines) from `gfa_path`.
///  2) Keep only nodes N with `start_node <= N <= end_node`.
///  3) Build adjacency (subgraph).
///  4) Run a force‐directed layout to get (x,y) for each node.
///  5) Draw edges and nodes as a 2D image.
///  6) Write an uncompressed TGA (24‐bit BGR).
pub fn run_viz(
    gfa_path: &str,
    start_node: &str,
    end_node: &str,
    output_tga: &str
) -> Result<(), Box<dyn Error>> 
{
    eprintln!("[viz] Loading GFA from: {}", gfa_path);

    // 1) Parse GFA into a local HashMap: node_id -> NodeData
    let mut node_map = parse_gfa_full(gfa_path)?;

    let total_nodes = node_map.len();
    eprintln!("[viz] Parsed {} total nodes from GFA (S-lines).", total_nodes);

    if total_nodes == 0 {
        return Err("No nodes in GFA; cannot visualize.".into());
    }

    // 2) Filter by [start_node..end_node]
    // We'll gather a set of kept node IDs
    let mut kept_ids = Vec::new();
    for nid in node_map.keys() {
        if nid.as_str() >= start_node && nid.as_str() <= end_node {
            kept_ids.push(nid.clone());
        }
    }
    kept_ids.sort();

    if kept_ids.is_empty() {
        return Err(format!("No nodes found in range [{start_node}..{end_node}].").into());
    }
    eprintln!("[viz] Subgraph has {} nodes after filter.", kept_ids.len());

    // Build a new subgraph: keep adjacency only among these node IDs
    let kept_set: HashSet<_> = kept_ids.iter().cloned().collect();
    // We'll store each node's adjacency in a simpler structure
    let mut subgraph = HashMap::new();
    for k in &kept_ids {
        let full_nd = node_map.get(k).unwrap();
        // Filter neighbors to only those also in kept set
        let mut filtered_neighbors = HashSet::new();
        for nbr in &full_nd.neighbors {
            if kept_set.contains(nbr) {
                filtered_neighbors.insert(nbr.clone());
            }
        }
        subgraph.insert(
            k.clone(),
            NodeData {
                length: full_nd.length,
                neighbors: filtered_neighbors
            }
        );
    }

    // 3) Force-based layout
    // Convert subgraph to a list of node IDs, then node -> index in a vector
    let node_count = subgraph.len();
    if node_count > 5000 {
        // Threshold for "too big to draw"
        return Err(format!(
            "Refusing to force-layout {} nodes. Please narrow your range.",
            node_count
        ).into());
    }

    let node_ids: Vec<_> = subgraph.keys().cloned().collect();
    let mut node_index_map = HashMap::new();
    for (i, nid) in node_ids.iter().enumerate() {
        node_index_map.insert(nid.clone(), i);
    }

    // Build adjacency in numeric form
    let mut edges = Vec::new();
    for (i, nid) in node_ids.iter().enumerate() {
        let nd = &subgraph[nid];
        for nbr in &nd.neighbors {
            let j = *node_index_map.get(nbr).unwrap();
            // We'll only push (i,j) if i<j to avoid duplicates
            if i < j {
                edges.push((i, j));
            }
        }
    }

    // We'll store a 2D (x,y) position for each node in [0..1] range initially
    let mut positions = vec![(0.0_f32, 0.0_f32); node_count];
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for i in 0..node_count {
        // random from 0..1
        let rx = rng.gen::<f32>();
        let ry = rng.gen::<f32>();
        positions[i] = (rx, ry);
    }

    // Simple force-directed layout parameters
    let iterations = 200;
    let repulsion = 0.00005_f32; // tweak
    let attraction = 0.05_f32;   // tweak
    let dt = 0.85_f32;          // "cooling" factor

    for _iter in 0..iterations {
        // We'll accumulate displacement for each node
        let mut disp = vec![(0.0_f32, 0.0_f32); node_count];

        // Repulsive forces (all pairs) => O(n^2) but n might be small
        for i in 0..node_count {
            for j in (i + 1)..node_count {
                let (xi, yi) = positions[i];
                let (xj, yj) = positions[j];
                let dx = xi - xj;
                let dy = yi - yj;
                let dist2 = dx*dx + dy*dy + 0.000001; // avoid /0
                let force = repulsion / dist2;
                // normalized direction
                let len = dist2.sqrt();
                let ux = dx / len;
                let uy = dy / len;
                disp[i].0 += ux * force;
                disp[i].1 += uy * force;
                disp[j].0 -= ux * force;
                disp[j].1 -= uy * force;
            }
        }

        // Attractive forces on edges
        for &(i, j) in &edges {
            let (xi, yi) = positions[i];
            let (xj, yj) = positions[j];
            let dx = xi - xj;
            let dy = yi - yj;
            let dist = (dx*dx + dy*dy + 0.000001).sqrt();
            let force = attraction * (dist - 0.05); // desired edge length ~0.05
            let ux = dx / dist;
            let uy = dy / dist;
            disp[i].0 -= ux * force;
            disp[i].1 -= uy * force;
            disp[j].0 += ux * force;
            disp[j].1 += uy * force;
        }

        // Apply displacement
        for i in 0..node_count {
            let (mut dx, mut dy) = disp[i];
            // limit the max displacement to avoid instability
            let max_disp = 0.01;
            let dd = (dx*dx + dy*dy).sqrt();
            if dd > max_disp {
                dx = dx * (max_disp / dd);
                dy = dy * (max_disp / dd);
            }
            positions[i].0 += dx * dt;
            positions[i].1 += dy * dt;
            // Keep them in [0..1] region
            if positions[i].0 < 0.0 { positions[i].0 = 0.0; }
            if positions[i].1 < 0.0 { positions[i].1 = 0.0; }
            if positions[i].0 > 1.0 { positions[i].0 = 1.0; }
            if positions[i].1 > 1.0 { positions[i].1 = 1.0; }
        }
    }
    eprintln!("[viz] Force layout done for {} nodes, {} edges.", node_count, edges.len());

    // 4) Draw the result in a 2D image. We'll define a max width/height.
    //    Because we might have up to a few thousand nodes, let's do e.g. 2000x2000.
    let width = 2000u16;
    let height = 2000u16;
    let mut buffer = vec![255u8; (width as usize)*(height as usize)*3]; // white background

    // Helper to draw lines and circles
    // We'll scale (0..1) => (0..width-1)
    let sx = |x: f32| -> i32 { (x * (width - 1) as f32).round() as i32 };
    let sy = |y: f32| -> i32 { (y * (height - 1) as f32).round() as i32 };

    // Draw edges first (thin lines)
    for &(i, j) in &edges {
        let (xi, yi) = positions[i];
        let (xj, yj) = positions[j];
        draw_line_bgr(
            &mut buffer, width, height,
            sx(xi), sy(yi), sx(xj), sy(yj),
            (200, 200, 200) // light gray
        );
    }

    // Now draw nodes as circles
    // We'll color them by hashing the node ID, plus we can modulate radius by length
    for (idx, nid) in node_ids.iter().enumerate() {
        let nd = &subgraph[nid];
        let (b, g, r) = color_from_node(nid, nd.length);
        let radius = 3.max((nd.length as f32).log2().round() as i32).min(20);

        let (xf, yf) = positions[idx];
        let cx = sx(xf);
        let cy = sy(yf);
        draw_filled_circle_bgr(&mut buffer, width, height, cx, cy, radius, (b, g, r));
    }

    // 5) Write out TGA
    write_uncompressed_tga(width, height, &buffer, output_tga)?;

    eprintln!("[viz] Wrote subgraph layout with {} nodes to {}. Image is {}x{}.",
        node_count, output_tga, width, height
    );
    Ok(())
}

/// Parse the GFA from disk, collecting node lengths and adjacency (via L lines).
/// This is a custom parser that:
///   - Reads lines in streaming mode (no fancy memmap).
///   - For `S` lines: store node length from the sequence or LN:i: tag.
///   - For `L` lines: store adjacency in both directions (if the line has valid node names).
fn parse_gfa_full(gfa_path: &str) -> Result<HashMap<String, NodeData>, Box<dyn Error>> {
    use std::io::BufReader;
    let f = File::open(gfa_path)?;
    let reader = BufReader::new(f);

    let mut nodes = HashMap::<String, NodeData>::new();

    for line_res in reader.lines() {
        let line = line_res?;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split('\t');
        let rec_type = match parts.next() {
            Some(x) => x,
            None => continue,
        };
        match rec_type {
            "S" => {
                // S <Name> <Sequence> ...
                let name = match parts.next() {
                    Some(n) => n.to_string(),
                    None => continue,
                };
                let seq_or_star = match parts.next() {
                    Some(s) => s,
                    None => continue,
                };
                let mut length = 0usize;
                if seq_or_star == "*" {
                    // parse LN:i: if present
                    // or length=0 if missing
                    for field in parts {
                        if let Some(stripped) = field.strip_prefix("LN:i:") {
                            if let Ok(val) = stripped.parse::<usize>() {
                                length = val;
                                break;
                            }
                        }
                    }
                } else {
                    length = seq_or_star.len();
                }
                // Insert new NodeData if not present
                nodes.entry(name).or_insert_with(|| NodeData {
                    length,
                    neighbors: HashSet::new()
                }).length = length;
            },
            "L" => {
                // L <From> <FOrient> <To> <TOrient> <CIGAR> ...
                // We only care about adjacency from 'From' to 'To'
                let from_name = match parts.next() {
                    Some(x) => x.to_string(),
                    None => continue,
                };
                let _from_ori = parts.next().unwrap_or("+");
                let to_name = match parts.next() {
                    Some(x) => x.to_string(),
                    None => continue,
                };
                let _to_ori = parts.next().unwrap_or("+");
                let _cigar = parts.next().unwrap_or("*");
                
                // Make sure both exist in the map, or create placeholders
                nodes.entry(from_name.clone()).or_insert_with(|| NodeData {
                    length: 0,
                    neighbors: HashSet::new()
                });
                nodes.entry(to_name.clone()).or_insert_with(|| NodeData {
                    length: 0,
                    neighbors: HashSet::new()
                });
                
                // Insert adjacency
                nodes.get_mut(&from_name).unwrap().neighbors.insert(to_name.clone());
                nodes.get_mut(&to_name).unwrap().neighbors.insert(from_name.clone());
            },
            _ => {
                // skip other lines (P, H, C, etc.)
            }
        }
    }

    Ok(nodes)
}

/// A simple function to produce a stable BGR color from node ID + length.
fn color_from_node(node_id: &str, length: usize) -> (u8, u8, u8) {
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    node_id.hash(&mut hasher);
    let h = hasher.finish();
    let hb = (h & 0xFF) as u8;
    let hg = ((h >> 8) & 0xFF) as u8;
    let hr = ((h >> 16) & 0xFF) as u8;
    let factor = (length % 256) as u8;

    let b = hb.wrapping_add(factor);
    let g = hg.wrapping_add(factor / 2);
    let r = hr;
    (b, g, r)
}

/// Draw a line in BGR buffer using a simple Bresenham approach.
fn draw_line_bgr(
    buffer: &mut [u8],
    width: u16,
    height: u16,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: (u8, u8, u8)
) {
    let (b, g, r) = color;
    let w = width as i32;
    let h = height as i32;

    // Bresenham
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -((y1 - y0).abs());
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let (mut x, mut y) = (x0, y0);

    loop {
        if x >= 0 && x < w && y >= 0 && y < h {
            let idx = (y as usize * w as usize + x as usize) * 3;
            if idx+2 < buffer.len() {
                buffer[idx]   = b;
                buffer[idx+1] = g;
                buffer[idx+2] = r;
            }
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2*err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

/// Draw a filled circle of radius `radius` at (cx, cy).
/// We use a naive approach: check all points within bounding box.
fn draw_filled_circle_bgr(
    buffer: &mut [u8],
    width: u16,
    height: u16,
    cx: i32,
    cy: i32,
    radius: i32,
    color: (u8, u8, u8)
) {
    let (b, g, r) = color;
    let w = width as i32;
    let h = height as i32;
    let rr = (radius*radius) as i32;

    for dy in -radius..=radius {
        let yy = cy + dy;
        if yy < 0 || yy >= h {
            continue;
        }
        for dx in -radius..=radius {
            let xx = cx + dx;
            if xx < 0 || xx >= w {
                continue;
            }
            if dx*dx + dy*dy <= rr {
                let idx = (yy as usize * w as usize + xx as usize) * 3;
                if idx+2 < buffer.len() {
                    buffer[idx]   = b;
                    buffer[idx+1] = g;
                    buffer[idx+2] = r;
                }
            }
        }
    }
}

/// Write a 24-bit uncompressed TGA in BGR order.
fn write_uncompressed_tga(
    width: u16,
    height: u16,
    buffer: &[u8],
    path: &str
) -> Result<(), Box<dyn Error>> {
    // Check size
    let expected = (width as usize)*(height as usize)*3;
    if buffer.len() != expected {
        return Err(format!(
            "Buffer length {} != expected {} for TGA ({}x{})",
            buffer.len(), expected, width, height
        ).into());
    }

    // TGA 18-byte header
    let mut header = [0u8; 18];
    header[2] = 2; // uncompressed truecolor
    header[12] = (width & 0xFF) as u8;
    header[13] = ((width >> 8) & 0xFF) as u8;
    header[14] = (height & 0xFF) as u8;
    header[15] = ((height >> 8) & 0xFF) as u8;
    header[16] = 24; // bpp

    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    w.write_all(&header)?;
    w.write_all(buffer)?;
    w.flush()?;

    Ok(())
}
