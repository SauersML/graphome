// src/viz.rs

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufWriter, Write, BufRead};
use std::path::Path;
use image::GenericImageView;

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


    
    // Possibly build or reuse .gam adjacency
    use std::path::PathBuf;
    use crate::convert::convert_gfa_to_edge_list;
    use crate::extract::load_adjacency_matrix;
    
    let gfa_pathbase = Path::new(gfa_path);
    let mut gam_path = gfa_pathbase.with_extension("gam");
    
    // If no .gam exists for this GFA, run convert_gfa_to_edge_list
    if !gam_path.exists() {
        eprintln!("[viz] No .gam found at {:?}. Converting GFA -> .gam...", gam_path);
        convert_gfa_to_edge_list(gfa_pathbase, &gam_path)?;
    } else {
        eprintln!("[viz] Using cached adjacency file {:?}", gam_path);
    }
    
    // Interpret start_node..end_node as indices
    let start_idx = start_node.parse::<usize>()
        .map_err(|_| format!("start-node must be an integer, got {}", start_node))?;
    let end_idx = end_node.parse::<usize>()
        .map_err(|_| format!("end-node must be an integer, got {}", end_node))?;
    if end_idx < start_idx {
        return Err(format!("end-node < start-node: {} < {}", end_idx, start_idx).into());
    }
    let node_count = end_idx - start_idx + 1;
    if node_count > 5000 {
        return Err(format!(
            "Refusing to force-layout {} nodes. Please narrow your range.",
            node_count
        ).into());
    }
    eprintln!("[viz] Building subgraph for node indices [{start_idx}..{end_idx}], total {} nodes", node_count);
    
    // Load edges in that index range
    let edges_vec = load_adjacency_matrix(&gam_path, start_idx, end_idx)?;
    // edges_vec is Vec<(u32,u32)>
    
    // Build a minimal structure to store node “length”, adjacency, etc.
    // Length=1 for each node (just for a radius in the draw)... for now
    let mut adjacency = vec![Vec::new(); node_count]; // adjacency[i] = list of neighbors i->?
    for &(f, t) in &edges_vec {
        // subtract start_idx so i in [0..(node_count-1)]
        let i = (f as usize) - start_idx;
        let j = (t as usize) - start_idx;
        adjacency[i].push(j);
        adjacency[j].push(i);
    }
    
    // We'll store node_data in an array
    let mut node_data = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        node_data.push( NodeData {
            length: 1, // placeholder... for now
            neighbors: HashSet::new() // we won't use string-based adjacency, so unused... for now
        });
    }
    
    // Build a numeric edges list for the force layout
    let mut edges = Vec::new();
    for (i, nbrs) in adjacency.iter().enumerate() {
        // push (i,j) only if i<j to avoid duplicates
        for &j in nbrs {
            if i < j {
                edges.push((i,j));
            }
        }
    }
    
    eprintln!("[viz] Subgraph has {} edges after dedup.", edges.len());





    

    // Force-based layout
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
        }
    }
    eprintln!("[viz] Force layout done for {} nodes, {} edges.", node_count, edges.len());

    // Re-center and re-scale positions to fill [0..1] with a small margin
    let mut minx = f32::MAX;
    let mut miny = f32::MAX;
    let mut maxx = f32::MIN;
    let mut maxy = f32::MIN;
    
    for &(x, y) in &positions {
        if x < minx { minx = x; }
        if x > maxx { maxx = x; }
        if y < miny { miny = y; }
        if y > maxy { maxy = y; }
    }
    
    let rangex = (maxx - minx).max(0.00001);
    let rangey = (maxy - miny).max(0.00001);
    
    for (x, y) in &mut positions {
        // Scale into 0..1
        *x = (*x - minx) / rangex;
        *y = (*y - miny) / rangey;
        // Add 5% margin on each side
        *x = 0.05 + *x * 0.90;
        *y = 0.05 + *y * 0.90;
    }

    // Draw the result in a 2D image. We'll define a max width/height.
    let width = 1500u16;
    let height = 800u16;
    let mut buffer = vec![0u8; (width as usize)*(height as usize)*3]; // black background

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
            (80, 80, 80) // darker gray edges for black background
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
        
        // Draw a radial glow with additive blending
        draw_radial_glow(&mut buffer, width, height, cx, cy, radius + 10, (b, g, r));
        
        // Now draw the main circle on top
        draw_filled_circle_bgr(&mut buffer, width, height, cx, cy, radius, (b, g, r));
    }

    // Write out TGA
    write_uncompressed_tga(width, height, &buffer, output_tga)?;

    eprintln!(
        "[viz] Wrote subgraph layout with {} nodes to {}. Image is {}x{}.",
        node_count, output_tga, width, height
    );
    
    // ---- AUTO-DISPLAY in terminal ----
    eprintln!("[viz] Displaying {} in the terminal...", output_tga);
    use termimage::ops;  // re-use from display.rs crate
    use std::io::Write;
    let path_info = (String::new(), std::path::PathBuf::from(output_tga));
    let guessed_fmt = ops::guess_format(&path_info)
        .map_err(|e| format!("Termimage guess_format error: {:?}", e))?;
    let img = ops::load_image(&path_info, guessed_fmt)
        .map_err(|e| format!("Termimage load_image error: {:?}", e))?;

    
    // Get the original size from the TGA itself
    let original_size = (img.width(), img.height());
    
    // Terminal size
    let term_size = (600, 400);
    let resized_size = ops::image_resized_size(original_size, term_size, true);
    
    // Resize & print
    let resized = ops::resize_image(&img, resized_size);
    ops::write_ansi_truecolor(&mut std::io::stdout(), &resized);
    std::io::stdout().flush()?;
    // -----------------------------------
    
    Ok(())
}

/// Convert HSL to RGB, each in [0..255]. 
/// h in [0..360], s,l in [0..1].
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let c = (1.0 - (2.0*l - 1.0).abs()) * s;
    let hh = h / 60.0;
    let x = c * (1.0 - (hh % 2.0 - 1.0).abs());

    let (mut r, mut g, mut b) = match hh as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    let m = l - c/2.0;
    r += m; 
    g += m; 
    b += m;

    ((r*255.0) as u8, (g*255.0) as u8, (b*255.0) as u8)
}

/// Assign a bright, high‐saturation color based on a hash of the node ID.
/// Then return it in BGR order for the TGA image.
fn color_from_node(node_id: &str, length: usize) -> (u8, u8, u8) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    node_id.hash(&mut hasher);
    let h64 = hasher.finish();

    // Hue in [0..360]
    let hue = (h64 % 360) as f32;
    // High saturation and moderate lightness
    let saturation = 0.9;
    let lightness = 0.55;

    // Convert HSL -> RGB
    let (r, g, b) = hsl_to_rgb(hue, saturation, lightness);

    // Return in BGR order for TGA
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

/// Draw an additive radial glow extending out to `glow_radius`.
/// Uses a falloff so that brightness fades as distance increases.
fn draw_radial_glow(
    buffer: &mut [u8],
    width: u16,
    height: u16,
    cx: i32,
    cy: i32,
    glow_radius: i32,
    color: (u8, u8, u8)
) {
    let (b, g, r) = color;
    let w = width as i32;
    let h = height as i32;
    let rr = glow_radius as f32;

    for dy in -glow_radius..=glow_radius {
        let yy = cy + dy;
        if yy < 0 || yy >= h {
            continue;
        }
        for dx in -glow_radius..=glow_radius {
            let xx = cx + dx;
            if xx < 0 || xx >= w {
                continue;
            }
            // Distance from center
            let dist2 = (dx*dx + dy*dy) as f32;
            if dist2 <= rr*rr {
                let d = dist2.sqrt();
                // alpha goes from 1.0 at center to 0.0 at outer edge
                let mut alpha = 1.0 - (d / rr);
                // Optionally square it for a smoother fade:
                alpha *= alpha;

                let idx = (yy as usize * w as usize + xx as usize) * 3;
                // Read the old pixel (BGR)
                let old_b = buffer[idx] as f32;
                let old_g = buffer[idx + 1] as f32;
                let old_r = buffer[idx + 2] as f32;

                // Additively blend the glow color
                let new_b = (old_b + b as f32 * alpha).min(255.0);
                let new_g = (old_g + g as f32 * alpha).min(255.0);
                let new_r = (old_r + r as f32 * alpha).min(255.0);

                buffer[idx]   = new_b as u8;
                buffer[idx+1] = new_g as u8;
                buffer[idx+2] = new_r as u8;
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
