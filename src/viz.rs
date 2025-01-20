// src/viz.rs

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufWriter, Write, BufRead};
use std::path::{Path, PathBuf};

use image::GenericImageView;
use hdbscan::{Hdbscan, HdbscanHyperParams};

use crate::convert::convert_gfa_to_edge_list;
use crate::extract::load_adjacency_matrix;
use crate::display::display_tga;
use crate::eigen_print::{adjacency_matrix_to_ndarray, call_eigendecomp};

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
///  1) Parse GFA from gfa_path.
///  2) Keep only nodes N with start_node <= N <= end_node.
///  3) Build adjacency (subgraph).
///  4) Run a spectral layout to get initial (x,y) positions.
///  5) If force_directed is true, apply a force-directed refinement.
///  6) Draw edges and nodes as a 2D image.
///  7) Write an uncompressed TGA (24‐bit BGR).
pub fn run_viz(
    gfa_path: &str,
    start_node: &str,
    end_node: &str,
    output_tga: &str,
    force_directed: bool
) -> Result<(), Box<dyn Error>> {
    eprintln!("[viz] Loading GFA from: {}", gfa_path);

    let gfa_pathbase = Path::new(gfa_path);
    let mut gam_path = gfa_pathbase.with_extension("gam");

    if !gam_path.exists() {
        eprintln!("[viz] No .gam found at {:?}. Converting GFA -> .gam", gam_path);
        convert_gfa_to_edge_list(gfa_pathbase, &gam_path)?;
    } else {
        eprintln!("[viz] Using cached adjacency file {:?}", gam_path);
    }

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

    let edges_vec = load_adjacency_matrix(&gam_path, start_idx, end_idx)?;


    // Build adjacency
    let mut adjacency = vec![Vec::new(); node_count];
    for &(f, t) in &edges_vec {
        let i = (f as usize) - start_idx;
        let j = (t as usize) - start_idx;
        adjacency[i].push(j);
        adjacency[j].push(i);
    }
    
    // Filter out nodes that have no neighbors
    let mut keep_list = Vec::new();
    for i in 0..node_count {
        if !adjacency[i].is_empty() {
            keep_list.push(i);
        }
    }
    if keep_list.is_empty() {
        return Err("No nodes with edges found in the specified range".into());
    }
    
    // Create a mapping from old node indices to new indices
    let new_count = keep_list.len();
    let mut old_to_new = vec![usize::MAX; node_count];
    for (new_i, &old_i) in keep_list.iter().enumerate() {
        old_to_new[old_i] = new_i;
    }
    
    // Rebuild adjacency and node data for only those kept nodes
    #[derive(Debug)]
    struct NodeData {
        length: usize,
        neighbors: std::collections::HashSet<String>,
    }
    let mut node_data = Vec::with_capacity(new_count);
    let mut adjacency_filtered = vec![Vec::new(); new_count];
    for _ in 0..new_count {
        node_data.push(NodeData {
            length: 1,
            neighbors: std::collections::HashSet::new(),
        });
    }
    for &old_i in &keep_list {
        let new_i = old_to_new[old_i];
        for &old_j in &adjacency[old_i] {
            let new_j = old_to_new[old_j];
            adjacency_filtered[new_i].push(new_j);
        }
    }
    
    // Gather edges (in the reduced set)
    let mut edges = Vec::new();
    for (i, nbrs) in adjacency_filtered.iter().enumerate() {
        for &j in nbrs {
            if i < j {
                edges.push((i, j));
            }
        }
    }
    eprintln!(
        "[viz] Subgraph now has {} nodes and {} edges (after hiding no-edge nodes).",
        new_count,
        edges.len()
    );
    
    // Replace the old adjacency with the filtered one, and reset node_count
    let adjacency = adjacency_filtered;
    let node_count = new_count;
    
    let adjacency_nd = adjacency_matrix_to_ndarray(&edges_vec, start_idx, end_idx);
    let size = node_count;
    let mut laplacian = adjacency_nd.clone();

    // Store the degree of each node in an array
    let mut degrees = vec![0u32; size];
    for i in 0..size {
        let mut deg = 0u32;
        for j in 0..size {
            if adjacency_nd[(i, j)] != 0.0 {
                deg += 1;
            }
        }
        degrees[i] = deg;
    }
    
    // Build the normalized Laplacian
    for i in 0..size {
        for j in 0..size {
            if adjacency_nd[(i, j)] != 0.0 && degrees[i] > 0 && degrees[j] > 0 {
                // Off-diagonal entries: -1 / sqrt(deg_i * deg_j)
                laplacian[(i, j)] = -1.0f64 / f64::sqrt((degrees[i] as f64) * (degrees[j] as f64));
            } else {
                laplacian[(i, j)] = 0.0;
            }
        }
        // Diagonal entries: 1.0
        laplacian[(i, i)] = 1.0;
    }

    let (eigvals, eigvecs) = call_eigendecomp(&laplacian)?;
    let mut pairs: Vec<(f64, Vec<f64>)> = eigvals
        .iter()
        .enumerate()
        .map(|(idx, &val)| {
            let column_vec = (0..size)
                .map(|row| eigvecs[(row, idx)])
                .collect::<Vec<_>>();
            (val, column_vec)
        })
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    
    // Skip all zero (or near-zero) eigenvalues
    let near_zero_threshold = 1e-9;
    let nonzero_pairs: Vec<(f64, Vec<f64>)> = pairs
        .iter()
        .filter(|(val, _)| val.abs() > near_zero_threshold)
        .cloned()
        .collect();
    
    // Make sure we have at least two non-zero eigenvalues
    if nonzero_pairs.len() < 2 {
        return Err(format!(
            "Not enough non-zero eigenvalues for a 2D spectral layout. Found only {}.",
            nonzero_pairs.len()
        ).into());
    }
    
    // Build a weighted spectral embedding from ALL nonzero eigenpairs.
    // We weight each eigenvector's columns by 1/sqrt(lambda_i).
    let m = nonzero_pairs.len();
    let mut embedding = vec![vec![0.0_f32; m]; size];
    for (dim, (lambda, vec_col)) in nonzero_pairs.iter().enumerate() {
        let w = 1.0 / (lambda.sqrt().max(1e-12));
        for i in 0..size {
            embedding[i][dim] = vec_col[i] as f32 * w as f32;
        }
    }
    
    // Now cluster with HDBSCAN, letting it decide clusters automatically.
    let clusterer = Hdbscan::default_hyper_params(&embedding);
    let labels = clusterer
        .cluster()
        .map_err(|e| format!("HDBSCAN failed: {:?}", e))?;
    
    // We'll define color_from_cluster below. For positioning, we only need 2D:
    let mut positions = vec![(0.0_f32, 0.0_f32); size];
    if m >= 2 {
        // If at least 2 nonzero eigenpairs, use the first two dims for an initial layout
        for i in 0..size {
            positions[i] = (embedding[i][0], embedding[i][1]);
        }
    } else {
        // If we only have 1 dimension or none, just spread them in a line
        for i in 0..size {
            positions[i] = (i as f32, 0.0);
        }
    }
    

    if force_directed {
        eprintln!("[viz] Running force-directed refinement on initial spectral layout...");
        force_directed_refinement(&mut positions, &edges);
    }

    let mut minx = f32::MAX;
    let mut miny = f32::MAX;
    let mut maxx = f32::MIN;
    let mut maxy = f32::MIN;
    for &(x, y) in &positions {
        if x < minx {
            minx = x;
        }
        if x > maxx {
            maxx = x;
        }
        if y < miny {
            miny = y;
        }
        if y > maxy {
            maxy = y;
        }
    }
    let rangex = (maxx - minx).max(0.00001);
    let rangey = (maxy - miny).max(0.00001);
    for (x, y) in &mut positions {
        *x = (*x - minx) / rangex;
        *y = (*y - miny) / rangey;
        *x = 0.05 + *x * 0.90;
        *y = 0.05 + *y * 0.90;
    }

    let size = termsize::get().unwrap_or(termsize::Size { rows: 24, cols: 80 });
    let width = (size.cols * 8) as u16;
    let height = (size.rows * 8) as u16;
    let mut buffer = vec![0u8; (width as usize) * (height as usize) * 3];
    let sx = |xx: f32| -> i32 { (xx * (width - 1) as f32).round() as i32 };
    let sy = |yy: f32| -> i32 { (yy * (height - 1) as f32).round() as i32 };

    for &(i, j) in &edges {
        let (xi, yi) = positions[i];
        let (xj, yj) = positions[j];
        draw_line_bgr(
            &mut buffer,
            width,
            height,
            sx(xi),
            sy(yi),
            sx(xj),
            sy(yj),
            (80, 80, 80)
        );
    }

    // Render the glow for all nodes first
    for i in 0..node_count {
        let (xf, yf) = positions[i];
        let cx = sx(xf);
        let cy = sy(yf);
        let length = node_data[i].length;
        let cluster_id = labels[i];
        let (b, g, r) = color_from_cluster(cluster_id);
        let radius = 3.max((length as f32).log2().round() as i32).min(20);
        draw_radial_glow(&mut buffer, width, height, cx, cy, radius + 10, (b, g, r));
    }
    
    // Render the filled circles for all nodes after the glows
    for i in 0..node_count {
        let (xf, yf) = positions[i];
        let cx = sx(xf);
        let cy = sy(yf);
        let length = node_data[i].length;
        let cluster_id = labels[i];
        let (b, g, r) = color_from_cluster(cluster_id);
        let radius = 3.max((length as f32).log2().round() as i32).min(20);
        draw_filled_circle_bgr(&mut buffer, width, height, cx, cy, radius, (b, g, r));
    }

    write_uncompressed_tga(width, height, &buffer, output_tga)?;
    eprintln!(
        "[viz] Wrote subgraph layout with {} nodes to {}. Image is {}x{}.",
        node_count,
        output_tga,
        width,
        height
    );

    eprintln!("[viz] Displaying image in terminal...");

    let mut tga_data = Vec::with_capacity(buffer.len() + 18);
    tga_data.extend_from_slice(&[
        0, 0, 2,
        0, 0, 0, 0, 0,
        0, 0, 0, 0,
        (width & 0xFF) as u8,
        ((width >> 8) & 0xFF) as u8,
        (height & 0xFF) as u8,
        ((height >> 8) & 0xFF) as u8,
        24, 0
    ]);
    tga_data.extend_from_slice(&buffer);

    display_tga(&tga_data)?;

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

fn color_from_cluster(cluster_id: i32) -> (u8, u8, u8) {
    // Noise gets a gray color:
    if cluster_id < 0 {
        return (128, 128, 128); // BGR
    }

    // Spread clusters around the hue wheel in a stable manner:
    let hue = (cluster_id as f32 * 37.0) % 360.0;
    let saturation = 0.9;
    let lightness = 0.55;
    let (r, g, b) = hsl_to_rgb(hue, saturation, lightness);
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

/// Draw a filled circle of radius radius at (cx, cy).
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

/// Draw an additive radial glow extending out to glow_radius.
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

fn force_directed_refinement(positions: &mut [(f32, f32)], edges: &[(usize, usize)]) {
    // Basic FR config
    let n = positions.len();
    if n < 2 {
        return;
    }
    let iterations = 50;
    let area = 1.0;
    let k = (area / n as f32).sqrt();
    let mut disp = vec![(0.0_f32, 0.0_f32); n];

    // Collision thresholds
    let edge_overlap_dist = 0.01_f32;   // how close is "too close" for edges
    // Tiny epsilon to avoid divide-by-zero
    let eps = 0.000001_f32;

    // A helper function that returns the squared distance between two points
    fn dist_sq(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
        let dx = bx - ax;
        let dy = by - ay;
        dx*dx + dy*dy
    }

    // A helper to compute the minimum distance between line segment AB and segment CD
    // We'll push them apart if this distance < edge_overlap_dist
    fn segment_segment_min_dist(a: (f32, f32), b: (f32, f32), c: (f32, f32), d: (f32, f32)) -> f32 {
        // Parametric approach adapted from standard geometry
        // Source references: e.g. "Real-Time Collision Detection" by Christer Ericson
        let (ax, ay) = a;
        let (bx, by) = b;
        let (cx, cy) = c;
        let (dx, dy) = d;

        let abx = bx - ax;
        let aby = by - ay;
        let cdx = dx - cx;
        let cdy = dy - cy;

        let ab_dot_ab = abx*abx + aby*aby + 1e-12;  // to avoid zero
        let cd_dot_cd = cdx*cdx + cdy*cdy + 1e-12;  // to avoid zero

        // We will solve for param t on AB, u on CD
        // AB(t) = A + t*(B - A)
        // CD(u) = C + u*(D - C)
        // We'll clamp t,u in [0..1] to stay on the segments
        let mut t = 0.0_f32;
        let mut u = 0.0_f32;

        // The cross terms
        let r_x = ax - cx;
        let r_y = ay - cy;
        let ab_dot_cd = abx*cdx + aby*cdy;
        let r_dot_ab = r_x*abx + r_y*aby;
        let r_dot_cd = r_x*cdx + r_y*cdy;

        let denom = ab_dot_ab*cd_dot_cd - ab_dot_cd*ab_dot_cd;
        if denom.abs() > 1e-12 {
            t = (r_dot_ab*cd_dot_cd - r_dot_cd*ab_dot_cd) / denom;
            u = (r_dot_cd*ab_dot_ab - r_dot_ab*ab_dot_cd) / denom;
        }

        // Clamp t, u
        t = t.clamp(0.0, 1.0);
        u = u.clamp(0.0, 1.0);

        // Compute the actual closest points in that param range
        let closest_ab_x = ax + abx * t;
        let closest_ab_y = ay + aby * t;
        let closest_cd_x = cx + cdx * u;
        let closest_cd_y = cy + cdy * u;

        let dxm = closest_cd_x - closest_ab_x;
        let dym = closest_cd_y - closest_ab_y;
        (dxm*dxm + dym*dym).sqrt()
    }

    for iter in 0..iterations {
        // Reset displacement
        for i in 0..n {
            disp[i] = (0.0, 0.0);
        }

        // Node–node repulsion
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = positions[j].0 - positions[i].0;
                let dy = positions[j].1 - positions[i].1;
                let dist_sqr = dx*dx + dy*dy + eps;
                let dist = dist_sqr.sqrt();
                let rep = k * k / dist;
                let rx = (dx / dist) * rep;
                let ry = (dy / dist) * rep;
                disp[i].0 -= rx;
                disp[i].1 -= ry;
                disp[j].0 += rx;
                disp[j].1 += ry;
            }
        }

        // Edge attraction
        for &(ii, jj) in edges {
            let dx = positions[jj].0 - positions[ii].0;
            let dy = positions[jj].1 - positions[ii].1;
            let dist_sqr = dx*dx + dy*dy + eps;
            let dist = dist_sqr.sqrt();
            let att = (dist * dist) / k;
            let ax = (dx / dist) * att;
            let ay = (dy / dist) * att;
            disp[ii].0 += ax;
            disp[ii].1 += ay;
            disp[jj].0 -= ax;
            disp[jj].1 -= ay;
        }

        // Edge–edge overlap repulsion
        // For each pair of edges, if they come within edge_overlap_dist, push endpoints away
        for i in 0..edges.len() {
            let (a1, a2) = edges[i];
            let seg_a1 = positions[a1];
            let seg_a2 = positions[a2];
            for j in (i + 1)..edges.len() {
                let (b1, b2) = edges[j];
                // If they share a node, skip (because it's the same edge or adjacent edges)
                if a1 == b1 || a1 == b2 || a2 == b1 || a2 == b2 {
                    continue;
                }
                let seg_b1 = positions[b1];
                let seg_b2 = positions[b2];
                let d = segment_segment_min_dist(seg_a1, seg_a2, seg_b1, seg_b2);
                if d < edge_overlap_dist {
                    // We'll push each pair of endpoints away from the midpoint
                    // to reduce overlap
                    let mid_a_x = 0.5 * (seg_a1.0 + seg_a2.0);
                    let mid_a_y = 0.5 * (seg_a1.1 + seg_a2.1);
                    let mid_b_x = 0.5 * (seg_b1.0 + seg_b2.0);
                    let mid_b_y = 0.5 * (seg_b1.1 + seg_b2.1);
                    let dx = mid_b_x - mid_a_x;
                    let dy = mid_b_y - mid_a_y;
                    let dist_sqr = dx*dx + dy*dy + eps;
                    let dist = dist_sqr.sqrt();
                    // A small repulsive force that grows as we get closer
                    let repel = 0.5 * k * (edge_overlap_dist - d).max(0.0) / dist;
                    let rx = dx * repel;
                    let ry = dy * repel;

                    // Push segment A away from B
                    disp[a1].0 -= rx; 
                    disp[a1].1 -= ry;
                    disp[a2].0 -= rx; 
                    disp[a2].1 -= ry;

                    // Push segment B away from A
                    disp[b1].0 += rx; 
                    disp[b1].1 += ry;
                    disp[b2].0 += rx; 
                    disp[b2].1 += ry;
                }
            }
        }

        // Node collision resolution
        let node_collision_dist = 0.01_f32;
        for i in 0..n {
            for j in (i + 1)..n {
                let mut dx = positions[j].0 - positions[i].0;
                let mut dy = positions[j].1 - positions[i].1;
                let mut dist_sqr = dx*dx + dy*dy;
                
                // If they're essentially at the same spot, nudge one slightly so we have a direction
                if dist_sqr < 1e-12 {
                    positions[j].0 += 0.0001 * (j as f32 + 1.0);
                    positions[j].1 += 0.0001 * (i as f32 + 1.0);
                    dx = positions[j].0 - positions[i].0;
                    dy = positions[j].1 - positions[i].1;
                    dist_sqr = dx*dx + dy*dy;
                }
                
                dist_sqr += eps;
                let dist = dist_sqr.sqrt();
                
                if dist < node_collision_dist {
                    let overlap = (node_collision_dist - dist) * 0.5;
                    let nx = dx / dist;
                    let ny = dy / dist;
                    disp[i].0 -= nx * overlap;
                    disp[i].1 -= ny * overlap;
                    disp[j].0 += nx * overlap;
                    disp[j].1 += ny * overlap;
                }
            }
        }

        // Apply displacements with a "temperature" (cooling)
        let temp = 0.1 * (1.0 - iter as f32 / iterations as f32);
        for i in 0..n {
            let (dx, dy) = disp[i];
            let len_sqr = dx*dx + dy*dy + eps;
            let len = len_sqr.sqrt();
            let limit = len.min(temp);
            positions[i].0 += (dx / len) * limit;
            positions[i].1 += (dy / len) * limit;
        }
    }

    // Final pass: rescale all positions into [0..1] after iterations
    let mut minx = f32::MAX;
    let mut maxx = f32::MIN;
    let mut miny = f32::MAX;
    let mut maxy = f32::MIN;
    for (x, y) in positions.iter() {
        if *x < minx { minx = *x; }
        if *x > maxx { maxx = *x; }
        if *y < miny { miny = *y; }
        if *y > maxy { maxy = *y; }
    }
    let rangex = (maxx - minx).max(1e-12);
    let rangey = (maxy - miny).max(1e-12);
    for i in 0..n {
        positions[i].0 = (positions[i].0 - minx) / rangex;
        positions[i].1 = (positions[i].1 - miny) / rangey;
    }
}
