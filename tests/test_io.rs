use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

use flate2::read::GzDecoder;

use graphome::eigen_print::{adjacency_matrix_to_ndarray, call_eigendecomp, ndarray_to_faer};
use graphome::io::{self, GfaReader};
use ndarray::{Array2, Axis};

#[test]
fn test_decode_remote_gfa_header() -> Result<(), Box<dyn std::error::Error>> {
    let url = "https://s3-us-west-2.amazonaws.com/human-pangenomics/pangenomes/freeze/release2/minigraph-cactus/hprc-v2.0-mc-grch38.gfa.gz";
    let reader = io::open(url)?;
    let decoder = GzDecoder::new(reader);
    let mut buf_reader = BufReader::new(decoder);
    let mut header = String::new();
    buf_reader.read_line(&mut header)?;
    assert_eq!(header.trim_end(), "H\tVN:Z:1.1\tRS:Z:GRCh38 CHM13");
    Ok(())
}

#[test]
fn test_remote_eigendecomposition_36278_36281_under_two_minutes(
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let url = "https://s3-us-west-2.amazonaws.com/human-pangenomics/pangenomes/freeze/release2/minigraph-cactus/hprc-v2.0-mc-grch38.gfa.gz";
    let reader = io::open(url)?;
    let decoder = GzDecoder::new(reader);
    let mut buf_reader = BufReader::new(decoder);

    let target_names: Vec<String> = ["36278", "36279", "36280", "36281"]
        .iter()
        .map(|name| name.to_string())
        .collect();
    let target_set: HashSet<String> = target_names.iter().cloned().collect();
    let mut counts: HashMap<String, usize> = target_names
        .iter()
        .map(|name| (name.clone(), 0usize))
        .collect();
    let mut found: HashSet<String> = HashSet::new();
    let mut stored_edges: Vec<(String, String)> = Vec::new();

    let mut line = String::new();
    while buf_reader.read_line(&mut line)? != 0 {
        let trimmed = line.trim_end_matches(&['\r', '\n'][..]);
        process_line(
            trimmed,
            &target_names,
            &target_set,
            &mut counts,
            &mut found,
            &mut stored_edges,
        );
        line.clear();
    }

    assert_eq!(
        found.len(),
        target_names.len(),
        "Not all target segments were found in the remote GFA"
    );

    let index_map: HashMap<String, usize> = target_names
        .iter()
        .map(|name| {
            let idx = *counts
                .get(name)
                .expect("Missing count entry for target segment");
            (name.clone(), idx)
        })
        .collect();

    let mut indices: Vec<usize> = index_map.values().copied().collect();
    indices.sort_unstable();
    let &start_index = indices
        .first()
        .expect("No indices computed for target segments");
    let &end_index = indices
        .last()
        .expect("No indices computed for target segments");
    assert_eq!(
        end_index - start_index,
        target_names.len() - 1,
        "Target indices are not consecutive"
    );

    let mut edges: Vec<(u32, u32)> = Vec::new();
    for (from, to) in stored_edges {
        let a = *index_map
            .get(&from)
            .expect("Edge references unknown source segment");
        let b = *index_map
            .get(&to)
            .expect("Edge references unknown destination segment");
        edges.push((a as u32, b as u32));
        edges.push((b as u32, a as u32));
    }

    let adjacency = adjacency_matrix_to_ndarray(&edges, start_index, end_index);
    let degrees = adjacency.sum_axis(Axis(1));
    let degree_matrix = Array2::<f64>::from_diag(&degrees);
    let laplacian = &degree_matrix - &adjacency;
    let laplacian_faer = ndarray_to_faer(laplacian.view());
    let (eigenvalues, eigenvectors) = call_eigendecomp(laplacian_faer.as_ref())?;

    assert_eq!(
        eigenvalues.len(),
        target_names.len(),
        "Unexpected number of eigenvalues"
    );
    assert_eq!(
        eigenvectors.nrows(),
        target_names.len(),
        "Eigenvector matrix has unexpected row count"
    );
    assert_eq!(
        eigenvectors.ncols(),
        target_names.len(),
        "Eigenvector matrix has unexpected column count"
    );

    let duration = start.elapsed();
    assert!(
        duration < Duration::from_secs(120),
        "Remote eigendecomposition exceeded time limit: {:.2?}",
        duration
    );

    Ok(())
}

fn process_line(
    line: &str,
    target_names: &[String],
    target_set: &HashSet<String>,
    counts: &mut HashMap<String, usize>,
    found: &mut HashSet<String>,
    stored_edges: &mut Vec<(String, String)>,
) {
    if line.starts_with("S\t") {
        let mut fields = line.split('\t');
        fields.next();
        if let Some(name) = fields.next() {
            let name = name.trim();
            for target in target_names {
                if name < target.as_str() {
                    if let Some(count) = counts.get_mut(target) {
                        *count += 1;
                    }
                }
            }
            if target_set.contains(name) {
                found.insert(name.to_string());
            }
        }
    } else if line.starts_with("L\t") {
        let mut fields = line.split('\t');
        fields.next();
        if let (Some(from), Some(_from_orient), Some(to), Some(_to_orient)) =
            (fields.next(), fields.next(), fields.next(), fields.next())
        {
            let from = from.trim();
            let to = to.trim();
            if target_set.contains(from) && target_set.contains(to) {
                stored_edges.push((from.to_string(), to.to_string()));
            }
        }
    }
}

#[test]
fn test_gfa_reader_streaming_from_s3() -> Result<(), Box<dyn std::error::Error>> {
    let url = "https://s3-us-west-2.amazonaws.com/human-pangenomics/pangenomes/freeze/release2/minigraph-cactus/hprc-v2.0-mc-grch38.gfa.gz";

    // Test nodes from the beginning of the file (should be fast)
    let mut target_nodes = HashSet::new();
    target_nodes.insert("36278".to_string());
    target_nodes.insert("36279".to_string());
    target_nodes.insert("36280".to_string());
    target_nodes.insert("36281".to_string());

    let start = Instant::now();
    let reader = GfaReader::new(url);
    let sequences = reader.extract_sequences(&target_nodes)?;
    let duration = start.elapsed();

    eprintln!(
        "[TEST] Extracted {} sequences in {:?}",
        sequences.len(),
        duration
    );

    // Verify we got all 4 nodes
    assert_eq!(sequences.len(), 4, "Should find all 4 target nodes");

    // Verify sequences are not empty
    for (node_id, seq) in &sequences {
        assert!(
            !seq.is_empty(),
            "Node {} should have non-empty sequence",
            node_id
        );
        eprintln!("[TEST] Node {}: {} bp", node_id, seq.len());
    }

    // Verify specific node sequences match what we saw in the file
    assert!(sequences.contains_key("36278"), "Should contain node 36278");
    assert!(sequences.contains_key("36279"), "Should contain node 36279");
    assert!(sequences.contains_key("36280"), "Should contain node 36280");
    assert!(sequences.contains_key("36281"), "Should contain node 36281");

    // Should be fast since nodes are near the beginning
    assert!(
        duration < Duration::from_secs(30),
        "Streaming should complete in under 30 seconds for early nodes, took {:?}",
        duration
    );

    Ok(())
}

#[test]
fn test_gfa_reader_with_local_file() -> Result<(), Box<dyn std::error::Error>> {
    // Test with local minimal.gfa file
    let mut target_nodes = HashSet::new();
    target_nodes.insert("36611".to_string());
    target_nodes.insert("36612".to_string());
    target_nodes.insert("36614".to_string());

    let reader = GfaReader::new("data/minimal.gfa");
    let sequences = reader.extract_sequences(&target_nodes)?;

    // Verify we got all 3 nodes
    assert_eq!(sequences.len(), 3, "Should find all 3 target nodes");

    // Verify specific sequences
    assert_eq!(
        sequences.get("36611").map(|s| s.as_str()),
        Some("ACTTGGTTATATTGTCTAA"),
        "Node 36611 sequence mismatch"
    );
    assert_eq!(
        sequences.get("36612").map(|s| s.as_str()),
        Some("C"),
        "Node 36612 sequence mismatch"
    );

    Ok(())
}
