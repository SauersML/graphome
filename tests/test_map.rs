use graphome::map::{node_to_coords, coord_to_nodes, validate_gfa_for_gbwt};
use gbwt::GBZ;
use simple_sds::serialize;
use std::path::Path;
use std::process::Command;

#[test]
fn test_node2coord_and_coord2node() -> Result<(), Box<dyn std::error::Error>> {
    let gfa_path = "data/minimal.gfa";
    
    // Ensure test data exists
    assert!(Path::new(gfa_path).exists(), "Test GFA file not found");
    
    // Validate and fix GFA if needed
    let fixed_gfa = validate_gfa_for_gbwt(gfa_path)?;
    eprintln!("Using GFA: {}", fixed_gfa);
    
    // Build GBZ index using vg
    let gbz_path = format!("{}.gbz", fixed_gfa);
    
    if !Path::new(&gbz_path).exists() {
        eprintln!("Building GBZ index...");
        let output = Command::new("./vg")
            .args(&["gbwt", "-G", &fixed_gfa, "-g", &gbz_path, "--gbz-format"])
            .output()?;
        
        if !output.status.success() {
            return Err(format!(
                "Failed to build GBZ: {}\nStdout: {}",
                String::from_utf8_lossy(&output.stderr),
                String::from_utf8_lossy(&output.stdout)
            ).into());
        }
    }
    
    // Load GBZ
    let gbz: GBZ = serialize::load_from(&gbz_path)?;
    eprintln!("Loaded GBZ index from {}", gbz_path);
    
    // Test node2coord: Pick a node from the minimal.gfa (e.g., node 36611)
    let test_node = 36611;
    eprintln!("\n=== Testing node2coord for node {} ===", test_node);
    
    let coords = node_to_coords(&gbz, test_node);
    eprintln!("Found {} coordinate mappings for node {}", coords.len(), test_node);
    
    for (chr, start, end) in &coords {
        eprintln!("  {}:{}-{}", chr, start, end);
    }
    
    // Verify we got some results
    assert!(!coords.is_empty(), "node2coord should return at least one coordinate mapping");
    
    // Test coord2node: Use the first coordinate we found
    if let Some((chr, start, end)) = coords.first() {
        eprintln!("\n=== Testing coord2node for region {}:{}-{} ===", chr, start, end);
        
        let nodes = coord_to_nodes(&gbz, chr, *start, *end);
        eprintln!("Found {} nodes in region {}:{}-{}", nodes.len(), chr, start, end);
        
        for result in &nodes {
            eprintln!(
                "  path={} node={} orient={} offsets=[{}..{}]",
                result.path_name,
                result.node_id,
                if result.node_orient { '+' } else { '-' },
                result.path_off_start,
                result.path_off_end
            );
        }
        
        // Verify we got results and that our original node is in the results
        assert!(!nodes.is_empty(), "coord2node should return at least one node");
        
        let found_original = nodes.iter().any(|r| {
            r.node_id.parse::<usize>().ok() == Some(test_node)
        });
        
        assert!(
            found_original,
            "coord2node should find the original node {} in the region",
            test_node
        );
    }
    
    // Test with a broader region to ensure we get multiple nodes
    eprintln!("\n=== Testing coord2node with broader region ===");
    if let Some((chr, _, _)) = coords.first() {
        // Use a region that should span multiple nodes (0-100 should cover several nodes)
        let broad_nodes = coord_to_nodes(&gbz, chr, 0, 100);
        eprintln!("Found {} nodes in broader region {}:0-100", broad_nodes.len(), chr);
        
        // Should find at least one node in this range
        assert!(
            !broad_nodes.is_empty(),
            "Broader region should contain at least one node"
        );
    }
    
    Ok(())
}

#[test]
fn test_node2coord_nonexistent_node() -> Result<(), Box<dyn std::error::Error>> {
    let gfa_path = "data/minimal.gfa";
    
    let fixed_gfa = validate_gfa_for_gbwt(gfa_path)?;
    let gbz_path = format!("{}.gbz", fixed_gfa);
    
    if !Path::new(&gbz_path).exists() {
        let output = Command::new("./vg")
            .args(&["gbwt", "-G", &fixed_gfa, "-g", &gbz_path, "--gbz-format"])
            .output()?;
        
        if !output.status.success() {
            return Err("Failed to build GBZ".into());
        }
    }
    
    let gbz: GBZ = serialize::load_from(&gbz_path)?;
    
    // Test with a node ID that doesn't exist
    let nonexistent_node = 999999;
    let coords = node_to_coords(&gbz, nonexistent_node);
    
    // Should return empty results for nonexistent node
    assert!(
        coords.is_empty(),
        "node2coord should return empty results for nonexistent node"
    );
    
    Ok(())
}

#[test]
fn test_coord2node_empty_region() -> Result<(), Box<dyn std::error::Error>> {
    let gfa_path = "data/minimal.gfa";
    
    let fixed_gfa = validate_gfa_for_gbwt(gfa_path)?;
    let gbz_path = format!("{}.gbz", fixed_gfa);
    
    if !Path::new(&gbz_path).exists() {
        let output = Command::new("./vg")
            .args(&["gbwt", "-G", &fixed_gfa, "-g", &gbz_path, "--gbz-format"])
            .output()?;
        
        if !output.status.success() {
            return Err("Failed to build GBZ".into());
        }
    }
    
    let gbz: GBZ = serialize::load_from(&gbz_path)?;
    
    // Test with a region that likely has no nodes
    let nodes = coord_to_nodes(&gbz, "chr99", 1, 10);
    
    // Should return empty results for nonexistent chromosome
    assert!(
        nodes.is_empty(),
        "coord2node should return empty results for nonexistent chromosome"
    );
    
    Ok(())
}
