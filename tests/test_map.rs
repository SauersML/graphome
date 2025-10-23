use gbwt::GBZ;
use graphome::map::{coord_to_nodes, node_to_coords, validate_gfa_for_gbwt};
use simple_sds::serialize;
use std::fs;
use std::io;
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

static TEST_GBZ_PATH: OnceLock<Result<String, String>> = OnceLock::new();

fn get_test_gbz_path() -> Result<&'static str, Box<dyn std::error::Error>> {
    match TEST_GBZ_PATH.get_or_init(build_test_gbz) {
        Ok(path) => Ok(path.as_str()),
        Err(err) => Err(io::Error::new(io::ErrorKind::Other, err.clone()).into()),
    }
}

fn build_test_gbz() -> Result<String, String> {
    let gfa_path = "data/minimal.gfa";

    if !Path::new(gfa_path).exists() {
        return Err(format!("Test GFA file not found at {}", gfa_path));
    }

    let fixed_gfa = validate_gfa_for_gbwt(gfa_path)?;
    let gbz_path = format!("{}.gbz", fixed_gfa);
    let gbz_path_ref = Path::new(&gbz_path);

    if gbz_path_ref.exists() {
        if GBZ::is_gbz(&gbz_path) {
            return Ok(gbz_path);
        }

        fs::remove_file(gbz_path_ref)
            .map_err(|e| format!("Failed to remove corrupt GBZ {}: {}", gbz_path, e))?;
    }

    let tmp_path = format!("{}.tmp", gbz_path);
    let tmp_path_ref = Path::new(&tmp_path);

    if tmp_path_ref.exists() {
        fs::remove_file(tmp_path_ref)
            .map_err(|e| format!("Failed to remove stale temporary GBZ {}: {}", tmp_path, e))?;
    }

    let vg_executable = if Path::new("./vg").exists() {
        "./vg"
    } else {
        "vg"
    };

    let output = Command::new(vg_executable)
        .args(["gbwt", "-G", &fixed_gfa, "-g", &tmp_path, "--gbz-format"])
        .output()
        .map_err(|e| format!("Failed to run {}: {}", vg_executable, e))?;

    if !output.status.success() {
        return Err(format!(
            "Failed to build GBZ: {}\nStdout: {}\nStderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    fs::rename(&tmp_path, &gbz_path)
        .map_err(|e| format!("Failed to move GBZ into place: {}", e))?;

    if !GBZ::is_gbz(&gbz_path) {
        return Err(format!("{} is not a valid GBZ file", gbz_path));
    }

    Ok(gbz_path)
}

#[test]
fn test_node2coord_and_coord2node() -> Result<(), Box<dyn std::error::Error>> {
    let gfa_path = "data/minimal.gfa";

    // Ensure test data exists
    assert!(Path::new(gfa_path).exists(), "Test GFA file not found");

    let gbz_path = get_test_gbz_path()?;
    eprintln!("Using GBZ index: {}", gbz_path);

    // Load GBZ
    let gbz: GBZ = serialize::load_from(gbz_path)?;
    eprintln!("Loaded GBZ index from {}", gbz_path);

    // Test node2coord: Pick a node from the minimal.gfa (e.g., node 36611)
    let test_node = 36611;
    eprintln!("\n=== Testing node2coord for node {} ===", test_node);

    let coords = node_to_coords(&gbz, test_node);
    eprintln!(
        "Found {} coordinate mappings for node {}",
        coords.len(),
        test_node
    );

    for (chr, start, end) in &coords {
        eprintln!("  {}:{}-{}", chr, start, end);
    }

    // Verify we got some results
    assert!(
        !coords.is_empty(),
        "node2coord should return at least one coordinate mapping"
    );

    // Test coord2node: Use the first coordinate we found
    if let Some((chr, start, end)) = coords.first() {
        eprintln!(
            "\n=== Testing coord2node for region {}:{}-{} ===",
            chr, start, end
        );

        let nodes = coord_to_nodes(&gbz, chr, *start, *end);
        eprintln!(
            "Found {} nodes in region {}:{}-{}",
            nodes.len(),
            chr,
            start,
            end
        );

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
        assert!(
            !nodes.is_empty(),
            "coord2node should return at least one node"
        );

        let found_original = nodes
            .iter()
            .any(|r| r.node_id.parse::<usize>().ok() == Some(test_node));

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
        eprintln!(
            "Found {} nodes in broader region {}:0-100",
            broad_nodes.len(),
            chr
        );

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
    let gbz_path = get_test_gbz_path()?;
    let gbz: GBZ = serialize::load_from(gbz_path)?;

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
    let gbz_path = get_test_gbz_path()?;
    let gbz: GBZ = serialize::load_from(gbz_path)?;

    // Test with a region that likely has no nodes
    let nodes = coord_to_nodes(&gbz, "chr99", 1, 10);

    // Should return empty results for nonexistent chromosome
    assert!(
        nodes.is_empty(),
        "coord2node should return empty results for nonexistent chromosome"
    );

    Ok(())
}

#[test]
fn integration_map_coord2node_hprc() -> Result<(), Box<dyn std::error::Error>> {
    let gbz_path = Path::new("data/hprc/hprc-v2.0-mc-grch38.gbz");
    assert!(
        gbz_path.exists(),
        "Expected HPRC GBZ at {}",
        gbz_path.display()
    );

    let binary = env!("CARGO_BIN_EXE_graphome");
    let output = Command::new(binary)
        .args([
            "map",
            "--gfa",
            gbz_path.to_str().expect("Non-UTF8 GBZ path"),
            "coord2node",
            "chr1:103553815-103579534",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "graphome map command failed: {}\nStdout: {}\nStderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.trim().is_empty(),
        "graphome map command produced no output"
    );

    Ok(())
}

#[test]
fn integration_eigen_region_hprc() -> Result<(), Box<dyn std::error::Error>> {
    let gbz_path = Path::new("data/hprc/hprc-v2.0-mc-grch38.gbz");
    assert!(
        gbz_path.exists(),
        "Expected HPRC GBZ at {}",
        gbz_path.display()
    );

    let binary = env!("CARGO_BIN_EXE_graphome");
    let output = Command::new(binary)
        .args([
            "eigen-region",
            "--gfa",
            "s3://human-pangenomics/pangenomes/freeze/release2/minigraph-cactus/hprc-v2.0-mc-grch38.gbz",
            "--region",
            "chr1:103554644-103758692",
            "--viz",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "graphome eigen-region command failed: {}\nStdout: {}\nStderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Verify key output elements
    assert!(
        stdout.contains("EIGENANALYSIS RESULTS"),
        "Output missing eigenanalysis results section"
    );
    assert!(
        stdout.contains("Region: chr1:103554644-103758692"),
        "Output missing region information"
    );
    assert!(
        stdout.contains("Nodes:"),
        "Output missing node count"
    );
    assert!(
        stdout.contains("Edges:"),
        "Output missing edge count"
    );
    assert!(
        stdout.contains("NGEC:"),
        "Output missing NGEC score"
    );
    assert!(
        stdout.contains("Top 10 Eigenvalues"),
        "Output missing eigenvalue list"
    );
    assert!(
        stdout.contains("LAPLACIAN HEATMAP"),
        "Output missing Laplacian heatmap (--viz flag)"
    );
    assert!(
        stdout.contains("EIGENVALUE DISTRIBUTION"),
        "Output missing eigenvalue distribution (--viz flag)"
    );

    // Verify stderr contains progress messages
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Found local copy of remote GBZ") || stderr.contains("Using GBZ"),
        "Missing GBZ loading message"
    );
    assert!(
        stderr.contains("unique nodes in region"),
        "Missing node discovery message"
    );
    assert!(
        stderr.contains("Extracting edges"),
        "Missing edge extraction message"
    );
    assert!(
        stderr.contains("Performing eigendecomposition"),
        "Missing eigendecomposition message"
    );

    Ok(())
}
