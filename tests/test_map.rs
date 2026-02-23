use gbwt::GBZ;
use graphome::map::{coord_to_nodes, node_to_coords, validate_gfa_for_gbwt};
use reqwest::blocking::{Client, Response};
use simple_sds::serialize;
use std::fs;
use std::io;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

static TEST_GBZ_PATH: OnceLock<Result<String, String>> = OnceLock::new();
static CHM13_GBZ_PATH: OnceLock<Result<PathBuf, String>> = OnceLock::new();

const CHM13_GBZ_URL: &str = "https://s3-us-west-2.amazonaws.com/human-pangenomics/pangenomes/freeze/release2/minigraph-cactus/hprc-v2.0-mc-chm13.gbz";
const CHM13_GBZ_NAME: &str = "hprc-v2.0-mc-chm13.gbz";

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

fn get_hprc_chm13_gbz_path() -> Result<&'static Path, Box<dyn std::error::Error>> {
    match CHM13_GBZ_PATH.get_or_init(resolve_or_download_hprc_chm13_gbz) {
        Ok(path) => Ok(path.as_path()),
        Err(err) => Err(io::Error::other(err.clone()).into()),
    }
}

fn resolve_or_download_hprc_chm13_gbz() -> Result<PathBuf, String> {
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let primary = project_root.join("data/hprc").join(CHM13_GBZ_NAME);
    let fallback = project_root.join("data").join(CHM13_GBZ_NAME);
    if primary.exists() {
        return Ok(primary);
    }
    if fallback.exists() {
        return Ok(fallback);
    }

    fs::create_dir_all(primary.parent().ok_or("invalid CHM13 cache path")?)
        .map_err(|e| format!("failed to create CHM13 cache directory: {e}"))?;

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(60 * 60))
        .build()
        .map_err(|e| format!("failed to build HTTP client: {e}"))?;
    let mut response: Response = client
        .get(CHM13_GBZ_URL)
        .send()
        .and_then(Response::error_for_status)
        .map_err(|e| format!("failed to download CHM13 GBZ: {e}"))?;

    let tmp = primary.with_extension("gbz.part");
    let mut out = fs::File::create(&tmp)
        .map_err(|e| format!("failed to create temporary file {:?}: {}", tmp, e))?;
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let read = response
            .read(&mut buf)
            .map_err(|e| format!("download read failed: {e}"))?;
        if read == 0 {
            break;
        }
        out.write_all(&buf[..read])
            .map_err(|e| format!("download write failed: {e}"))?;
    }
    out.flush()
        .map_err(|e| format!("flush failed for {:?}: {}", tmp, e))?;
    fs::rename(&tmp, &primary)
        .map_err(|e| format!("failed to move {:?} to {:?}: {}", tmp, primary, e))?;
    Ok(primary)
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
    let binary = env!("CARGO_BIN_EXE_graphome");
    let gbz_path = get_hprc_chm13_gbz_path()?;
    let output = Command::new(binary)
        .args([
            "map",
            "--gfa",
            gbz_path.to_str().ok_or("invalid CHM13 GBZ path")?,
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
fn test_coord2node_full_coverage() -> Result<(), Box<dyn std::error::Error>> {
    // This test verifies that coord_to_nodes finds ALL nodes in a region,
    // not just a sparse sample. This was a bug where reference_positions(1000)
    // was used, which only sampled every 1000 bp and missed ~90% of nodes.

    let gbz_path = get_test_gbz_path()?;
    let gbz: GBZ = serialize::load_from(gbz_path)?;

    // Get a coordinate range from the minimal.gfa
    // The paths start at position 0 in the GBZ (not the reference offset)
    // (Note: GBZ extracts just "chr10" from the full path name "grch38#0#chr10")
    let chr = "chr10";
    let start = 0;
    let end = 500;

    eprintln!(
        "\n=== Testing full node coverage for {}:{}-{} ===",
        chr, start, end
    );

    let nodes = coord_to_nodes(&gbz, chr, start, end);
    eprintln!("Found {} nodes in region", nodes.len());

    // Calculate total sequence length that should be extracted
    let mut total_extracted_length = 0;
    let mut paths_seen = std::collections::HashSet::new();
    for node in &nodes {
        let node_seq_len = node.path_off_end - node.path_off_start + 1;
        total_extracted_length += node_seq_len;
        paths_seen.insert(node.path_name.clone());
        eprintln!(
            "  path={} Node {} offsets {}..{} (length {})",
            node.path_name, node.node_id, node.path_off_start, node.path_off_end, node_seq_len
        );
    }

    eprintln!("Unique paths found: {:?}", paths_seen);

    eprintln!(
        "Total extracted sequence length: {} bp",
        total_extracted_length
    );
    eprintln!("Requested region length: {} bp", end - start);

    // We should find at least some nodes
    assert!(
        !nodes.is_empty(),
        "Expected to find at least one node in the region"
    );

    // Verify we're getting full path names with sample#haplotype#contig format
    eprintln!("Checking path name format...");
    for path in &paths_seen {
        let parts: Vec<&str> = path.split('#').collect();
        assert!(
            parts.len() >= 2,
            "Path name '{}' should be in format 'sample#haplotype#contig', but has {} parts",
            path,
            parts.len()
        );
        eprintln!("  ✓ Path '{}' has correct format", path);
    }

    // The minimal.gfa has two paths: grch38#0#chr10 and chm13#0#chr10
    // We should see both paths distinguished
    assert!(
        paths_seen.len() >= 2,
        "Expected to find at least 2 distinct paths (grch38 and chm13), but only found: {:?}",
        paths_seen
    );

    // The extracted length should be close to the requested region length
    // With the old sparse sampling bug, we would only get ~1-2 bp per 1000 bp
    // Now we should get close to the full region (allowing for some graph structure)
    let coverage_ratio = total_extracted_length as f64 / (end - start) as f64;
    eprintln!("Coverage ratio: {:.1}%", coverage_ratio * 100.0);

    // At minimum, we should extract more than 50% of the region PER PATH
    // Since we have 2 paths, total coverage will be ~200%
    // (The old bug would give us < 1%)
    assert!(
        coverage_ratio >= 0.5,
        "Expected to extract at least 50% of region ({} bp), but only got {:.1}% ({} bp). \
         This suggests nodes are being missed (sparse sampling bug).",
        (end - start) / 2,
        coverage_ratio * 100.0,
        total_extracted_length
    );

    Ok(())
}

#[test]
fn integration_eigen_region_hprc() -> Result<(), Box<dyn std::error::Error>> {
    let binary = env!("CARGO_BIN_EXE_graphome");
    let gbz_path = get_hprc_chm13_gbz_path()?;
    let output = Command::new(binary)
        .args([
            "eigen-region",
            "--gfa",
            gbz_path.to_str().ok_or("invalid CHM13 GBZ path")?,
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
    assert!(stdout.contains("Nodes:"), "Output missing node count");
    assert!(stdout.contains("Edges:"), "Output missing edge count");
    assert!(stdout.contains("NGEC:"), "Output missing NGEC score");
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
