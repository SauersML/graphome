use std::fs;
use std::path::PathBuf;
use std::process::Command;

mod common_chm13;
use common_chm13::{chm13_fixture_path_if_present, ensure_chm13_gbz, should_run_large_integration};

/// Test that coordinate-based extraction returns the correct sequence.
///
/// This test:
/// 1. Extracts first 100kb of chr22 for CHM13#0
/// 2. Inspects the FASTA to determine what coordinates were extracted
/// 3. Picks a 5kb region from the string
/// 4. Extracts middle 3kb (omitting 1kb flanks)
/// 5. Derives coordinate range FROM the FASTA filename/header
/// 6. Queries those coordinates in a follow-up command
/// 7. Asserts the 3kb string appears somewhere in the query result
#[test]
fn test_coordinate_extraction_correctness() {
    if chm13_fixture_path_if_present().is_none() && !should_run_large_integration() {
        eprintln!(
            "Skipping coordinate extraction integration test: CHM13 fixture missing and RUN_LARGE_INTEGRATION is not enabled"
        );
        return;
    }

    // Setup paths
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let gbz_path = ensure_chm13_gbz().expect("failed to prepare CHM13 GBZ test fixture");

    let graphome_bin = option_env!("CARGO_BIN_EXE_graphome")
        .map(PathBuf::from)
        .or_else(|| {
            let debug = project_root.join("target/debug/graphome");
            if debug.exists() {
                Some(debug)
            } else {
                None
            }
        })
        .or_else(|| {
            let release = project_root.join("target/release/graphome");
            if release.exists() {
                Some(release)
            } else {
                None
            }
        })
        .unwrap_or_else(|| {
            panic!("graphome binary not found in Cargo bin env, target/debug, or target/release")
        });

    let test_dir = project_root.join("target/test_coordinate_extraction");

    // Clean up old test files to avoid stale artifacts
    if test_dir.exists() {
        fs::remove_dir_all(&test_dir).expect("Failed to remove old test directory");
    }
    fs::create_dir_all(&test_dir).expect("Failed to create test directory");

    let test_sample = "CHM13#0";

    println!("=== Step 1: Extract FIRST 100kb of chr22 for CHM13#0 ===");

    // Extract 100kb of chr22 starting at 20M.
    let full_output = test_dir.join("chr22_100kb");
    let status = Command::new(&graphome_bin)
        .args(&[
            "make-sequence",
            "--gfa",
            gbz_path.to_str().unwrap(),
            "--assembly",
            "chm13",
            "--region",
            "chr22:20000000-20100000",
            "--sample",
            test_sample,
            "--output",
            full_output.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to execute make-sequence for chr22 100kb");

    assert!(status.success(), "Chr22 100kb extraction failed");

    // Find the output FASTA file
    let full_fasta = fs::read_dir(&test_dir)
        .expect("Failed to read test directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("chr22_100kb") && n.ends_with(".fa"))
                .unwrap_or(false)
        })
        .expect("No FASTA file found for chr22 100kb");

    println!("Found chr22 100kb FASTA: {:?}", full_fasta);

    println!("\n=== Step 2: Inspect the FASTA to determine coordinates ===");

    // Parse the filename to extract the actual coordinates
    // Filename format: chr22_100kb_GRCh38#0_GRCh38_0_chr22_START-END.fa
    let filename = full_fasta.file_name().unwrap().to_str().unwrap();
    println!("FASTA filename: {}", filename);

    // Extract coordinates from filename (format: ...chr22_START-END.fa)
    let coords_part = filename
        .split("chr22_")
        .last()
        .unwrap()
        .trim_end_matches(".fa");
    let coords: Vec<&str> = coords_part.split('-').collect();
    let extracted_start: usize = coords[0].parse().expect("Failed to parse start coordinate");
    let extracted_end: usize = coords[1].parse().expect("Failed to parse end coordinate");

    println!(
        "Extracted coordinates from filename: {}-{}",
        extracted_start, extracted_end
    );

    // Read the full sequence
    let full_content = fs::read_to_string(&full_fasta).expect("Failed to read chr22 100kb FASTA");

    let full_seq: String = full_content
        .lines()
        .filter(|line| !line.starts_with('>'))
        .collect::<Vec<_>>()
        .join("")
        .to_uppercase();

    println!("Extracted sequence length: {} bp", full_seq.len());

    println!("\n=== Step 3: Pick a 5kb region from the string ===");

    let region_size = 5_000;

    // Scan for a 5kb region with <50% N's
    let mut start_pos = 0;
    let mut region_5kb = "";
    let mut found_good_region = false;

    while start_pos + region_size <= full_seq.len() {
        let candidate = &full_seq[start_pos..(start_pos + region_size)];
        let n_count = candidate.chars().filter(|&c| c == 'N').count();
        let n_percentage = (n_count as f64 / region_size as f64) * 100.0;

        if n_percentage < 50.0 {
            region_5kb = candidate;
            found_good_region = true;
            println!(
                "Found good 5kb region at string position {}-{}",
                start_pos,
                start_pos + region_size
            );
            println!("5kb region length: {} bp", region_5kb.len());
            println!("First 60bp: {}", &region_5kb[..60]);
            println!("N content: {:.1}%", n_percentage);
            break;
        }

        start_pos += 1000; // Skip ahead 1kb at a time
    }

    assert!(
        found_good_region,
        "Could not find a 5kb region with <50% N's in the entire sequence"
    );

    assert_eq!(
        region_5kb.len(),
        region_size,
        "5kb region should be {} bp",
        region_size
    );

    println!("\n=== Step 4: Extract middle 3kb (omitting 1kb flanks) ===");

    let flank_size = 1_000;
    let middle_3kb = &region_5kb[flank_size..(region_size - flank_size)];
    assert_eq!(middle_3kb.len(), 3_000, "Middle region should be 3000 bp");

    println!("Middle 3kb length: {} bp", middle_3kb.len());
    println!("First 60bp: {}", &middle_3kb[..60]);
    println!("Last 60bp: {}", &middle_3kb[middle_3kb.len() - 60..]);

    println!("\n=== Step 5: Derive coordinate range FROM FASTA ===");

    // Calculate genomic coordinates from the extracted coordinates and string position
    // String position start_pos corresponds to genomic coordinate (extracted_start + start_pos)
    let genomic_start = extracted_start + start_pos;
    let genomic_end = genomic_start + region_size;

    println!(
        "String position {}-{} corresponds to genomic coordinates {}-{}",
        start_pos,
        start_pos + region_size,
        genomic_start,
        genomic_end
    );

    // Add 1kb tolerance on each side for the query
    let query_start = genomic_start.saturating_sub(1000) + 1; // 1-based, with 1kb buffer
    let query_end = genomic_end + 1000; // 1kb buffer on end
    let query_region = format!("chr22:{}-{}", query_start, query_end);

    println!("Query region (with 1kb tolerance): {}", query_region);

    println!("\n=== Step 6: Query coordinates in a follow-up command ===");

    println!("Querying region: {}", query_region);

    let query_output = test_dir.join("chr22_query");
    let status = Command::new(&graphome_bin)
        .args(&[
            "make-sequence",
            "--gfa",
            gbz_path.to_str().unwrap(),
            "--assembly",
            "chm13",
            "--region",
            &query_region,
            "--sample",
            test_sample,
            "--output",
            query_output.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to execute make-sequence for query");

    assert!(status.success(), "Query extraction failed");

    // Find the output FASTA file
    let query_fasta = fs::read_dir(&test_dir)
        .expect("Failed to read test directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("chr22_query") && n.ends_with(".fa"))
                .unwrap_or(false)
        })
        .expect("No FASTA file found for query");

    println!("Found query FASTA: {:?}", query_fasta);

    // Read the query sequence
    let query_content = fs::read_to_string(&query_fasta).expect("Failed to read query FASTA");

    let query_seq: String = query_content
        .lines()
        .filter(|line| !line.starts_with('>'))
        .collect::<Vec<_>>()
        .join("")
        .to_uppercase();

    println!("Query sequence length: {} bp", query_seq.len());
    println!("First 60bp: {}", &query_seq[..60.min(query_seq.len())]);

    println!("\n=== Step 7: Assert the 3kb string appears somewhere in the query result ===");

    // Check if the middle 3kb appears SOMEWHERE in the query result
    // (doesn't have to be exact position, just has to match somewhere)
    if query_seq.contains(middle_3kb) {
        println!("✅ SUCCESS: Middle 3kb found somewhere in query result");

        // Find position
        if let Some(pos) = query_seq.find(middle_3kb) {
            println!("   Found at position: {} bp in query result", pos);
            println!("   Query region had 1kb tolerance on each side");
        }

        println!("\n   Coordinate-based extraction is CORRECT!");
        println!("   The 3kb string from the original extraction appears in the queried region.");
    } else {
        // Print diagnostic information
        println!("❌ FAILURE: Middle 3kb NOT found anywhere in query result");
        println!("\nDiagnostics:");
        println!("  Expected (middle 3kb) length: {} bp", middle_3kb.len());
        println!("  Actual (query) length: {} bp", query_seq.len());
        println!("  Expected first 100bp: {}", &middle_3kb[..100]);
        println!(
            "  Actual first 100bp:   {}",
            &query_seq[..100.min(query_seq.len())]
        );

        // Check if sequences are similar but not exact
        let mut matches = 0;
        let compare_len = middle_3kb.len().min(query_seq.len());
        for (c1, c2) in middle_3kb.chars().zip(query_seq.chars()).take(compare_len) {
            if c1 == c2 {
                matches += 1;
            }
        }
        let similarity = (matches as f64 / compare_len as f64) * 100.0;
        println!("  Similarity: {:.1}%", similarity);

        panic!("Coordinate extraction did not return the expected sequence");
    }
}
