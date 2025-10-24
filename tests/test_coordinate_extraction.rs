use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Test that coordinate-based extraction returns the correct sequence.
/// 
/// This test:
/// 1. Extracts entire chr22 for GRCh38#0
/// 2. Takes a 5kb slice from position 1,000,000-1,005,000
/// 3. Extracts the middle 3kb (omitting 1kb flanks)
/// 4. Queries the same region via coordinate-based extraction
/// 5. Asserts the 3kb string appears in the coordinate-extracted sequence
#[test]
fn test_coordinate_extraction_correctness() {
    // Setup paths
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    
    // Try multiple possible GBZ locations
    let gbz_path = if project_root.join("data/hprc/hprc-v2.0-mc-grch38.gbz").exists() {
        project_root.join("data/hprc/hprc-v2.0-mc-grch38.gbz")
    } else if project_root.join("data/hprc-v2.0-mc-grch38.gbz").exists() {
        project_root.join("data/hprc-v2.0-mc-grch38.gbz")
    } else {
        panic!(
            "GBZ file not found. Tried:\n\
             - {:?}\n\
             - {:?}\n\
             Download with:\n\
             mkdir -p data/hprc && wget https://s3-us-west-2.amazonaws.com/human-pangenomics/pangenomes/freeze/release2/minigraph-cactus/hprc-v2.0-mc-grch38.gbz -O data/hprc/hprc-v2.0-mc-grch38.gbz",
            project_root.join("data/hprc/hprc-v2.0-mc-grch38.gbz"),
            project_root.join("data/hprc-v2.0-mc-grch38.gbz")
        );
    };
    
    let graphome_bin = project_root.join("target/release/graphome");
    
    // Check if binary is built
    if !graphome_bin.exists() {
        panic!(
            "graphome binary not found at {:?}\n\
             Build with: cargo build --release",
            graphome_bin
        );
    }
    
    let test_dir = project_root.join("target/test_coordinate_extraction");
    fs::create_dir_all(&test_dir).expect("Failed to create test directory");
    
    println!("=== Step 1: Extract entire chr22 for GRCh38#0 ===");
    
    // Extract entire chr22 (chr22 is ~50Mb, coordinates 1-50818468)
    // We'll extract a large region that includes our test region
    let chr22_output = test_dir.join("grch38_chr22_full");
    let status = Command::new(&graphome_bin)
        .args(&[
            "make-sequence",
            "--gfa", gbz_path.to_str().unwrap(),
            // No --paf needed for GBZ files
            "--region", "grch38#chr22:1-2000000",  // First 2Mb of chr22
            "--sample", "GRCh38#0",
            "--output", chr22_output.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to execute make-sequence");
    
    assert!(status.success(), "make-sequence command failed for full chr22");
    
    // Find the output FASTA file
    let chr22_fasta = fs::read_dir(&test_dir)
        .expect("Failed to read test directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("grch38_chr22_full") && n.ends_with(".fa"))
                .unwrap_or(false)
        })
        .expect("No chr22 FASTA file found");
    
    println!("Found chr22 FASTA: {:?}", chr22_fasta);
    
    println!("\n=== Step 2: Extract 5kb region (1,000,000-1,005,000) from full sequence ===");
    
    // Read the full chr22 sequence
    let chr22_content = fs::read_to_string(&chr22_fasta)
        .expect("Failed to read chr22 FASTA");
    
    // Parse FASTA (skip header, concatenate sequence lines)
    let chr22_seq: String = chr22_content
        .lines()
        .filter(|line| !line.starts_with('>'))
        .collect::<Vec<_>>()
        .join("")
        .to_uppercase();
    
    println!("Full chr22 length: {} bp", chr22_seq.len());
    
    // Extract 5kb region (0-based indexing: 999,999 to 1,004,999)
    let region_start = 999_999;
    let region_end = 1_004_999;
    
    if chr22_seq.len() <= region_end {
        panic!(
            "chr22 sequence too short: {} bp (need at least {} bp)",
            chr22_seq.len(),
            region_end + 1
        );
    }
    
    let region_5kb = &chr22_seq[region_start..=region_end];
    assert_eq!(region_5kb.len(), 5001, "5kb region should be 5001 bp");
    
    println!("Extracted 5kb region: {} bp", region_5kb.len());
    
    println!("\n=== Step 3: Extract middle 3kb (omitting 1kb flanks) ===");
    
    // Extract middle 3kb (skip first 1000bp and last 1000bp)
    let flank_size = 1000;
    let middle_3kb = &region_5kb[flank_size..(region_5kb.len() - flank_size)];
    assert_eq!(middle_3kb.len(), 3001, "Middle region should be 3001 bp");
    
    println!("Middle 3kb: {} bp", middle_3kb.len());
    println!("First 60bp: {}", &middle_3kb[..60]);
    println!("Last 60bp: {}", &middle_3kb[middle_3kb.len()-60..]);
    
    println!("\n=== Step 4: Query same region via coordinate-based extraction ===");
    
    // Query the same 5kb region using coordinate-based extraction
    // Note: coordinates are 1-based in the command, so 1,000,000-1,005,000
    let coord_output = test_dir.join("grch38_chr22_coord");
    let status = Command::new(&graphome_bin)
        .args(&[
            "make-sequence",
            "--gfa", gbz_path.to_str().unwrap(),
            // No --paf needed for GBZ files
            "--region", "grch38#chr22:1000000-1005000",
            "--sample", "GRCh38#0",
            "--output", coord_output.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to execute make-sequence");
    
    assert!(status.success(), "make-sequence command failed");
    
    // Find the coordinate-extracted FASTA file
    let coord_fasta = fs::read_dir(&test_dir)
        .expect("Failed to read test directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("grch38_chr22_coord") && n.ends_with(".fa"))
                .unwrap_or(false)
        })
        .expect("No coordinate-extracted FASTA file found");
    
    println!("Found coordinate-extracted FASTA: {:?}", coord_fasta);
    
    // Read the coordinate-extracted sequence
    let coord_content = fs::read_to_string(&coord_fasta)
        .expect("Failed to read coordinate-extracted FASTA");
    
    let coord_seq: String = coord_content
        .lines()
        .filter(|line| !line.starts_with('>'))
        .collect::<Vec<_>>()
        .join("")
        .to_uppercase();
    
    println!("Coordinate-extracted sequence length: {} bp", coord_seq.len());
    
    println!("\n=== Step 5: Verify middle 3kb appears in coordinate-extracted sequence ===");
    
    // Check if the middle 3kb appears in the coordinate-extracted sequence
    if coord_seq.contains(middle_3kb) {
        println!("✅ SUCCESS: Middle 3kb found in coordinate-extracted sequence");
        
        // Find position
        if let Some(pos) = coord_seq.find(middle_3kb) {
            println!("   Position in extracted sequence: {} bp", pos);
        }
    } else {
        // Print diagnostic information
        println!("❌ FAILURE: Middle 3kb NOT found in coordinate-extracted sequence");
        println!("\nDiagnostics:");
        println!("  Expected (middle 3kb) length: {} bp", middle_3kb.len());
        println!("  Actual (coord-extracted) length: {} bp", coord_seq.len());
        println!("  Expected first 100bp: {}", &middle_3kb[..100]);
        println!("  Actual first 100bp:   {}", &coord_seq[..100.min(coord_seq.len())]);
        
        // Check if first 100bp match
        let first_100_match = coord_seq.starts_with(&middle_3kb[..100]);
        println!("  First 100bp match: {}", first_100_match);
        
        // Check if it's a substring issue (maybe with small offset)
        let mut found_partial = false;
        for offset in 0..1000 {
            if offset + middle_3kb.len() > coord_seq.len() {
                break;
            }
            let matches = middle_3kb.chars()
                .zip(coord_seq[offset..].chars())
                .take(1000)
                .filter(|(a, b)| a == b)
                .count();
            
            if matches > 900 {
                println!("  Found high similarity at offset {}: {}/1000 matches", offset, matches);
                found_partial = true;
                break;
            }
        }
        
        if !found_partial {
            println!("  No high-similarity regions found");
        }
        
        panic!("Coordinate extraction did not return the expected sequence");
    }
}
