use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Test that coordinate-based extraction returns the correct sequence.
/// 
/// This test:
/// 1. Extracts entire chr22 for GRCh38#0
/// 2. Looks at the string and picks a 5kb region
/// 3. Extracts the middle 3kb (omitting 1kb flanks)
/// 4. Queries the same coordinate range in a NEW command
/// 5. Asserts the 3kb string appears in the queried result
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
    
    let test_sample = "GRCh38#0";
    
    println!("=== Step 1: Extract a large region of chr22 for GRCh38#0 ===");
    
    // Extract 1Mb region of chr22 (positions 10M-11M)
    // This is large enough to test coordinate extraction correctness
    let extraction_start = 10_000_000;
    let extraction_end = 11_000_000;
    let full_output = test_dir.join("chr22_full");
    let status = Command::new(&graphome_bin)
        .args(&[
            "make-sequence",
            "--gfa", gbz_path.to_str().unwrap(),
            "--region", &format!("grch38#chr22:{}-{}", extraction_start, extraction_end),
            "--sample", test_sample,
            "--output", full_output.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to execute make-sequence for chr22 region");
    
    assert!(status.success(), "Chr22 region extraction failed");
    
    // Find the output FASTA file
    let full_fasta = fs::read_dir(&test_dir)
        .expect("Failed to read test directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("chr22_full") && n.ends_with(".fa"))
                .unwrap_or(false)
        })
        .expect("No FASTA file found for full chr22");
    
    println!("Found full chr22 FASTA: {:?}", full_fasta);
    
    // Read the full sequence
    let full_content = fs::read_to_string(&full_fasta)
        .expect("Failed to read full chr22 FASTA");
    
    let full_seq: String = full_content
        .lines()
        .filter(|line| !line.starts_with('>'))
        .collect::<Vec<_>>()
        .join("")
        .to_uppercase();
    
    println!("Full chr22 length: {} bp", full_seq.len());
    
    println!("\n=== Step 2: Look at the string and pick a 5kb region ===");
    
    // Pick a 5kb region in the middle of the extracted sequence
    // We extracted 10M-11M, so pick a region around 10.5M
    // String position 500,000 = genomic position 10,500,000
    let start_pos = 500_000;
    let region_size = 5_000;
    
    // Make sure we have enough sequence
    assert!(
        full_seq.len() >= start_pos + region_size,
        "Full sequence too short: {} bp (need at least {} bp)",
        full_seq.len(),
        start_pos + region_size
    );
    
    // Extract 5kb region from the string
    let region_5kb = &full_seq[start_pos..(start_pos + region_size)];
    assert_eq!(region_5kb.len(), region_size, "5kb region should be {} bp", region_size);
    
    println!("Picked 5kb region at string position {}-{}", start_pos, start_pos + region_size);
    println!("5kb region length: {} bp", region_5kb.len());
    println!("First 60bp: {}", &region_5kb[..60]);
    
    // Verify it's not all N's
    let n_count = region_5kb.chars().filter(|&c| c == 'N').count();
    let n_percentage = (n_count as f64 / region_5kb.len() as f64) * 100.0;
    println!("N content: {:.1}%", n_percentage);
    
    assert!(
        n_percentage < 50.0,
        "Picked region is mostly N's ({:.1}%), choose a different position",
        n_percentage
    );
    
    println!("\n=== Step 3: Extract middle 3kb (omitting 1kb flanks) ===");
    
    let flank_size = 1_000;
    let middle_3kb = &region_5kb[flank_size..(region_size - flank_size)];
    assert_eq!(middle_3kb.len(), 3_000, "Middle region should be 3000 bp");
    
    println!("Middle 3kb length: {} bp", middle_3kb.len());
    println!("First 60bp: {}", &middle_3kb[..60]);
    println!("Last 60bp: {}", &middle_3kb[middle_3kb.len()-60..]);
    
    println!("\n=== Step 4: Query the same coordinate range in a NEW command ===");
    
    // The coordinate range we want to query
    // We extracted starting at genomic position 10M, so:
    // String position start_pos corresponds to genomic coordinate (extraction_start + start_pos)
    let genomic_start = extraction_start + start_pos;
    let query_start = genomic_start + 1;  // Convert to 1-based
    let query_end = genomic_start + region_size;  // End is inclusive in 1-based
    let query_region = format!("grch38#chr22:{}-{}", query_start, query_end);
    
    println!("Querying region: {}", query_region);
    
    let query_output = test_dir.join("chr22_query");
    let status = Command::new(&graphome_bin)
        .args(&[
            "make-sequence",
            "--gfa", gbz_path.to_str().unwrap(),
            "--region", &query_region,
            "--sample", test_sample,
            "--output", query_output.to_str().unwrap(),
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
    let query_content = fs::read_to_string(&query_fasta)
        .expect("Failed to read query FASTA");
    
    let query_seq: String = query_content
        .lines()
        .filter(|line| !line.starts_with('>'))
        .collect::<Vec<_>>()
        .join("")
        .to_uppercase();
    
    println!("Query sequence length: {} bp", query_seq.len());
    println!("First 60bp: {}", &query_seq[..60.min(query_seq.len())]);
    
    println!("\n=== Step 5: Assert the 3kb string appears in the queried result ===");
    
    // Check if the middle 3kb appears in the query result
    if query_seq.contains(middle_3kb) {
        println!("✅ SUCCESS: Middle 3kb found in query result");
        
        // Find position
        if let Some(pos) = query_seq.find(middle_3kb) {
            println!("   Position in query result: {} bp", pos);
            println!("   Expected position: ~{} bp (accounting for 1kb flank)", flank_size);
        }
        
        println!("\n   Coordinate-based extraction is CORRECT!");
    } else {
        // Print diagnostic information
        println!("❌ FAILURE: Middle 3kb NOT found in query result");
        println!("\nDiagnostics:");
        println!("  Expected (middle 3kb) length: {} bp", middle_3kb.len());
        println!("  Actual (query) length: {} bp", query_seq.len());
        println!("  Expected first 100bp: {}", &middle_3kb[..100]);
        println!("  Actual first 100bp:   {}", &query_seq[..100.min(query_seq.len())]);
        
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
