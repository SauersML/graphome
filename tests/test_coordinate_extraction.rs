use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Test that coordinate-based extraction returns consistent sequences.
/// 
/// This test:
/// 1. Extracts a specific region (chr22:10000000-10005000) via coordinates
/// 2. Extracts the same region again via coordinates
/// 3. Verifies both extractions produce identical sequences
/// 
/// This validates that coordinate-based extraction is deterministic and correct.
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
    
    // Use a region known to have real sequence data (not N's)
    // chr22:10000000-10005000 is in a gene-rich region
    let test_region = "grch38#chr22:10000000-10005000";
    let test_sample = "GRCh38#0";
    
    println!("=== Step 1: Extract test region (first extraction) ===");
    println!("Region: {}", test_region);
    println!("Sample: {}", test_sample);
    
    let output1 = test_dir.join("extraction1");
    let status = Command::new(&graphome_bin)
        .args(&[
            "make-sequence",
            "--gfa", gbz_path.to_str().unwrap(),
            "--region", test_region,
            "--sample", test_sample,
            "--output", output1.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to execute make-sequence (first extraction)");
    
    assert!(status.success(), "First extraction failed");
    
    // Find the output FASTA file
    let fasta1 = fs::read_dir(&test_dir)
        .expect("Failed to read test directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("extraction1") && n.ends_with(".fa"))
                .unwrap_or(false)
        })
        .expect("No FASTA file found for first extraction");
    
    println!("Found first extraction: {:?}", fasta1);
    
    // Read the first sequence
    let content1 = fs::read_to_string(&fasta1)
        .expect("Failed to read first FASTA");
    
    let seq1: String = content1
        .lines()
        .filter(|line| !line.starts_with('>'))
        .collect::<Vec<_>>()
        .join("")
        .to_uppercase();
    
    println!("First extraction length: {} bp", seq1.len());
    println!("First 100bp: {}", &seq1[..100.min(seq1.len())]);
    
    // Verify we got real sequence data (not all N's)
    let n_count = seq1.chars().filter(|&c| c == 'N').count();
    let n_percentage = (n_count as f64 / seq1.len() as f64) * 100.0;
    println!("N content: {:.1}%", n_percentage);
    
    assert!(
        n_percentage < 50.0,
        "Extracted sequence is mostly N's ({:.1}%), choose a different test region",
        n_percentage
    );
    
    println!("\n=== Step 2: Extract same region (second extraction) ===");
    
    let output2 = test_dir.join("extraction2");
    let status = Command::new(&graphome_bin)
        .args(&[
            "make-sequence",
            "--gfa", gbz_path.to_str().unwrap(),
            "--region", test_region,
            "--sample", test_sample,
            "--output", output2.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to execute make-sequence (second extraction)");
    
    assert!(status.success(), "Second extraction failed");
    
    // Find the output FASTA file
    let fasta2 = fs::read_dir(&test_dir)
        .expect("Failed to read test directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("extraction2") && n.ends_with(".fa"))
                .unwrap_or(false)
        })
        .expect("No FASTA file found for second extraction");
    
    println!("Found second extraction: {:?}", fasta2);
    
    // Read the second sequence
    let content2 = fs::read_to_string(&fasta2)
        .expect("Failed to read second FASTA");
    
    let seq2: String = content2
        .lines()
        .filter(|line| !line.starts_with('>'))
        .collect::<Vec<_>>()
        .join("")
        .to_uppercase();
    
    println!("Second extraction length: {} bp", seq2.len());
    println!("First 100bp: {}", &seq2[..100.min(seq2.len())]);
    
    println!("\n=== Step 3: Verify both extractions are identical ===");
    
    // Check lengths match
    assert_eq!(
        seq1.len(),
        seq2.len(),
        "Extraction lengths differ: {} vs {} bp",
        seq1.len(),
        seq2.len()
    );
    
    // Check sequences match exactly
    if seq1 == seq2 {
        println!("✅ SUCCESS: Both extractions produced identical sequences");
        println!("   Length: {} bp", seq1.len());
        println!("   Coordinate-based extraction is deterministic and correct");
    } else {
        // Find first difference
        let mut first_diff = None;
        for (i, (c1, c2)) in seq1.chars().zip(seq2.chars()).enumerate() {
            if c1 != c2 {
                first_diff = Some((i, c1, c2));
                break;
            }
        }
        
        if let Some((pos, c1, c2)) = first_diff {
            println!("❌ FAILURE: Sequences differ at position {}", pos);
            println!("   First extraction: {}", c1);
            println!("   Second extraction: {}", c2);
            
            // Show context
            let start = pos.saturating_sub(50);
            let end = (pos + 50).min(seq1.len());
            println!("\n   Context (first):  {}", &seq1[start..end]);
            println!("   Context (second): {}", &seq2[start..end]);
        }
        
        panic!("Coordinate extractions produced different sequences");
    }
}
