use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;

const REGION: &str = "chr22:1-50818468";
const EXPECTED_MIN_BP: usize = 40_000_000;
const EXPECTED_MAX_BP: usize = 60_000_000;
const EXPECTED_MIN_COVERAGE: f64 = 80.0;

fn repo_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn fasta_base_pair_count(path: &Path) -> std::io::Result<usize> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut count = 0usize;

    for line in reader.lines() {
        let line = line?;
        if !line.starts_with('>') {
            count += line.trim().len();
        }
    }

    Ok(count)
}

#[test]
fn chr22_sparse_sampling_regression() -> Result<(), Box<dyn std::error::Error>> {
    let repo_root = repo_path();
    let gfa_path = repo_root.join("data/hprc/hprc-v2.0-mc-grch38.gbz");
    assert!(
        gfa_path.exists(),
        "Expected graph file {:?} to exist",
        gfa_path
    );

    let temp_dir = tempfile::tempdir()?;
    let sequences_dir = temp_dir.path().join("sequences");
    fs::create_dir_all(&sequences_dir)?;
    let output_prefix = sequences_dir.join("chr22");

    let paf_path = temp_dir.path().join("dummy.paf");
    fs::File::create(&paf_path)?;

    let binary = std::env::var("CARGO_BIN_EXE_graphome")
        .unwrap_or_else(|_| env!("CARGO_BIN_EXE_graphome").to_string());
    let status = Command::new(binary)
        .arg("make-sequence")
        .arg("--gfa")
        .arg(&gfa_path)
        .arg("--paf")
        .arg(&paf_path)
        .arg("--region")
        .arg(REGION)
        .arg("--sample")
        .arg("HG002")
        .arg("--output")
        .arg(&output_prefix)
        .status()?;

    assert!(
        status.success(),
        "graphome make-sequence invocation failed: {status:?}"
    );

    let fasta_files: Vec<PathBuf> = fs::read_dir(&sequences_dir)?
        .filter_map(|entry| {
            entry.ok().and_then(|dir_entry| {
                let path = dir_entry.path();
                if path.extension().map(|ext| ext == "fa").unwrap_or(false) {
                    Some(path)
                } else {
                    None
                }
            })
        })
        .collect();

    assert!(
        !fasta_files.is_empty(),
        "Expected FASTA files to be produced in {:?}",
        sequences_dir
    );

    for fasta in &fasta_files {
        let metadata = fs::metadata(fasta)?;
        assert!(
            metadata.len() > 1_000_000,
            "File {:?} is too small ({} bytes), expected > 1MB",
            fasta,
            metadata.len()
        );

        let base_pairs = fasta_base_pair_count(fasta)?;
        assert!(
            base_pairs >= EXPECTED_MIN_BP,
            "File {:?} has too few base pairs ({}), expected >= {}",
            fasta,
            base_pairs,
            EXPECTED_MIN_BP
        );

        assert!(
            base_pairs <= EXPECTED_MAX_BP,
            "File {:?} has too many base pairs ({}), expected <= {}",
            fasta,
            base_pairs,
            EXPECTED_MAX_BP
        );

        let coverage = (base_pairs as f64 / 50_818_468.0) * 100.0;
        assert!(
            coverage >= EXPECTED_MIN_COVERAGE,
            "Coverage too low for {:?}: {:.1}%, expected >= {:.1}%",
            fasta,
            coverage,
            EXPECTED_MIN_COVERAGE
        );
    }

    Ok(())
}
