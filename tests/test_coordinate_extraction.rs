use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

const SAMPLE_NAME: &str = "HG002";
const TARGET_PATH_FRAGMENT: &str = "grch38_0_chr10";
const FULL_REGION: &str = "chr10:0-548";
const SUBSEQ_START: usize = 120;
const SUBSEQ_END: usize = 220; // inclusive end coordinate

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn graphome_binary() -> String {
    std::env::var("CARGO_BIN_EXE_graphome")
        .unwrap_or_else(|_| env!("CARGO_BIN_EXE_graphome").to_string())
}

fn run_make_sequence(
    binary: &str,
    work_dir: &Path,
    gfa_path: &Path,
    paf_path: &Path,
    region: &str,
    output_prefix: &Path,
) -> std::io::Result<Output> {
    Command::new(binary)
        .current_dir(work_dir)
        .arg("make-sequence")
        .arg("--gfa")
        .arg(gfa_path)
        .arg("--paf")
        .arg(paf_path)
        .arg("--assembly")
        .arg("grch38")
        .arg("--region")
        .arg(region)
        .arg("--sample")
        .arg(SAMPLE_NAME)
        .arg("--output")
        .arg(output_prefix)
        .output()
}

fn find_path_fasta(dir: &Path, fragment: &str) -> TestResult<PathBuf> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("fa"))
            .unwrap_or(false)
        {
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.contains(fragment))
                .unwrap_or(false)
            {
                return Ok(path);
            }
        }
    }

    Err(format!(
        "No FASTA file containing '{}' found in {}",
        fragment,
        dir.display()
    )
    .into())
}

fn read_fasta_sequence(path: &Path) -> TestResult<String> {
    let content = fs::read_to_string(path)?;
    let sequence: String = content
        .lines()
        .filter(|line| !line.starts_with('>'))
        .collect();
    Ok(sequence)
}

#[test]
fn test_coordinate_extraction_correctness() -> TestResult<()> {
    let repo_root = repo_root();
    let source_gfa = repo_root.join("data/minimal.gfa");
    let source_paf = repo_root.join("data/minimal.paf");
    assert!(source_gfa.exists(), "Missing GFA at {:?}", source_gfa);
    assert!(source_paf.exists(), "Missing PAF at {:?}", source_paf);

    let temp_dir = tempfile::tempdir()?;
    let work_dir = temp_dir.path();
    let sequences_dir = work_dir.join("sequences");
    fs::create_dir_all(&sequences_dir)?;

    let temp_gfa = work_dir.join("minimal.gfa");
    let temp_paf = work_dir.join("minimal.paf");
    fs::copy(&source_gfa, &temp_gfa)?;
    fs::copy(&source_paf, &temp_paf)?;

    let vg_source = repo_root.join("vg");
    assert!(vg_source.exists(), "Expected vg binary at {:?}", vg_source);
    let vg_dest = work_dir.join("vg");
    fs::copy(&vg_source, &vg_dest)?;
    #[cfg(unix)]
    {
        let mut perms = fs::metadata(&vg_dest)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&vg_dest, perms)?;
    }

    let binary = graphome_binary();
    let full_output_prefix = sequences_dir.join("chr10_full");

    let full_output = run_make_sequence(
        &binary,
        work_dir,
        &temp_gfa,
        &temp_paf,
        FULL_REGION,
        &full_output_prefix,
    )?;

    assert!(
        full_output.status.success(),
        "graphome make-sequence failed for initial region.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&full_output.stdout),
        String::from_utf8_lossy(&full_output.stderr)
    );

    let full_fasta_path = find_path_fasta(&sequences_dir, TARGET_PATH_FRAGMENT)?;
    let full_sequence = read_fasta_sequence(&full_fasta_path)?;

    assert!(
        SUBSEQ_END < full_sequence.len(),
        "Subsequence end {} exceeds sequence length {}",
        SUBSEQ_END,
        full_sequence.len()
    );

    let start_index = if SUBSEQ_START == 0 {
        0
    } else {
        SUBSEQ_START - 1
    };

    assert!(
        start_index <= SUBSEQ_END,
        "Calculated start index {} exceeds end {}",
        start_index,
        SUBSEQ_END
    );

    let expected_bytes = &full_sequence.as_bytes()[start_index..=SUBSEQ_END];
    let expected_subsequence = String::from_utf8(expected_bytes.to_vec())?;

    let query_region = format!("chr10:{}-{}", SUBSEQ_START, SUBSEQ_END);
    let query_output_prefix = sequences_dir.join("chr10_query");
    let query_output = run_make_sequence(
        &binary,
        work_dir,
        &temp_gfa,
        &temp_paf,
        &query_region,
        &query_output_prefix,
    )?;

    assert!(
        query_output.status.success(),
        "graphome make-sequence failed for query region.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&query_output.stdout),
        String::from_utf8_lossy(&query_output.stderr)
    );

    let query_fasta_path = find_path_fasta(&sequences_dir, TARGET_PATH_FRAGMENT)?;
    let query_sequence = read_fasta_sequence(&query_fasta_path)?;

    let expected_length = SUBSEQ_END - start_index + 1;
    assert_eq!(
        query_sequence.len(),
        expected_length,
        "Query sequence length mismatch"
    );

    if query_sequence != expected_subsequence {
        let doubled = format!("{}{}", query_sequence, query_sequence);
        assert!(
            doubled.contains(&expected_subsequence),
            "Expected subsequence from {} not found in query output",
            query_region
        );
    }

    Ok(())
}
