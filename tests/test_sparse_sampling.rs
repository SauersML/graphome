use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

const SAMPLE_NAME: &str = "HG002";
const REGION: &str = "chr10:0-548";
const OUTPUT_STEM: &str = "chr10";
const EXPECTED_PATH_LENGTHS: [(&str, usize); 2] = [("grch38_0_chr10", 549), ("chm13_0_chr10", 549)];

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
    let source_gfa = repo_root.join("data/minimal.gfa");
    let source_paf = repo_root.join("data/minimal.paf");
    assert!(
        source_gfa.exists(),
        "Expected GFA file {:?} to exist",
        source_gfa
    );
    assert!(
        source_paf.exists(),
        "Expected PAF file {:?} to exist",
        source_paf
    );

    let temp_dir = tempfile::tempdir()?;
    let sequences_dir = temp_dir.path().join("sequences");
    fs::create_dir_all(&sequences_dir)?;
    let output_prefix = sequences_dir.join(OUTPUT_STEM);

    let temp_gfa = temp_dir.path().join("minimal.gfa");
    let temp_paf = temp_dir.path().join("minimal.paf");
    fs::copy(&source_gfa, &temp_gfa)?;
    fs::copy(&source_paf, &temp_paf)?;

    let vg_source = repo_root.join("vg");
    assert!(
        vg_source.exists(),
        "Expected vg binary {:?} to exist",
        vg_source
    );
    let vg_dest = temp_dir.path().join("vg");
    fs::copy(&vg_source, &vg_dest)?;
    #[cfg(unix)]
    {
        let mut perms = fs::metadata(&vg_dest)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&vg_dest, perms)?;
    }

    let binary = std::env::var("CARGO_BIN_EXE_graphome")
        .unwrap_or_else(|_| env!("CARGO_BIN_EXE_graphome").to_string());

    let (chr_name, coords) = REGION
        .split_once(':')
        .expect("Region constant must contain ':'");
    let (start_str, end_str) = coords
        .split_once('-')
        .expect("Region constant must contain '-' after ':'");
    let region_start: usize = start_str.parse()?;
    let region_end: usize = end_str.parse()?;

    let status = Command::new(binary)
        .current_dir(temp_dir.path())
        .arg("make-sequence")
        .arg("--gfa")
        .arg(&temp_gfa)
        .arg("--paf")
        .arg(&temp_paf)
        .arg("--region")
        .arg(REGION)
        .arg("--sample")
        .arg(SAMPLE_NAME)
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

    assert_eq!(
        fasta_files.len(),
        EXPECTED_PATH_LENGTHS.len(),
        "Expected {} FASTA files in {:?}, found {}",
        EXPECTED_PATH_LENGTHS.len(),
        sequences_dir,
        fasta_files.len()
    );

    let expected_prefix = format!("{}_{}_", OUTPUT_STEM, SAMPLE_NAME);
    let expected_suffix = format!("_{}-{}.fa", region_start, region_end);
    let mut expected_lengths: HashMap<String, usize> = EXPECTED_PATH_LENGTHS
        .iter()
        .map(|(name, len)| ((*name).to_string(), *len))
        .collect();
    let expected_header = format!(
        ">{}_{}:{}-{}",
        SAMPLE_NAME, chr_name, region_start, region_end
    );

    for fasta in &fasta_files {
        let file_name = fasta
            .file_name()
            .and_then(|name| name.to_str())
            .expect("FASTA file name must be valid UTF-8");
        assert!(
            file_name.starts_with(&expected_prefix),
            "Unexpected FASTA prefix in {:?}",
            fasta
        );
        assert!(
            file_name.ends_with(&expected_suffix),
            "Unexpected FASTA suffix in {:?}",
            fasta
        );
        let safe_path_name =
            &file_name[expected_prefix.len()..file_name.len() - expected_suffix.len()];

        let mut header_line = String::new();
        BufReader::new(fs::File::open(fasta)?).read_line(&mut header_line)?;
        assert_eq!(
            header_line.trim(),
            expected_header,
            "FASTA header mismatch for {:?}",
            fasta
        );

        let base_pairs = fasta_base_pair_count(fasta)?;
        let expected = expected_lengths.remove(safe_path_name).unwrap_or_else(|| {
            panic!(
                "Unexpected path name '{}' derived from {:?}",
                safe_path_name, fasta
            )
        });
        assert_eq!(
            base_pairs, expected,
            "Unexpected sequence length for path '{}' in {:?}",
            safe_path_name, fasta
        );
    }

    assert!(
        expected_lengths.is_empty(),
        "Missing FASTA outputs for paths: {:?}",
        expected_lengths.keys().collect::<Vec<_>>()
    );

    Ok(())
}
