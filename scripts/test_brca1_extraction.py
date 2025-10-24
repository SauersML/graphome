#!/usr/bin/env python3
"""
Integration test for BRCA1 extraction from pangenome GBZ.

This test:
1. Extracts BRCA1 region (chr17:43,044,295-43,170,245) for HG00290#1
2. BLASTs the extracted sequence against NCBI nt database
3. Verifies the top hit matches the expected BRCA1 coordinates on chr17
   for GRCh38 or T2T-CHM13v2.0

Requirements:
- graphome binary built (target/release/graphome)
- HPRC GBZ file (data/hprc-v2.0-mc-grch38.gbz)
- NCBI BLAST+ installed (blastn command)
- Internet connection for remote BLAST
- Python 3
"""

import subprocess
import sys
import os
import tempfile
import re
from pathlib import Path
from typing import Optional

# Expected BRCA1 coordinates for supported reference assemblies
# Coordinates are 1-based inclusive.
REFERENCE_COORDINATES = {
    "grch38": {
        "chromosome": "chr17",
        "start": 43044295,
        "end": 43170245,
    },
    # The NCBI nt database often returns the T2T-CHM13v2.0 assembly as the
    # top hit for chromosome 17. BRCA1 resides at ~71.3 Mbp on that assembly.
    "t2t-chm13v2": {
        "chromosome": "chr17",
        "start": 71322675,
        "end": 71448371,
    },
}

GRCH38_COORDS = REFERENCE_COORDINATES["grch38"]

# Tolerance for coordinate matching (BLAST alignment may not be exact at boundaries)
COORD_TOLERANCE = 1000  # 1kb tolerance

def run_command(cmd, cwd=None, timeout=300):
    """Run a shell command and return stdout, stderr, returncode."""
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    return result.stdout, result.stderr, result.returncode

def extract_brca1(graphome_bin, gbz_path, output_dir):
    """Extract BRCA1 region for HG00290#1."""
    print("\n=== Step 1: Extracting BRCA1 sequence ===")
    
    cmd = [
        str(graphome_bin),
        "make-sequence",
        "--gfa", str(gbz_path),
        "--paf", "/dev/null",
        "--region",
        (
            "grch38#"  # graphome region spec uses GRCh38 coordinates
            f"{GRCH38_COORDS['chromosome']}:{GRCH38_COORDS['start']}"
            f"-{GRCH38_COORDS['end']}"
        ),
        "--sample", "HG00290#1",
        "--output", "HG00290_BRCA1"
    ]
    
    stdout, stderr, returncode = run_command(cmd, cwd=output_dir)
    
    if returncode != 0:
        print(f"[ERROR] Extraction failed with return code {returncode}")
        print(f"STDERR: {stderr}")
        sys.exit(1)
    
    # Check for COMPLETE status in logs
    if "COMPLETE: Found both anchors" not in stderr:
        print("[ERROR] Extraction did not find both anchors (incomplete region)")
        print(f"STDERR: {stderr}")
        sys.exit(1)
    
    print("[OK] Extraction completed successfully")
    
    # Find the output FASTA file
    fasta_files = list(Path(output_dir).glob("HG00290_BRCA1_*.fa"))
    if not fasta_files:
        print("[ERROR] No FASTA output file found")
        sys.exit(1)
    
    fasta_path = fasta_files[0]
    print(f"[OK] Output file: {fasta_path}")
    
    # Verify file size (should be ~126kb)
    file_size = fasta_path.stat().st_size
    if file_size < 100000 or file_size > 200000:
        print(f"[WARNING] Unexpected file size: {file_size} bytes (expected ~126kb)")
    
    return fasta_path

def blast_sequence(fasta_path):
    """BLAST the extracted sequence against NCBI to verify coordinates."""
    print("\n=== Step 2: BLASTing sequence against NCBI ===")
    
    # Check if blastn is available
    try:
        subprocess.run(["blastn", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] blastn not found. Please install NCBI BLAST+")
        print("  Ubuntu/Debian: sudo apt-get install ncbi-blast+")
        print("  macOS: brew install blast")
        sys.exit(1)
    
    print("[INFO] Running BLAST against NCBI nt database (this may take 2-5 minutes)...")
    
    # Run BLAST with optimized parameters for large query
    cmd = [
        "blastn",
        "-query", str(fasta_path),
        "-db", "nt",
        "-remote",
        "-entrez_query", "Homo sapiens[Organism] AND (chromosome 17[Title] OR chr17[Title])",
        "-outfmt", "6 qseqid sseqid pident length qstart qend sstart send evalue bitscore stitle",
        "-max_target_seqs", "5",
        "-evalue", "1e-100",
        "-word_size", "28",  # Larger word size for faster search
        "-perc_identity", "95"  # Only high-identity hits
    ]
    
    stdout, stderr, returncode = run_command(cmd, timeout=600)
    
    if returncode != 0:
        print(f"[ERROR] BLAST failed with return code {returncode}")
        print(f"STDERR: {stderr}")
        sys.exit(1)
    
    if not stdout.strip():
        print("[ERROR] BLAST returned no results")
        print("This likely means:")
        print("  1. The extracted sequence is not from chr17")
        print("  2. The sequence quality is too low")
        print("  3. NCBI remote BLAST service is unavailable")
        sys.exit(1)
    
    print("[OK] BLAST completed successfully")
    print("\n=== BLAST Results (top 5 hits) ===")
    for line in stdout.strip().split('\n')[:5]:
        fields = line.split('\t')
        if len(fields) >= 11:
            print(f"  {fields[1]}: {fields[2]}% identity, coords {fields[6]}-{fields[7]}")
    print()
    
    # Parse top hit
    top_hit = stdout.strip().split('\n')[0].split('\t')
    if len(top_hit) < 11:
        print(f"[ERROR] Unexpected BLAST output format")
        sys.exit(1)
    
    subject_id = top_hit[1]
    pident = float(top_hit[2])
    align_length = int(top_hit[3])
    sstart = int(top_hit[6])
    send = int(top_hit[7])
    stitle = top_hit[10]
    
    # Normalize coordinates
    coord_start = min(sstart, send)
    coord_end = max(sstart, send)
    
    # Extract chromosome from title
    chr_match = re.search(r'chromosome (\d+|X|Y|MT)', stitle, re.IGNORECASE)
    chromosome = f"chr{chr_match.group(1)}" if chr_match else "unknown"
    
    print(f"=== Top BLAST Hit ===")
    print(f"Subject: {subject_id}")
    print(f"Chromosome: {chromosome}")
    print(f"Coordinates: {coord_start:,} - {coord_end:,}")
    print(f"Identity: {pident:.2f}%")
    print(f"Alignment length: {align_length:,} bp")
    
    return {
        'chromosome': chromosome,
        'start': coord_start,
        'end': coord_end,
        'identity': pident,
        'align_length': align_length,
        'subject_id': subject_id,
        'title': stitle
    }


def identify_reference_assembly(subject_id: str, title: str) -> Optional[str]:
    """Infer the reference assembly of a BLAST hit based on metadata."""
    metadata = f"{subject_id} {title}".lower()

    if "t2t" in metadata or "chm13" in metadata or "cp139549.2" in metadata:
        return "t2t-chm13v2"
    if "grch38" in metadata or "hg38" in metadata:
        return "grch38"

    return None

def verify_blast_results(blast_result):
    """Verify BLAST results match expected BRCA1 coordinates."""
    print("\n=== Step 3: Verifying BLAST results ===")

    chromosome = blast_result['chromosome']
    start = blast_result['start']
    end = blast_result['end']
    identity = blast_result['identity']
    subject_id = blast_result['subject_id']
    title = blast_result['title']

    success = True

    assembly = identify_reference_assembly(subject_id, title)
    if assembly is None:
        # Fall back to the closest known reference based on coordinate proximity
        diffs = {
            ref: abs(start - coords['start']) + abs(end - coords['end'])
            for ref, coords in REFERENCE_COORDINATES.items()
        }
        assembly = min(diffs, key=diffs.get)
        print(
            "[WARNING] Unable to determine assembly from BLAST metadata. "
            f"Assuming {assembly} based on coordinate similarity."
        )

    expected = REFERENCE_COORDINATES[assembly]
    expected_chr = expected['chromosome']
    expected_start = expected['start']
    expected_end = expected['end']

    print(f"[INFO] Detected reference assembly: {assembly}")

    # Check chromosome
    if chromosome != expected_chr:
        print(f"[FAIL] Wrong chromosome: {chromosome} (expected {expected_chr})")
        success = False
    else:
        print(f"[OK] Correct chromosome: {chromosome}")

    # Check coordinates (with tolerance for alignment boundaries)
    start_diff = abs(start - expected_start)
    end_diff = abs(end - expected_end)

    if start_diff > COORD_TOLERANCE:
        print(
            "[FAIL] Start coordinate mismatch: "
            f"{start:,} (expected {expected_start:,}, diff: {start_diff:,} bp)"
        )
        success = False
    else:
        print(
            f"[OK] Start coordinate matches: {start:,} (diff: {start_diff:,} bp)"
        )

    if end_diff > COORD_TOLERANCE:
        print(
            "[FAIL] End coordinate mismatch: "
            f"{end:,} (expected {expected_end:,}, diff: {end_diff:,} bp)"
        )
        success = False
    else:
        print(f"[OK] End coordinate matches: {end:,} (diff: {end_diff:,} bp)")

    # Check identity
    if identity < 95.0:
        print(f"[FAIL] Low sequence identity: {identity:.2f}% (expected >95%)")
        success = False
    else:
        print(f"[OK] High sequence identity: {identity:.2f}%")

    return success

def main():
    """Main test function."""
    print("=" * 80)
    print("BRCA1 Extraction Integration Test")
    print("=" * 80)
    
    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Check for required files
    graphome_bin = project_root / "target" / "release" / "graphome"
    gbz_path = project_root / "data" / "hprc-v2.0-mc-grch38.gbz"
    
    if not graphome_bin.exists():
        print(f"[ERROR] graphome binary not found: {graphome_bin}")
        print("Please build with: cargo build --release")
        sys.exit(1)
    
    if not gbz_path.exists():
        print(f"[ERROR] GBZ file not found: {gbz_path}")
        print("Please download with:")
        print("  wget https://s3-us-west-2.amazonaws.com/human-pangenomics/pangenomes/freeze/release2/minigraph-cactus/hprc-v2.0-mc-grch38.gbz -O data/hprc-v2.0-mc-grch38.gbz")
        sys.exit(1)
    
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n[INFO] Working directory: {tmpdir}")
        
        # Step 1: Extract BRCA1 for HG00290#1
        sample_fasta = extract_brca1(graphome_bin, gbz_path, tmpdir)
        
        # Step 2: BLAST the sequence
        blast_result = blast_sequence(sample_fasta)
        
        # Step 3: Verify BLAST results
        success = verify_blast_results(blast_result)
        
        print("\n" + "=" * 80)
        if success:
            print("✅ TEST PASSED: BRCA1 extraction verified via BLAST")
            print("=" * 80)
            sys.exit(0)
        else:
            print("❌ TEST FAILED: BLAST verification failed")
            print("=" * 80)
            sys.exit(1)

if __name__ == "__main__":
    main()
