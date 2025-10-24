#!/usr/bin/env python3
"""
Integration test for BRCA1 extraction from pangenome GBZ.

This test:
1. Extracts BRCA1 region (chr17:43,044,295-43,170,245) for HG00290#1
2. Verifies extraction completed successfully (both anchors found)
3. Verifies sequence length is approximately correct (~126kb)

Note: This test does NOT verify sequence correctness via BLAST because:
- Remote BLAST is unreliable (timeouts, CPU limits)
- Local BLAST requires downloading GRCh38 genome (~3GB)
- Sequence comparison fails due to indels (HG00290 has variants vs GRCh38)

Requirements:
- graphome binary built (target/release/graphome)
- HPRC GBZ file (data/hprc-v2.0-mc-grch38.gbz)
- Python 3
"""

import subprocess
import sys
import os
import tempfile
import re
from pathlib import Path

# Expected BRCA1 coordinates (GRCh38/hg38)
EXPECTED_CHR = "chr17"
EXPECTED_START = 43044295
EXPECTED_END = 43170245
EXPECTED_LENGTH = EXPECTED_END - EXPECTED_START + 1  # ~126kb

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
        "--region", f"grch38#chr17:{EXPECTED_START}-{EXPECTED_END}",
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

def read_fasta_sequence(fasta_path):
    """Read sequence from FASTA file (excluding header)."""
    with open(fasta_path, 'r') as f:
        lines = f.readlines()
    # Skip header line, join sequence lines, remove whitespace
    sequence = ''.join(line.strip() for line in lines if not line.startswith('>'))
    return sequence.upper()

def verify_extraction(fasta_path):
    """Verify extraction quality."""
    print("\n=== Step 2: Verifying extraction ===")
    
    # Read sequence
    sequence = read_fasta_sequence(fasta_path)
    seq_length = len(sequence)
    
    print(f"Sequence length: {seq_length:,} bp")
    
    # Check length is approximately correct for BRCA1 (~126kb)
    expected_length = EXPECTED_LENGTH
    length_diff = abs(seq_length - expected_length)
    length_diff_pct = (length_diff / expected_length) * 100
    
    success = True
    
    if length_diff_pct > 5.0:
        print(f"[FAIL] Length differs too much: {seq_length:,} bp (expected ~{expected_length:,} bp, diff: {length_diff_pct:.1f}%)")
        success = False
    else:
        print(f"[OK] Length is correct: {seq_length:,} bp (expected ~{expected_length:,} bp, diff: {length_diff_pct:.1f}%)")
    
    # Check for valid DNA sequence
    valid_bases = set('ACGT')
    invalid_bases = set(sequence) - valid_bases
    if invalid_bases:
        print(f"[WARNING] Found invalid bases: {invalid_bases}")
    else:
        print(f"[OK] All bases are valid DNA (ACGT)")
    
    # Check GC content (should be reasonable for human genome, ~40-60%)
    gc_count = sequence.count('G') + sequence.count('C')
    gc_content = (gc_count / seq_length) * 100
    
    if gc_content < 30 or gc_content > 70:
        print(f"[WARNING] Unusual GC content: {gc_content:.1f}% (expected 40-60%)")
    else:
        print(f"[OK] GC content is reasonable: {gc_content:.1f}%")
    
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
        
        # Step 2: Verify extraction quality
        success = verify_extraction(sample_fasta)
        
        print("\n" + "=" * 80)
        if success:
            print("✅ TEST PASSED: BRCA1 extraction completed successfully")
            print("=" * 80)
            sys.exit(0)
        else:
            print("❌ TEST FAILED: Extraction verification failed")
            print("=" * 80)
            sys.exit(1)

if __name__ == "__main__":
    main()
