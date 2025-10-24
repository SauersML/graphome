#!/usr/bin/env python3
"""
Integration test for BRCA1 extraction from pangenome GBZ.

This test:
1. Extracts BRCA1 region (chr17:43,044,295-43,170,245) for HG00290#1
2. BLASTs the extracted sequence against GRCh38
3. Verifies the top hit matches the expected BRCA1 coordinates

Requirements:
- graphome binary built (target/release/graphome)
- HPRC GBZ file (data/hprc-v2.0-mc-grch38.gbz)
- NCBI BLAST+ installed (blastn command available)
- Internet connection (for BLAST remote database)
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

def blast_sequence(fasta_path, skip_blast=False):
    """BLAST the sequence against GRCh38 and return top hit coordinates."""
    print("\n=== Step 2: BLASTing sequence against GRCh38 ===")
    
    if skip_blast:
        print("[INFO] Skipping BLAST (use --blast flag to enable)")
        print("[INFO] To manually BLAST this sequence:")
        print(f"  1. Go to https://blast.ncbi.nlm.nih.gov/Blast.cgi?PROGRAM=blastn&PAGE_TYPE=BlastSearch")
        print(f"  2. Upload file: {fasta_path}")
        print(f"  3. Database: Nucleotide collection (nr/nt)")
        print(f"  4. Organism: Homo sapiens (taxid:9606)")
        print(f"  5. Expected result: chr17:43,044,295-43,170,245 (BRCA1)")
        return None
    
    # Check if blastn is available
    try:
        subprocess.run(["blastn", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] blastn not found. Please install NCBI BLAST+")
        print("  Ubuntu/Debian: sudo apt-get install ncbi-blast+")
        print("  macOS: brew install blast")
        sys.exit(1)
    
    # Run BLAST against remote human genome database
    cmd = [
        "blastn",
        "-query", str(fasta_path),
        "-db", "nt",
        "-remote",
        "-entrez_query", "Homo sapiens[Organism]",
        "-outfmt", "6 qseqid sseqid pident length qstart qend sstart send evalue bitscore stitle",
        "-max_target_seqs", "5",
        "-evalue", "1e-50"
    ]
    
    print("[INFO] Running remote BLAST (this may take 1-2 minutes)...")
    stdout, stderr, returncode = run_command(cmd, timeout=600)
    
    if returncode != 0:
        print(f"[ERROR] BLAST failed with return code {returncode}")
        print(f"STDERR: {stderr}")
        sys.exit(1)
    
    if not stdout.strip():
        print("[ERROR] BLAST returned no results")
        sys.exit(1)
    
    print("[OK] BLAST completed")
    print("\n=== BLAST Results (top 5 hits) ===")
    print(stdout)
    
    # Parse top hit
    lines = stdout.strip().split('\n')
    if not lines:
        print("[ERROR] No BLAST hits found")
        sys.exit(1)
    
    top_hit = lines[0].split('\t')
    if len(top_hit) < 11:
        print(f"[ERROR] Unexpected BLAST output format: {top_hit}")
        sys.exit(1)
    
    subject_id = top_hit[1]
    pident = float(top_hit[2])
    align_length = int(top_hit[3])
    sstart = int(top_hit[6])
    send = int(top_hit[7])
    evalue = float(top_hit[8])
    bitscore = float(top_hit[9])
    stitle = top_hit[10]
    
    # Extract chromosome and coordinates from subject title
    # Format: "gi|568815597|ref|NC_000017.11| Homo sapiens chromosome 17, GRCh38.p14 Primary Assembly"
    chr_match = re.search(r'chromosome (\d+|X|Y|MT)', stitle, re.IGNORECASE)
    chromosome = f"chr{chr_match.group(1)}" if chr_match else "unknown"
    
    # Normalize coordinates (BLAST may return them in reverse order)
    coord_start = min(sstart, send)
    coord_end = max(sstart, send)
    
    print(f"\n=== Top BLAST Hit ===")
    print(f"Subject: {subject_id}")
    print(f"Title: {stitle}")
    print(f"Chromosome: {chromosome}")
    print(f"Coordinates: {coord_start:,} - {coord_end:,}")
    print(f"Identity: {pident:.2f}%")
    print(f"Alignment length: {align_length:,} bp")
    print(f"E-value: {evalue}")
    print(f"Bit score: {bitscore}")
    
    return {
        'chromosome': chromosome,
        'start': coord_start,
        'end': coord_end,
        'identity': pident,
        'align_length': align_length,
        'evalue': evalue,
        'subject_id': subject_id,
        'title': stitle
    }

def verify_coordinates(blast_result):
    """Verify BLAST coordinates match expected BRCA1 region."""
    print("\n=== Step 3: Verifying coordinates ===")
    
    chromosome = blast_result['chromosome']
    start = blast_result['start']
    end = blast_result['end']
    identity = blast_result['identity']
    
    # Check chromosome
    if chromosome != EXPECTED_CHR:
        print(f"[FAIL] Chromosome mismatch: expected {EXPECTED_CHR}, got {chromosome}")
        return False
    print(f"[OK] Chromosome matches: {chromosome}")
    
    # Check coordinates (with tolerance)
    start_diff = abs(start - EXPECTED_START)
    end_diff = abs(end - EXPECTED_END)
    
    if start_diff > COORD_TOLERANCE:
        print(f"[FAIL] Start coordinate mismatch: expected {EXPECTED_START:,}, got {start:,} (diff: {start_diff:,} bp)")
        return False
    print(f"[OK] Start coordinate matches: {start:,} (diff: {start_diff} bp)")
    
    if end_diff > COORD_TOLERANCE:
        print(f"[FAIL] End coordinate mismatch: expected {EXPECTED_END:,}, got {end:,} (diff: {end_diff:,} bp)")
        return False
    print(f"[OK] End coordinate matches: {end:,} (diff: {end_diff} bp)")
    
    # Check identity (should be very high for same individual)
    if identity < 95.0:
        print(f"[WARNING] Low identity: {identity:.2f}% (expected >95%)")
    else:
        print(f"[OK] High identity: {identity:.2f}%")
    
    return True

def main():
    """Main test function."""
    import argparse
    parser = argparse.ArgumentParser(description="Test BRCA1 extraction from pangenome GBZ")
    parser.add_argument("--blast", action="store_true", help="Run BLAST verification (requires internet)")
    args = parser.parse_args()
    
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
        
        # Step 1: Extract BRCA1
        fasta_path = extract_brca1(graphome_bin, gbz_path, tmpdir)
        
        # Step 2: BLAST sequence (optional)
        blast_result = blast_sequence(fasta_path, skip_blast=not args.blast)
        
        # Step 3: Verify coordinates (only if BLAST was run)
        if blast_result:
            success = verify_coordinates(blast_result)
            
            print("\n" + "=" * 80)
            if success:
                print("✅ TEST PASSED: BRCA1 extraction verified successfully")
                print("=" * 80)
                sys.exit(0)
            else:
                print("❌ TEST FAILED: Coordinate verification failed")
                print("=" * 80)
                sys.exit(1)
        else:
            print("\n" + "=" * 80)
            print("✅ TEST PASSED: BRCA1 extraction completed successfully")
            print("   (BLAST verification skipped - use --blast to enable)")
            print("=" * 80)
            sys.exit(0)

if __name__ == "__main__":
    main()
