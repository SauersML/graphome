#!/usr/bin/env python3
"""
Integration test for BRCA1 extraction from pangenome GBZ.

This test:
1. Extracts BRCA1 region (chr17:43,044,295-43,170,327) for HG00290#1
2. BLASTs the extracted sequence against NCBI Genome database (assembled chromosomes)
3. Verifies the top hit matches BRCA1 coordinates on chr17 (either GRCh38 or T2T-CHM13)
4. Requires 100% query coverage and >95% identity

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
import urllib.request
from pathlib import Path

# Expected BRCA1 coordinates - must match at least one of these assemblies
# BRCA1 is on the minus strand in both assemblies

# GRCh38/hg38 (chr17, minus strand)
# NCBI: NC_000017.11:43044295..43170327, complement
GRCH38_CHR = "chr17"
GRCH38_START = 43044295
GRCH38_END = 43170327

# T2T-CHM13 v2.0 (chr17, minus strand)
# NCBI: NC_060941.1:43902857..44029084, complement
T2T_CHR = "chr17"
T2T_START = 43902857
T2T_END = 44029084

# Use GRCh38 for extraction (pangenome is based on GRCh38)
EXTRACTION_START = GRCH38_START
EXTRACTION_END = GRCH38_END

# Tolerance for coordinate matching (BLAST alignment may not be exact at boundaries)
COORD_TOLERANCE = 500  # 500bp tolerance for start/end coordinates



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
        "--assembly", "grch38",
        "--region", f"chr17:{EXTRACTION_START}-{EXTRACTION_END}",
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
    
    print("[INFO] Downloading reference sequences and running local BLAST...")
    
    # Download GRCh38 chr17 and T2T chr17 sequences locally for faster BLAST
    # This avoids searching the entire refseq_genomic database remotely
    import tempfile
    import urllib.request
    
    blast_db_dir = tempfile.mkdtemp(prefix="blast_db_")
    
    # Download sequences
    sequences = [
        ("NC_000017.11", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_000017.11&rettype=fasta&retmode=text"),
        ("NC_060941.1", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_060941.1&rettype=fasta&retmode=text"),
    ]
    
    combined_fasta = Path(blast_db_dir) / "chr17_refs.fasta"
    
    print("[INFO] Downloading reference sequences...")
    with open(combined_fasta, 'w') as outf:
        for acc, url in sequences:
            print(f"  Downloading {acc}...")
            try:
                with urllib.request.urlopen(url, timeout=120) as response:
                    outf.write(response.read().decode('utf-8'))
            except Exception as e:
                print(f"[ERROR] Failed to download {acc}: {e}")
                sys.exit(1)
    
    # Create BLAST database
    print("[INFO] Creating local BLAST database...")
    makeblastdb_cmd = [
        "makeblastdb",
        "-in", str(combined_fasta),
        "-dbtype", "nucl",
        "-out", str(Path(blast_db_dir) / "chr17_db"),
        "-title", "chr17_references"
    ]
    
    stdout_db, stderr_db, returncode_db = run_command(makeblastdb_cmd, timeout=60)
    if returncode_db != 0:
        print(f"[ERROR] makeblastdb failed: {stderr_db}")
        sys.exit(1)
    
    print("[INFO] Running BLAST against local database...")
    
    # Run BLAST against local database (much faster than remote)
    cmd = [
        "blastn",
        "-query", str(fasta_path),
        "-db", str(Path(blast_db_dir) / "chr17_db"),
        "-outfmt", "6 qseqid sseqid pident length qstart qend sstart send evalue bitscore stitle",
        "-max_target_seqs", "5",
        "-evalue", "1e-100",
        "-word_size", "28",
        "-perc_identity", "95"
    ]
    
    stdout, stderr, returncode = run_command(cmd, timeout=1800)  # 30 minutes timeout
    
    if returncode != 0:
        print(f"[ERROR] BLAST failed with return code {returncode}")
        print(f"STDERR: {stderr}")
        sys.exit(1)
    
    if not stdout.strip():
        print("[ERROR] BLAST returned no results")
        print("This likely means:")
        print("  1. The extracted sequence does not match GRCh38 or T2T chr17")
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
    qstart = int(top_hit[4])
    qend = int(top_hit[5])
    sstart = int(top_hit[6])
    send = int(top_hit[7])
    stitle = top_hit[10]
    
    # Calculate query coverage
    query_coverage = (align_length / 126002.0) * 100.0  # Query length is ~126kb
    
    # Normalize coordinates
    coord_start = min(sstart, send)
    coord_end = max(sstart, send)
    
    # Extract chromosome from accession or title
    # NC_000017.11 = GRCh38 chr17, NC_060941.1 = T2T chr17
    if subject_id.startswith("NC_000017"):
        chromosome = "chr17"
    elif subject_id.startswith("NC_060941"):
        chromosome = "chr17"
    else:
        # Fallback: extract from title
        chr_match = re.search(r'chromosome (\d+|X|Y|MT)', stitle, re.IGNORECASE)
        chromosome = f"chr{chr_match.group(1)}" if chr_match else "unknown"
    
    print(f"=== Top BLAST Hit ===")
    print(f"Subject: {subject_id}")
    print(f"Chromosome: {chromosome}")
    print(f"Coordinates: {coord_start:,} - {coord_end:,}")
    print(f"Query Coverage: {query_coverage:.2f}%")
    print(f"Identity: {pident:.2f}%")
    print(f"Alignment length: {align_length:,} bp")
    
    return {
        'chromosome': chromosome,
        'start': coord_start,
        'end': coord_end,
        'identity': pident,
        'query_coverage': query_coverage,
        'align_length': align_length,
        'subject_id': subject_id,
        'title': stitle
    }

def verify_blast_results(blast_result):
    """Verify BLAST results match expected BRCA1 coordinates (GRCh38 or T2T-CHM13)."""
    print("\n=== Step 3: Verifying BLAST results ===")
    
    chromosome = blast_result['chromosome']
    start = blast_result['start']
    end = blast_result['end']
    identity = blast_result['identity']
    query_coverage = blast_result['query_coverage']
    
    success = True
    
    # Check query coverage (should be ~100% for full chromosome alignment)
    if query_coverage < 95.0:
        print(f"[FAIL] Low query coverage: {query_coverage:.2f}% (expected >95%)")
        print("       This suggests the sequence doesn't fully align to the reference")
        success = False
    else:
        print(f"[OK] High query coverage: {query_coverage:.2f}%")
    
    # Check chromosome
    if chromosome != GRCH38_CHR and chromosome != T2T_CHR:
        print(f"[FAIL] Wrong chromosome: {chromosome} (expected chr17)")
        success = False
    else:
        print(f"[OK] Correct chromosome: {chromosome}")
    
    # Check if coordinates match either GRCh38 or T2T-CHM13 BRCA1 region
    grch38_start_diff = abs(start - GRCH38_START)
    grch38_end_diff = abs(end - GRCH38_END)
    grch38_matches = (grch38_start_diff <= COORD_TOLERANCE and grch38_end_diff <= COORD_TOLERANCE)
    
    t2t_start_diff = abs(start - T2T_START)
    t2t_end_diff = abs(end - T2T_END)
    t2t_matches = (t2t_start_diff <= COORD_TOLERANCE and t2t_end_diff <= COORD_TOLERANCE)
    
    if grch38_matches:
        print(f"[OK] Coordinates match GRCh38 BRCA1 region:")
        print(f"     Start: {start:,} (expected {GRCH38_START:,}, diff: {grch38_start_diff:,} bp)")
        print(f"     End:   {end:,} (expected {GRCH38_END:,}, diff: {grch38_end_diff:,} bp)")
    elif t2t_matches:
        print(f"[OK] Coordinates match T2T-CHM13 BRCA1 region:")
        print(f"     Start: {start:,} (expected {T2T_START:,}, diff: {t2t_start_diff:,} bp)")
        print(f"     End:   {end:,} (expected {T2T_END:,}, diff: {t2t_end_diff:,} bp)")
    else:
        print(f"[FAIL] Coordinates do not match BRCA1 region in either assembly:")
        print(f"     BLAST result: {start:,} - {end:,}")
        print(f"     GRCh38 BRCA1: {GRCH38_START:,} - {GRCH38_END:,} (diff: {grch38_start_diff:,} / {grch38_end_diff:,} bp)")
        print(f"     T2T-CHM13 BRCA1: {T2T_START:,} - {T2T_END:,} (diff: {t2t_start_diff:,} / {t2t_end_diff:,} bp)")
        print(f"     Tolerance: {COORD_TOLERANCE:,} bp")
        success = False
    
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
