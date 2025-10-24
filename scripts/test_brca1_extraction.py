#!/usr/bin/env python3
"""
Integration test for BRCA1 extraction from pangenome GBZ.

This test verifies the --assembly parameter works correctly by extracting BRCA1
using GRCh38 coordinates and verifying the result matches T2T-CHM13 BRCA1.

TEST: Extract with GRCh38, verify against CHM13
1. Extracts BRCA1 using --assembly grch38 with GRCh38 coordinates (chr17:43,044,295-43,170,327)
2. Verifies the code uses GRCh38#0#chr17 reference path (correct anchor nodes)
3. BLASTs the extracted sequence against T2T-CHM13 chr17
4. Verifies the top hit matches T2T-CHM13 BRCA1 coordinates (chr17:43,902,857-44,029,084)
5. This proves we extracted the correct BRCA1 region (not chr17:69M or other wrong location)
6. Requires >95% query coverage and >95% identity

Note: HG00290's BRCA1 sequence is more similar to T2T-CHM13 than GRCh38 (99.95% identity),
which is why we BLAST against CHM13 to verify correct extraction.

Known limitation: CHM13 extraction currently has issues with anchor node disambiguation
when nodes appear in multiple locations (segmental duplications). This will be addressed
in a future update with coordinate-based validation.

Requirements:
- graphome binary built (target/release/graphome)
- HPRC GBZ file (data/hprc-v2.0-mc-grch38.gbz)
- NCBI BLAST+ installed (blastn command)
- Internet connection to download reference sequences
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

def extract_brca1(graphome_bin, gbz_path, output_dir, assembly, region_start, region_end, output_prefix):
    """Extract BRCA1 region for HG00290#1 using specified assembly."""
    print(f"\n=== Extracting BRCA1 sequence using {assembly.upper()} coordinates ===")
    
    cmd = [
        str(graphome_bin),
        "make-sequence",
        "--gfa", str(gbz_path),
        "--assembly", assembly,
        "--region", f"chr17:{region_start}-{region_end}",
        "--sample", "HG00290#1",
        "--output", output_prefix
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
    
    # Verify correct reference path was used
    expected_ref_path = "GRCh38#0#chr17" if assembly == "grch38" else "CHM13#0#chr17"
    if f"Using reference path {expected_ref_path}" not in stderr:
        print(f"[ERROR] Expected to use reference path {expected_ref_path}")
        print(f"STDERR: {stderr}")
        sys.exit(1)
    
    print(f"[OK] Extraction completed successfully using {expected_ref_path}")
    
    # Find the output FASTA file
    fasta_files = list(Path(output_dir).glob(f"{output_prefix}_*.fa"))
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

def blast_sequence(fasta_path, target_assembly):
    """BLAST the extracted sequence against specified reference assembly."""
    print(f"\n=== BLASTing sequence against {target_assembly.upper()} reference ===")
    
    # Check if blastn is available
    try:
        subprocess.run(["blastn", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] blastn not found. Please install NCBI BLAST+")
        print("  Ubuntu/Debian: sudo apt-get install ncbi-blast+")
        print("  macOS: brew install blast")
        sys.exit(1)
    
    print(f"[INFO] Downloading {target_assembly.upper()} reference and running local BLAST...")
    
    import tempfile
    import urllib.request
    
    blast_db_dir = tempfile.mkdtemp(prefix="blast_db_")
    
    # Download only the target assembly's chr17
    if target_assembly == "grch38":
        accession = "NC_000017.11"
        assembly_name = "GRCh38"
    else:  # chm13
        accession = "NC_060941.1"
        assembly_name = "T2T-CHM13"
    
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={accession}&rettype=fasta&retmode=text"
    ref_fasta = Path(blast_db_dir) / f"{accession}.fasta"
    
    print(f"[INFO] Downloading {assembly_name} chr17 ({accession})...")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            with open(ref_fasta, 'w') as outf:
                outf.write(response.read().decode('utf-8'))
    except Exception as e:
        print(f"[ERROR] Failed to download {accession}: {e}")
        sys.exit(1)
    
    # Create BLAST database
    print("[INFO] Creating local BLAST database...")
    makeblastdb_cmd = [
        "makeblastdb",
        "-in", str(ref_fasta),
        "-dbtype", "nucl",
        "-out", str(Path(blast_db_dir) / "chr17_db"),
        "-title", f"{assembly_name}_chr17"
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

def verify_blast_results(blast_result, expected_assembly):
    """Verify BLAST results match expected BRCA1 coordinates for the specified assembly."""
    print(f"\n=== Verifying BLAST results for {expected_assembly.upper()} extraction ===")
    
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
    if chromosome != "chr17":
        print(f"[FAIL] Wrong chromosome: {chromosome} (expected chr17)")
        success = False
    else:
        print(f"[OK] Correct chromosome: {chromosome}")
    
    # Check if coordinates match the expected assembly's BRCA1 region
    if expected_assembly == "grch38":
        expected_start = GRCH38_START
        expected_end = GRCH38_END
        assembly_name = "GRCh38"
    else:  # chm13
        expected_start = T2T_START
        expected_end = T2T_END
        assembly_name = "T2T-CHM13"
    
    start_diff = abs(start - expected_start)
    end_diff = abs(end - expected_end)
    matches = (start_diff <= COORD_TOLERANCE and end_diff <= COORD_TOLERANCE)
    
    if matches:
        print(f"[OK] Coordinates match {assembly_name} BRCA1 region:")
        print(f"     Start: {start:,} (expected {expected_start:,}, diff: {start_diff:,} bp)")
        print(f"     End:   {end:,} (expected {expected_end:,}, diff: {end_diff:,} bp)")
    else:
        print(f"[FAIL] Coordinates do not match {assembly_name} BRCA1 region:")
        print(f"     BLAST result: {start:,} - {end:,}")
        print(f"     Expected: {expected_start:,} - {expected_end:,}")
        print(f"     Difference: start {start_diff:,} bp, end {end_diff:,} bp")
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
        
        # Extract using GRCh38 coordinates, BLAST against T2T-CHM13
        # This verifies that using --assembly grch38 with GRCh38 coords extracts the correct
        # BRCA1 region (which happens to match T2T better for HG00290)
        print("\n" + "=" * 80)
        print("TEST: Extract with GRCh38, verify against T2T-CHM13")
        print("=" * 80)
        
        grch38_fasta = extract_brca1(
            graphome_bin, gbz_path, tmpdir,
            assembly="grch38",
            region_start=GRCH38_START,
            region_end=GRCH38_END,
            output_prefix="HG00290_BRCA1_GRCh38"
        )
        
        # BLAST against T2T-CHM13 (HG00290's BRCA1 is more similar to T2T)
        grch38_blast = blast_sequence(grch38_fasta, target_assembly="chm13")
        success = verify_blast_results(grch38_blast, expected_assembly="chm13")
        
        # Final summary
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
