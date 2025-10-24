#!/usr/bin/env python3
"""
Integration test for BRCA1 extraction from pangenome GBZ.

This test verifies the --assembly parameter works correctly with cross-validation:

TEST 1: Extract sample HG00290#1 with CHM13, verify against GRCh38
1. Extracts BRCA1 using --assembly chm13 with CHM13 coordinates (chr17:43,902,857-44,029,084)
2. Verifies the code uses CHM13#0#chr17 reference path
3. BLASTs the extracted sequence against GRCh38 chr17
4. Verifies the top hit matches GRCh38 BRCA1 coordinates (chr17:43,044,295-43,170,327)
5. Requires ≥98% query coverage and ≥98% identity

TEST 2: Extract sample HG00290#1 with GRCh38, verify against CHM13
1. Extracts BRCA1 using --assembly grch38 with GRCh38 coordinates (chr17:43,044,295-43,170,327)
2. Verifies the code uses GRCh38#0#chr17 reference path
3. BLASTs the extracted sequence against T2T-CHM13 chr17
4. Verifies the top hit matches CHM13 BRCA1 coordinates (chr17:43,902,857-44,029,084)
5. Requires ≥98% query coverage and ≥98% identity

TEST 3: Extract reference GRCh38#0 with GRCh38, verify against CHM13
1. Extracts BRCA1 reference using --assembly grch38 with GRCh38 coordinates
2. BLASTs against T2T-CHM13 chr17
3. Verifies the top hit matches CHM13 BRCA1 coordinates
4. Requires ≥98% query coverage and ≥98% identity

TEST 4: Extract reference CHM13#0 with CHM13, verify against GRCh38
1. Extracts BRCA1 reference using --assembly chm13 with CHM13 coordinates
2. BLASTs against GRCh38 chr17
3. Verifies the top hit matches GRCh38 BRCA1 coordinates
4. Requires ≥98% query coverage and ≥98% identity

This cross-validation proves that both assembly parameters extract the correct BRCA1 region
for both sample haplotypes and reference sequences.

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

# Minimum thresholds for test success
MIN_COVERAGE = 98.0  # Minimum query coverage percentage
MIN_IDENTITY = 98.0  # Minimum sequence identity percentage



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

def extract_brca1(graphome_bin, gbz_path, output_dir, assembly, region_start, region_end, output_prefix, sample="HG00290#1"):
    """Extract BRCA1 region for specified sample using specified assembly."""
    print(f"\n=== Extracting BRCA1 sequence using {assembly.upper()} coordinates for sample {sample} ===")
    
    cmd = [
        str(graphome_bin),
        "make-sequence",
        "--gfa", str(gbz_path),
        "--assembly", assembly,
        "--region", f"chr17:{region_start}-{region_end}",
        "--sample", sample,
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
    print(f"\n=== Verifying BLAST results for {expected_assembly.upper()} reference ===")
    
    chromosome = blast_result['chromosome']
    start = blast_result['start']
    end = blast_result['end']
    identity = blast_result['identity']
    query_coverage = blast_result['query_coverage']
    
    success = True
    
    # Check query coverage (must be ≥98%)
    if query_coverage < MIN_COVERAGE:
        print(f"[FAIL] Low query coverage: {query_coverage:.2f}% (required ≥{MIN_COVERAGE}%)")
        print("       This suggests the sequence doesn't fully align to the reference")
        success = False
    else:
        print(f"[OK] Query coverage: {query_coverage:.2f}% (≥{MIN_COVERAGE}%)")
    
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
    
    # Check identity (must be ≥98%)
    if identity < MIN_IDENTITY:
        print(f"[FAIL] Low sequence identity: {identity:.2f}% (required ≥{MIN_IDENTITY}%)")
        success = False
    else:
        print(f"[OK] Sequence identity: {identity:.2f}% (≥{MIN_IDENTITY}%)")
    
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
        
        all_success = True
        test_results = []
        
        # TEST 1: Extract sample HG00290#1 with CHM13, BLAST against GRCh38
        print("\n" + "=" * 80)
        print("TEST 1: Extract sample HG00290#1 with CHM13, verify against GRCh38")
        print("=" * 80)
        
        chm13_fasta = extract_brca1(
            graphome_bin, gbz_path, tmpdir,
            assembly="chm13",
            region_start=T2T_START,
            region_end=T2T_END,
            output_prefix="HG00290_BRCA1_CHM13",
            sample="HG00290#1"
        )
        
        # BLAST against GRCh38
        chm13_blast = blast_sequence(chm13_fasta, target_assembly="grch38")
        chm13_success = verify_blast_results(chm13_blast, expected_assembly="grch38")
        test_results.append(("TEST 1 (HG00290#1 CHM13→GRCh38)", chm13_success))
        
        if chm13_success:
            print("\n✅ TEST 1 PASSED: CHM13 extraction matches GRCh38 BRCA1")
        else:
            print("\n❌ TEST 1 FAILED: CHM13 extraction did not match GRCh38 BRCA1")
            all_success = False
        
        # TEST 2: Extract sample HG00290#1 with GRCh38, BLAST against CHM13
        print("\n" + "=" * 80)
        print("TEST 2: Extract sample HG00290#1 with GRCh38, verify against CHM13")
        print("=" * 80)
        
        grch38_fasta = extract_brca1(
            graphome_bin, gbz_path, tmpdir,
            assembly="grch38",
            region_start=GRCH38_START,
            region_end=GRCH38_END,
            output_prefix="HG00290_BRCA1_GRCh38",
            sample="HG00290#1"
        )
        
        # BLAST against T2T-CHM13
        grch38_blast = blast_sequence(grch38_fasta, target_assembly="chm13")
        grch38_success = verify_blast_results(grch38_blast, expected_assembly="chm13")
        test_results.append(("TEST 2 (HG00290#1 GRCh38→CHM13)", grch38_success))
        
        if grch38_success:
            print("\n✅ TEST 2 PASSED: GRCh38 extraction matches CHM13 BRCA1")
        else:
            print("\n❌ TEST 2 FAILED: GRCh38 extraction did not match CHM13 BRCA1")
            all_success = False
        
        # TEST 3: Extract reference GRCh38#0 with GRCh38, BLAST against CHM13
        print("\n" + "=" * 80)
        print("TEST 3: Extract reference GRCh38#0 with GRCh38, verify against CHM13")
        print("=" * 80)
        
        grch38_ref_fasta = extract_brca1(
            graphome_bin, gbz_path, tmpdir,
            assembly="grch38",
            region_start=GRCH38_START,
            region_end=GRCH38_END,
            output_prefix="GRCh38_REF_BRCA1",
            sample="GRCh38#0"
        )
        
        # BLAST against T2T-CHM13
        grch38_ref_blast = blast_sequence(grch38_ref_fasta, target_assembly="chm13")
        grch38_ref_success = verify_blast_results(grch38_ref_blast, expected_assembly="chm13")
        test_results.append(("TEST 3 (GRCh38#0 GRCh38→CHM13)", grch38_ref_success))
        
        if grch38_ref_success:
            print("\n✅ TEST 3 PASSED: GRCh38 reference extraction matches CHM13 BRCA1")
        else:
            print("\n❌ TEST 3 FAILED: GRCh38 reference extraction did not match CHM13 BRCA1")
            all_success = False
        
        # TEST 4: Extract reference CHM13#0 with CHM13, BLAST against GRCh38
        print("\n" + "=" * 80)
        print("TEST 4: Extract reference CHM13#0 with CHM13, verify against GRCh38")
        print("=" * 80)
        
        chm13_ref_fasta = extract_brca1(
            graphome_bin, gbz_path, tmpdir,
            assembly="chm13",
            region_start=T2T_START,
            region_end=T2T_END,
            output_prefix="CHM13_REF_BRCA1",
            sample="CHM13#0"
        )
        
        # BLAST against GRCh38
        chm13_ref_blast = blast_sequence(chm13_ref_fasta, target_assembly="grch38")
        chm13_ref_success = verify_blast_results(chm13_ref_blast, expected_assembly="grch38")
        test_results.append(("TEST 4 (CHM13#0 CHM13→GRCh38)", chm13_ref_success))
        
        if chm13_ref_success:
            print("\n✅ TEST 4 PASSED: CHM13 reference extraction matches GRCh38 BRCA1")
        else:
            print("\n❌ TEST 4 FAILED: CHM13 reference extraction did not match GRCh38 BRCA1")
            all_success = False
        
        # Final summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        for test_name, success in test_results:
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{status}: {test_name}")
        print("=" * 80)
        
        if all_success:
            print("✅ ALL TESTS PASSED: All extractions verified via BLAST")
            print("=" * 80)
            sys.exit(0)
        else:
            passed = sum(1 for _, s in test_results if s)
            total = len(test_results)
            print(f"❌ SOME TESTS FAILED: {passed}/{total} tests passed")
            print("=" * 80)
            sys.exit(1)

if __name__ == "__main__":
    main()
