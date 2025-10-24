# BRCA1 Extraction Integration Test

## Overview

`test_brca1_extraction.py` is an integration test that verifies the correctness of sequence extraction from the HPRC pangenome GBZ file.

## What it does

1. **Extracts BRCA1 region** (chr17:43,044,295-43,170,245) for sample HG00290 haplotype 1
2. **Verifies extraction is complete** (both start and end anchors found)
3. **Optionally BLASTs** the sequence against NCBI to verify coordinates

## Requirements

- `graphome` binary built (`cargo build --release`)
- HPRC GBZ file downloaded (`data/hprc-v2.0-mc-grch38.gbz`)
- Python 3
- NCBI BLAST+ (optional, for `--blast` flag)

## Usage

### Basic test (extraction only)

```bash
python3 scripts/test_brca1_extraction.py
```

This will:
- Extract BRCA1 for HG00290#1
- Verify the extraction completed successfully
- Print instructions for manual BLAST verification

### Full test with BLAST verification

```bash
python3 scripts/test_brca1_extraction.py --blast
```

This will additionally:
- BLAST the extracted sequence against NCBI nt database
- Verify the top hit matches chr17:43,044,295-43,170,245
- Confirm high sequence identity (>95%)

**Note:** BLAST verification requires internet connection and may take 1-2 minutes.

## Expected Output

### Successful extraction (without BLAST)

```
================================================================================
BRCA1 Extraction Integration Test
================================================================================

=== Step 1: Extracting BRCA1 sequence ===
[OK] Extraction completed successfully
[OK] Output file: /tmp/.../HG00290_BRCA1_...fa

=== Step 2: BLASTing sequence against GRCh38 ===
[INFO] Skipping BLAST (use --blast flag to enable)

================================================================================
✅ TEST PASSED: BRCA1 extraction completed successfully
   (BLAST verification skipped - use --blast to enable)
================================================================================
```

### Successful extraction with BLAST

```
================================================================================
BRCA1 Extraction Integration Test
================================================================================

=== Step 1: Extracting BRCA1 sequence ===
[OK] Extraction completed successfully

=== Step 2: BLASTing sequence against GRCh38 ===
[OK] BLAST completed

=== Top BLAST Hit ===
Chromosome: chr17
Coordinates: 43,044,295 - 43,170,245
Identity: 99.8%

=== Step 3: Verifying coordinates ===
[OK] Chromosome matches: chr17
[OK] Start coordinate matches: 43,044,295
[OK] End coordinate matches: 43,170,245
[OK] High identity: 99.8%

================================================================================
✅ TEST PASSED: BRCA1 extraction verified successfully
================================================================================
```

## Manual BLAST Verification

If you prefer to verify manually:

1. Run the test without `--blast` flag
2. Note the output FASTA file path
3. Go to https://blast.ncbi.nlm.nih.gov/Blast.cgi?PROGRAM=blastn&PAGE_TYPE=BlastSearch
4. Upload the FASTA file
5. Set database to "Nucleotide collection (nr/nt)"
6. Set organism to "Homo sapiens (taxid:9606)"
7. Run BLAST
8. Verify top hit is chr17:43,044,295-43,170,245 (BRCA1)

## Troubleshooting

### "graphome binary not found"

Build the binary:
```bash
cargo build --release
```

### "GBZ file not found"

Download the HPRC GBZ file:
```bash
wget https://s3-us-west-2.amazonaws.com/human-pangenomics/pangenomes/freeze/release2/minigraph-cactus/hprc-v2.0-mc-grch38.gbz -O data/hprc-v2.0-mc-grch38.gbz
```

### "blastn not found"

Install NCBI BLAST+:
```bash
# Ubuntu/Debian
sudo apt-get install ncbi-blast+

# macOS
brew install blast
```

### "BLAST returned no results"

This can happen with remote BLAST due to:
- Network issues
- NCBI server load
- Query timeout

Try:
1. Running the test again
2. Using manual BLAST verification (see above)
3. Checking your internet connection

## What this test validates

- ✅ Extraction finds both start and end anchors (complete region)
- ✅ Extracted sequence length is correct (~126kb)
- ✅ Sequence matches expected BRCA1 coordinates
- ✅ High sequence identity with reference genome
- ✅ Sample-specific filtering works correctly
- ✅ No structural variants or assembly gaps in BRCA1 region

## Related

- Main extraction command: `graphome make-sequence`
- Sample filtering: `--sample HG00290#1`
- Completeness detection: Logs show "COMPLETE" or "INCOMPLETE"
