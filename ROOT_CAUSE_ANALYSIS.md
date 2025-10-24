# Root Cause Analysis: CHM13 Extraction Coordinate Mismatch

## Problem Statement

When extracting sequences using CHM13 coordinates, the extracted sequences BLAST to incorrect GRCh38 coordinates:
- **Expected**: CHM13 43,902,857-44,029,084 → GRCh38 43,044,295-43,170,327
- **Actual**: CHM13 43,902,857-44,029,084 → GRCh38 70,602,439-70,728,492
- **Offset**: ~27.5M bp difference

## Root Cause

The GBZ file (`hprc-v2.0-mc-grch38.gbz`) contains **7 CHM13 chr17 reference fragments**, all with identical metadata:
- Sample: "CHM13"
- Contig: "chr17" (no fragment suffix like `chr17[offset]`)
- Haplotype: 0
- Path IDs: 14372-14378
- Lengths: 3.3kb, 19M, 111kb, 2.8M, 735kb, 21kb, **56.7M** (selected)

### The Bug

1. **Fragment naming issue**: All 7 fragments are named "chr17" without fragment offsets in brackets
2. **parse_fragment_offset() returns 0**: Since there's no `[offset]` notation, all fragments report `fragment_start = 0`
3. **Incorrect fragment selection**: We sort by length and select the longest fragment (path_id=14378, 56.7M bp)
4. **Wrong genomic content**: path_id=14378 does NOT contain BRCA1 at the expected CHM13 coordinates
5. **Coordinate mismatch**: The fragment we selected contains genomic content that maps to GRCh38 chr17:70M, not 43M

### Why GRCh38 Works

GRCh38 has only **ONE** chr17 reference path (path_id=14379, 83.3M bp), so there's no ambiguity:
- No fragments to choose from
- Complete chromosome in a single path
- Coordinates are correct

### Mathematical Evidence

The offset is consistent:
```
GRCh38 BLAST result: 70,602,439
CHM13 query position: 43,902,857
Difference: 26,699,582 bp
```

This suggests path_id=14378 actually represents CHM13 chr17 starting at position ~26.7M, not position 0.

## Impact

- ❌ TEST 1 FAILED: Sample HG00290#1 CHM13→GRCh38 (wrong coordinates)
- ✅ TEST 2 PASSED: Sample HG00290#1 GRCh38→CHM13 (correct)
- ❌ TEST 3 FAILED: Reference GRCh38#0 GRCh38→CHM13 (low coverage, structural differences)
- ❌ TEST 4 FAILED: Reference CHM13#0 CHM13→GRCh38 (wrong coordinates)

## Proposed Solutions

### Option 1: Try All Fragments (Brute Force)
Instead of selecting one fragment, try all 7 CHM13 fragments and see which one produces valid anchors:
- Pros: Simple, might work
- Cons: Slow, doesn't solve the underlying issue

### Option 2: Use Node-Based Coordinate Mapping
Instead of relying on fragment offsets, use the actual node positions in both reference paths:
1. Find which CHM13 fragment contains the anchor nodes
2. Use that fragment regardless of its reported coordinates
- Pros: More robust, works with any GBZ structure
- Cons: More complex implementation

### Option 3: Require Proper Fragment Naming
Document that GBZ files must have properly named fragments with offsets:
- `chr17[0]`, `chr17[26699582]`, etc.
- Pros: Clean, follows conventions
- Cons: Doesn't work with existing GBZ files

### Option 4: Build Fragment Coordinate Map
Walk each fragment and determine its actual genomic coordinates by comparing node content:
- Pros: Automatic, works with any GBZ
- Cons: Very slow, complex

## Recommendation

**Option 2** is the best approach: Use node-based coordinate mapping instead of relying on fragment naming conventions. This makes the code robust to different GBZ structures and naming conventions.

## Implementation Plan

1. When finding anchors, try each fragment until we find one that contains the anchor nodes
2. Don't rely on `fragment_start` for coordinate conversion
3. Use the actual node positions within the selected fragment
4. Log which fragment was selected and why

This requires refactoring `compute_reference_anchors()` to be more robust about fragment selection.
