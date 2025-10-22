# Memory-Mapped GBZ Hardening Status

## Completed Improvements ✅

### 1. Typed Error Handling
- **Added `MappedGbzError` enum** with specific error types:
  - `Io`: I/O errors
  - `InvalidFormat`: Header/format validation failures
  - `BoundaryMismatch`: BWT record boundary issues
  - `MissingData`: Required data not found
  - `OutOfRange`: Index out of bounds
- **Replaced `io::Result` with custom `Result<T>`** throughout
- **Removed unwraps** from critical paths

### 2. BWT Record Slicing Hardening
- **Fixed record count**: Now uses `count_ones()` correctly (not comparing to byte lengths)
- **Added `validate()` method** with four invariants:
  - (a) Record count ≥ 0 (empty is valid)
  - (b) First record starts at offset 0
  - (c) All records strictly increasing: `start[i] < start[i+1]`
  - (d) Last record limit equals `data_len`
- **Removed all `unwrap()` calls** in record access
- **Proper error handling** for `select()` failures
- **Boundary checks** before slicing

### 3. Serializer Helpers
- **Replaced manual `from_le_bytes`** with `usize::load()`
- **Use `bits::round_up_to_word_bytes()`** for padding calculation
- **Proper size field handling** using `std::mem::size_of::<usize>()`

### 4. API Hygiene
- **Removed tag mutation**: No longer calls `tags.insert("source", ...)`
- **Preserved provenance**: Tags loaded as-is from file
- **Validated GBZ header** before parsing GBWT section
- **ENDMARKER sentinel check** already in place for `start()`

### 5. Forward Semantics
- **Verified `forward()` implementation** is correct
- **Added documentation** explaining LF-mapping semantics
- **Single-parameter `lf()`** is the correct API (internally finds successor)

## Remaining Work 🚧

### Priority 1: Critical for Correctness

#### 6. Graph Length View (IN PROGRESS)
**Status**: Not started
**What's needed**:
```rust
pub struct GraphDescriptor {
    /// Offset to sequence offset array in memory map
    offsets_offset: usize,
    /// Number of sequences
    sequence_count: usize,
}

impl GraphDescriptor {
    /// Get length of node without loading sequence data
    pub fn node_len(&self, map: &MemoryMap, node_id: u64) -> Option<u32> {
        // Read offsets[node_id] and offsets[node_id+1]
        // Return difference
    }
}
```
**Why**: coord2node needs node lengths but doesn't need base sequences

#### 7. Lazy Metadata Loading
**Status**: Not started
**Current problem**: Metadata takes ~900 MB for HPRC file
**Solution**:
- Parse metadata header to get counts/offsets
- Store offsets to path_names, sample_names, contig_names tables
- Load only requested names on-demand
- Target: <100 MB memory usage

#### 8. Container Parsing Invariants
**Status**: Partial (GBZ header validated)
**Still needed**:
- Record absolute offsets for each block (GBZ header, Tags, GBWT, Graph)
- Assert GBWT ends before Graph begins
- Validate padding ≤ 7 bytes
- Check bidirectional flag matches actual structure

### Priority 2: Testing & Validation

#### 9. Test Suite
**Status**: Not started
**Required tests**:
1. **Round-trip parity**: 1000 random path walks vs fully-loaded GBZ
2. **Coord2node parity**: Random interval translations vs full loader
3. **Corruption tests**: Fuzz record boundaries, ensure clean errors
4. **Boundary tests**: Empty BWT, single record, max-size records
5. **Endmarker tests**: Verify ENDMARKER handling in all paths

#### 10. Benchmarking
**Status**: Basic timing done, needs depth
**Still needed**:
- Wall-clock time
- Minor/major page faults (`perf stat`)
- RSS over time
- Bytes actually read (eBPF or `perf`)
- Test scenarios:
  - Open-only (no queries)
  - Single interval query
  - 100 intervals (same chromosome)
  - Mixed chromosomes
  - Cold cache vs warm cache

### Priority 3: Integration & Polish

#### 11. Coord2node Integration
**Status**: Not started
**Design**:
```rust
pub struct ReferenceIndex {
    /// Checkpoint every 32-128 kbp
    checkpoints: Vec<(usize, Pos)>, // (genomic_pos, gbwt_pos)
}

impl ReferenceIndex {
    pub fn query(&self, gbz: &MappedGBZ, chr: &str, start: usize, end: usize) 
        -> Vec<NodeHit> {
        // 1. Find nearest checkpoint ≤ start
        // 2. Walk path from checkpoint to start
        // 3. Continue walking through [start, end]
        // 4. Return overlapping nodes
    }
}
```

#### 12. Safety & Portability
**Status**: Needs review
**Issues**:
- Treat on-disk counts as `u64`, cast to `usize` with range checks
- Document 64-bit OS assumption
- Ensure `Send + Sync` for descriptors
- Make MemoryMap lifetime explicit in public API
- Add safety comments for `unsafe` blocks

## Code Quality Metrics

### Before Hardening
- ❌ Multiple `unwrap()` calls in hot paths
- ❌ Manual byte manipulation
- ❌ Generic `io::Error` for all failures
- ❌ No boundary validation
- ❌ Tag mutation

### After Hardening
- ✅ Zero `unwrap()` in record access
- ✅ Proper serializer helpers
- ✅ Typed error enum with context
- ✅ Four BWT invariants validated
- ✅ Preserved provenance

## Performance Impact

Current implementation (after hardening):
- **Load time**: ~6.8s (5.1 GB file)
- **Memory**: ~928 MB (18% of file size)
- **Correctness**: Improved (validated invariants)
- **Error handling**: Much better (typed errors, no panics)

Target after full hardening:
- **Load time**: <1s (lazy metadata)
- **Memory**: <100 MB (lazy everything)
- **Correctness**: Production-ready (full test suite)
- **Error handling**: Comprehensive (all edge cases)

## Next Steps

1. **Implement Graph length view** (Priority 1, item 6)
2. **Add lazy metadata loading** (Priority 1, item 7)
3. **Create basic test suite** (Priority 2, item 9)
4. **Integrate with coord2node** (Priority 3, item 11)

## Files Modified

- `src/mapped_gbz.rs`: All improvements applied here
- `HARDENING_STATUS.md`: This document (new)
- `MEMORY_MAPPED_GBZ_RESULTS.md`: Original results (still valid)

## Conclusion

The architectural foundation is solid and the critical correctness issues have been addressed. The implementation is now much more robust with proper error handling, boundary validation, and correct use of serializer helpers. However, it's not yet "production-ready" - it needs:

1. Graph length access for coord2node
2. Lazy metadata to reduce memory
3. Comprehensive test suite
4. Integration with actual coord2node queries

The path forward is clear and the remaining work is well-defined.
