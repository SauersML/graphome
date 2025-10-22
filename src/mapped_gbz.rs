// mapped_gbz.rs
// Memory-mapped GBZ structures for efficient coordinate queries without loading entire file
//
// Design: Store offsets/sizes instead of borrowed slices to avoid self-referential lifetime issues.
// All data access happens through methods that take &MemoryMap and derive slices on-demand.

use gbwt::{Pos, ENDMARKER};
use simple_sds::bits;
use simple_sds::ops::{BitVec, Select};
use simple_sds::serialize::{MemoryMap, MappingMode, Serialize};
use simple_sds::sparse_vector::SparseVector;
use std::io::{self, Cursor};
use std::path::Path;
use std::fmt;

/// Errors that can occur when working with memory-mapped GBZ files
#[derive(Debug)]
pub enum MappedGbzError {
    /// I/O error
    Io(io::Error),
    /// Invalid header or format
    InvalidFormat(String),
    /// Boundary mismatch in BWT records
    BoundaryMismatch(String),
    /// Missing required data
    MissingData(String),
    /// Out of range access
    OutOfRange(String),
}

impl fmt::Display for MappedGbzError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MappedGbzError::Io(e) => write!(f, "I/O error: {}", e),
            MappedGbzError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            MappedGbzError::BoundaryMismatch(msg) => write!(f, "Boundary mismatch: {}", msg),
            MappedGbzError::MissingData(msg) => write!(f, "Missing data: {}", msg),
            MappedGbzError::OutOfRange(msg) => write!(f, "Out of range: {}", msg),
        }
    }
}

impl std::error::Error for MappedGbzError {}

impl From<io::Error> for MappedGbzError {
    fn from(e: io::Error) -> Self {
        MappedGbzError::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, MappedGbzError>;

/// Descriptor for memory-mapped BWT data
/// Stores offsets instead of references to avoid lifetime issues
pub struct BwtDescriptor {
    /// Record boundary index (loaded into memory, relatively small)
    index: SparseVector,
    /// Offset in the memory map where BWT data starts (in bytes)
    data_offset: usize,
    /// Length of BWT data (in bytes)
    data_len: usize,
}

impl BwtDescriptor {
    /// Returns the number of records in the BWT (count of 1-bits in index)
    #[inline]
    pub fn len(&self) -> usize {
        self.index.count_ones()
    }

    /// Returns true if the BWT is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Validate BWT invariants
    fn validate(&self) -> Result<()> {
        let record_count = self.len();
        
        // Invariant (a): At least one record (or empty)
        if record_count == 0 {
            return Ok(()); // Empty is valid
        }

        // Invariant (b): First record starts at 0
        let first_start = self.index.select(0)
            .ok_or_else(|| MappedGbzError::BoundaryMismatch(
                "BWT index has records but select(0) failed".to_string()
            ))?;
        if first_start != 0 {
            return Err(MappedGbzError::BoundaryMismatch(
                format!("BWT first record must start at 0, got {}", first_start)
            ));
        }

        // Invariant (c): Records are strictly increasing
        let mut prev_start = first_start;
        for i in 1..record_count {
            let curr_start = self.index.select(i)
                .ok_or_else(|| MappedGbzError::BoundaryMismatch(
                    format!("BWT select({}) failed but count_ones={}", i, record_count)
                ))?;
            if curr_start <= prev_start {
                return Err(MappedGbzError::BoundaryMismatch(
                    format!("BWT records not strictly increasing: record {} at {}, record {} at {}",
                            i-1, prev_start, i, curr_start)
                ));
            }
            prev_start = curr_start;
        }

        // Invariant (d): Last record limit equals data_len
        // The limit of the last record is data_len
        if prev_start >= self.data_len {
            return Err(MappedGbzError::BoundaryMismatch(
                format!("BWT last record starts at {} but data_len is {}", prev_start, self.data_len)
            ));
        }

        Ok(())
    }

    /// Get the byte slice for the i-th record from the memory map
    fn record_bytes<'a>(&self, map: &'a MemoryMap, i: usize) -> Result<&'a [u8]> {
        let record_count = self.len();
        if i >= record_count {
            return Err(MappedGbzError::OutOfRange(
                format!("Record index {} out of range (count={})", i, record_count)
            ));
        }

        let map_slice: &[u64] = map.as_ref();
        let byte_slice: &[u8] = unsafe {
            std::slice::from_raw_parts(
                map_slice.as_ptr() as *const u8,
                map_slice.len() * 8,
            )
        };
        
        let data = &byte_slice[self.data_offset..self.data_offset + self.data_len];
        
        // Get start position
        let start = self.index.select(i)
            .ok_or_else(|| MappedGbzError::BoundaryMismatch(
                format!("BWT select({}) failed", i)
            ))?;

        // Get limit position
        let limit = if i + 1 < record_count {
            self.index.select(i + 1)
                .ok_or_else(|| MappedGbzError::BoundaryMismatch(
                    format!("BWT select({}) failed", i + 1)
                ))?
        } else {
            // Last record extends to end of data
            self.data_len
        };

        if start >= limit {
            return Err(MappedGbzError::BoundaryMismatch(
                format!("BWT record {} has invalid range [{}..{})", i, start, limit)
            ));
        }

        if limit > data.len() {
            return Err(MappedGbzError::BoundaryMismatch(
                format!("BWT record {} limit {} exceeds data length {}", i, limit, data.len())
            ));
        }

        Ok(&data[start..limit])
    }

    /// Returns the i-th record, or None if there is no such node
    pub fn record<'a>(&self, map: &'a MemoryMap, i: usize) -> Option<gbwt::bwt::Record<'a>> {
        let bytes = self.record_bytes(map, i).ok()?;
        gbwt::bwt::Record::new(i, bytes)
    }
}

/// Descriptor for memory-mapped GBWT
/// Stores offsets and small loaded structures, derives views on-demand
pub struct GbwtDescriptor {
    /// GBWT header (loaded)
    header: gbwt::headers::Header<gbwt::headers::GBWTPayload>,
    /// Tags (loaded, small) - kept for potential future use
    #[allow(dead_code)]
    tags: gbwt::support::Tags,
    /// BWT descriptor (stores offsets, not references)
    bwt: BwtDescriptor,
    /// Endmarker array (loaded, small - one per sequence)
    endmarker: Vec<Pos>,
    /// Metadata (loaded, small)
    metadata: Option<gbwt::gbwt::Metadata>,
}

impl GbwtDescriptor {
    /// Parse a GBWT from a memory-mapped file starting at a specific offset
    /// 
    /// This loads small structures (headers, metadata, endmarker) into memory
    /// but stores offsets for the large BWT data
    pub fn parse_from_offset(map: &MemoryMap, offset: usize) -> Result<Self> {
        // Create a cursor over the memory map to read sequentially
        let map_slice: &[u64] = map.as_ref();
        let byte_slice: &[u8] = unsafe {
            std::slice::from_raw_parts(
                map_slice.as_ptr() as *const u8,
                map_slice.len() * 8,
            )
        };
        let mut cursor = Cursor::new(byte_slice);
        cursor.set_position(offset as u64);

        // Load GBWT header
        let header = gbwt::headers::Header::<gbwt::headers::GBWTPayload>::load(&mut cursor)?;
        if let Err(msg) = header.validate() {
            return Err(MappedGbzError::InvalidFormat(msg));
        }

        // Load tags (don't mutate - preserve provenance)
        let tags = gbwt::support::Tags::load(&mut cursor)?;

        // Load BWT index (SparseVector - relatively small)
        let bwt_index = SparseVector::load(&mut cursor)?;

        // Get the position where BWT data starts
        let bwt_data_offset = cursor.position() as usize;

        // Load the size of the BWT data array using proper serializer
        let data_size = usize::load(&mut cursor)?;

        // Store offset and size (data starts after the size field)
        let data_start = bwt_data_offset + std::mem::size_of::<usize>();
        
        let bwt = BwtDescriptor {
            index: bwt_index,
            data_offset: data_start,
            data_len: data_size,
        };

        // Validate BWT invariants
        bwt.validate()?;

        // Advance cursor past the BWT data (with proper padding)
        let padded_len = bits::round_up_to_word_bytes(data_size);
        cursor.set_position((data_start + padded_len) as u64);

        // Decompress the endmarker
        let endmarker = if bwt.is_empty() {
            Vec::new()
        } else {
            bwt.record(map, ENDMARKER)
                .ok_or_else(|| MappedGbzError::MissingData("Missing endmarker record".to_string()))?
                .decompress()
        };

        // Skip document array samples (optional)
        simple_sds::serialize::skip_option(&mut cursor)?;

        // Load metadata
        let metadata = Option::<gbwt::gbwt::Metadata>::load(&mut cursor)?;
        if header.is_set(gbwt::headers::GBWTPayload::FLAG_METADATA) != metadata.is_some() {
            return Err(MappedGbzError::InvalidFormat(
                "GBWT: Invalid metadata flag in the header".to_string()
            ));
        }

        if let Some(meta) = metadata.as_ref() {
            if meta.has_path_names() {
                let expected = if header.is_set(gbwt::headers::GBWTPayload::FLAG_BIDIRECTIONAL) {
                    header.payload().sequences / 2
                } else {
                    header.payload().sequences
                };
                if meta.paths() > 0 && meta.paths() != expected {
                    return Err(MappedGbzError::InvalidFormat(
                        "GBWT: Invalid path count in the metadata".to_string()
                    ));
                }
            }
        }

        Ok(GbwtDescriptor {
            header,
            tags,
            bwt,
            endmarker,
            metadata,
        })
    }

    /// Returns the first position in sequence `id`
    pub fn start(&self, id: usize) -> Option<Pos> {
        if id < self.endmarker.len() && self.endmarker[id].node != ENDMARKER {
            Some(self.endmarker[id])
        } else {
            None
        }
    }

    /// Follows the sequence forward and returns the next position
    /// 
    /// Uses the LF-mapping (Last-to-First) to traverse the BWT.
    /// The Record::lf() method internally finds the correct successor
    /// by iterating through RLE runs at the given offset.
    pub fn forward(&self, map: &MemoryMap, pos: Pos) -> Option<Pos> {
        if pos.node < self.first_node() {
            return None;
        }
        let record = self.bwt.record(map, self.node_to_record(pos.node))?;
        // LF-mapping: finds the successor at this offset in the BWT
        record.lf(pos.offset)
    }

    /// Returns the first node in the alphabet
    #[inline]
    pub fn first_node(&self) -> usize {
        self.header.payload().offset
    }

    /// Converts a node id to a record id
    #[inline]
    fn node_to_record(&self, node: usize) -> usize {
        node - self.first_node()
    }

    /// Returns the metadata
    pub fn metadata(&self) -> Option<&gbwt::gbwt::Metadata> {
        self.metadata.as_ref()
    }

    /// Returns the number of sequences
    #[inline]
    pub fn sequences(&self) -> usize {
        self.header.payload().sequences
    }
}

/// Container that holds both the memory map and the GBWT descriptor
pub struct MappedGBZ {
    map: MemoryMap,
    gbwt: GbwtDescriptor,
}

impl MappedGBZ {
    /// Load a GBZ file with memory-mapped access
    /// 
    /// GBZ format: Header | Tags | GBWT | Graph
    /// We only parse the GBWT portion for path queries
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let map = MemoryMap::new(path, MappingMode::ReadOnly)?;
        
        // Parse GBZ container to find GBWT offset
        let map_slice: &[u64] = map.as_ref();
        let byte_slice: &[u8] = unsafe {
            std::slice::from_raw_parts(
                map_slice.as_ptr() as *const u8,
                map_slice.len() * 8,
            )
        };
        let mut cursor = Cursor::new(byte_slice);
        
        // Load GBZ header
        let gbz_header = gbwt::headers::Header::<gbwt::headers::GBZPayload>::load(&mut cursor)?;
        if let Err(msg) = gbz_header.validate() {
            return Err(MappedGbzError::InvalidFormat(format!("GBZ header: {}", msg)));
        }
        
        // Load and skip tags
        let _tags = gbwt::support::Tags::load(&mut cursor)?;
        
        // Now we're at the GBWT section - parse it with memory mapping
        let gbwt_offset = cursor.position() as usize;
        let gbwt = GbwtDescriptor::parse_from_offset(&map, gbwt_offset)?;
        
        Ok(MappedGBZ { map, gbwt })
    }

    /// Get the GBWT descriptor
    pub fn gbwt(&self) -> &GbwtDescriptor {
        &self.gbwt
    }

    /// Get the memory map
    pub fn map(&self) -> &MemoryMap {
        &self.map
    }

    /// Walk a path starting from a sequence ID
    pub fn walk_path(&self, sequence_id: usize) -> Option<PathWalker<'_>> {
        let start_pos = self.gbwt.start(sequence_id)?;
        Some(PathWalker {
            gbz: self,
            current_pos: Some(start_pos),
        })
    }

    /// Get metadata (compatible with GBZ interface)
    pub fn metadata(&self) -> Option<&gbwt::gbwt::Metadata> {
        self.gbwt.metadata()
    }

    /// Get path iterator (compatible with GBZ interface)
    /// Returns an iterator over (node_id, orientation) pairs
    pub fn path(&self, path_id: usize, _orientation: gbwt::Orientation) -> Option<PathWalker<'_>> {
        // For now, we only support forward orientation
        // The path_id is the sequence_id in GBWT terms
        self.walk_path(path_id)
    }

    /// Get sequence length for a node (stub - needs Graph implementation)
    /// For now, returns None since we don't have Graph loaded
    pub fn sequence_len(&self, _node_id: usize) -> Option<usize> {
        // TODO: Implement Graph descriptor to get node lengths
        // For now, return a default length to allow testing
        Some(100) // Placeholder
    }

    /// Get reference positions (stub - not implemented yet)
    /// Returns empty vec since we don't have the reference position index
    pub fn reference_positions(&self, _sample_interval: usize, _generic: bool) -> Vec<gbwt::gbz::ReferencePath> {
        // TODO: Implement reference position index
        Vec::new()
    }
}

/// Iterator for walking a path in the graph
pub struct PathWalker<'a> {
    gbz: &'a MappedGBZ,
    current_pos: Option<Pos>,
}

impl<'a> Iterator for PathWalker<'a> {
    type Item = (usize, gbwt::Orientation); // (node_id, orientation)

    fn next(&mut self) -> Option<Self::Item> {
        let pos = self.current_pos?;
        
        // Decode the node and orientation
        let (node_id, orientation) = gbwt::support::decode_node(pos.node);
        
        // Advance to next position
        self.current_pos = self.gbz.gbwt.forward(&self.gbz.map, pos);
        
        Some((node_id, orientation))
    }
}
