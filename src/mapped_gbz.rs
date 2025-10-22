// mapped_gbz.rs
// Memory-mapped GBZ structures for efficient coordinate queries without loading entire file

use gbwt::{Pos, ENDMARKER};
use simple_sds::ops::{BitVec, Select};
use simple_sds::serialize::{MemoryMap, MappingMode, Serialize};
use simple_sds::sparse_vector::SparseVector;
use std::io::{self, Cursor, Error, ErrorKind};
use std::path::Path;

/// Memory-mapped BWT structure that avoids loading the entire data array into heap
pub struct MappedBWT<'a> {
    /// Record boundary index (loaded into memory, relatively small)
    index: SparseVector,
    /// BWT data (memory-mapped, large)
    data: &'a [u8],
}

impl<'a> MappedBWT<'a> {
    /// Returns the number of records in the BWT
    #[inline]
    pub fn len(&self) -> usize {
        self.index.count_ones()
    }

    /// Returns true if the BWT is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the byte slice for the i-th record
    fn record_bytes(&self, i: usize) -> &[u8] {
        let mut iter = self.index.select_iter(i);
        let (_, start) = iter.next().unwrap();
        let limit = if i + 1 < self.len() {
            iter.next().unwrap().1
        } else {
            self.data.len()
        };
        &self.data[start..limit]
    }

    /// Returns the i-th record, or None if there is no such node
    pub fn record(&self, i: usize) -> Option<gbwt::bwt::Record<'_>> {
        if i >= self.len() {
            return None;
        }
        let bytes = self.record_bytes(i);
        gbwt::bwt::Record::new(i, bytes)
    }
}

/// Memory-mapped GBWT structure
pub struct MappedGBWT<'a> {
    /// GBWT header (loaded)
    header: gbwt::headers::Header<gbwt::headers::GBWTPayload>,
    /// Tags (loaded, small)
    tags: gbwt::support::Tags,
    /// Memory-mapped BWT
    bwt: MappedBWT<'a>,
    /// Endmarker array (loaded, small - one per sequence)
    endmarker: Vec<Pos>,
    /// Metadata (loaded, small)
    metadata: Option<gbwt::gbwt::Metadata>,
}

impl<'a> MappedGBWT<'a> {
    /// Load a GBWT from a memory-mapped file
    /// 
    /// This loads small structures (headers, metadata, endmarker) into memory
    /// but keeps the large BWT data memory-mapped
    pub fn new(map: &'a MemoryMap) -> io::Result<Self> {
        // Create a cursor over the memory map to read sequentially
        let map_slice: &[u64] = map.as_ref();
        let byte_slice: &[u8] = unsafe {
            std::slice::from_raw_parts(
                map_slice.as_ptr() as *const u8,
                map_slice.len() * 8,
            )
        };
        let mut cursor = Cursor::new(byte_slice);

        // Load GBWT header
        let header = gbwt::headers::Header::<gbwt::headers::GBWTPayload>::load(&mut cursor)?;
        if let Err(msg) = header.validate() {
            return Err(Error::new(ErrorKind::InvalidData, msg));
        }

        // Load tags
        let mut tags = gbwt::support::Tags::load(&mut cursor)?;
        tags.insert("source", "mapped-gbwt");

        // Load BWT index (SparseVector - relatively small)
        let bwt_index = SparseVector::load(&mut cursor)?;

        // Get the position where BWT data starts
        let bwt_data_start = cursor.position() as usize;

        // Load the size of the BWT data array
        let data_size = {
            let size_bytes = &byte_slice[bwt_data_start..bwt_data_start + 8];
            usize::from_le_bytes(size_bytes.try_into().unwrap())
        };

        // Create a reference to the BWT data (memory-mapped, not copied!)
        let data_start = bwt_data_start + 8; // Skip the size field
        let data_end = data_start + data_size;
        let bwt_data = &byte_slice[data_start..data_end];

        // Verify index/data length match
        if bwt_index.len() != bwt_data.len() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "BWT: Index / data length mismatch",
            ));
        }

        let bwt = MappedBWT {
            index: bwt_index,
            data: bwt_data,
        };

        // Advance cursor past the BWT data (with padding)
        let padded_len = ((data_size + 7) / 8) * 8;
        cursor.set_position((data_end + (padded_len - data_size)) as u64);

        // Decompress the endmarker
        let endmarker = if bwt.is_empty() {
            Vec::new()
        } else {
            bwt.record(ENDMARKER)
                .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Missing endmarker record"))?
                .decompress()
        };

        // Skip document array samples (optional)
        simple_sds::serialize::skip_option(&mut cursor)?;

        // Load metadata
        let metadata = Option::<gbwt::gbwt::Metadata>::load(&mut cursor)?;
        if header.is_set(gbwt::headers::GBWTPayload::FLAG_METADATA) != metadata.is_some() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "GBWT: Invalid metadata flag in the header",
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
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        "GBWT: Invalid path count in the metadata",
                    ));
                }
            }
        }

        Ok(MappedGBWT {
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
    pub fn forward(&self, pos: Pos) -> Option<Pos> {
        if pos.node < self.first_node() {
            return None;
        }
        let record = self.bwt.record(self.node_to_record(pos.node))?;
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

/// Container that holds both the memory map and the mapped GBWT
/// The map must outlive the GBWT since GBWT contains references into it
pub struct MappedGBZ {
    _map: MemoryMap,
    // Note: We can't actually store MappedGBWT here due to self-referential struct issues
    // This is a known Rust limitation - we'll need a different approach
}

// For now, let's just provide a function that loads the GBWT
// The caller must keep the MemoryMap alive
pub fn load_mapped_gbwt_from_gbz<P: AsRef<Path>>(path: P) -> io::Result<MemoryMap> {
    MemoryMap::new(path, MappingMode::ReadOnly)
}
