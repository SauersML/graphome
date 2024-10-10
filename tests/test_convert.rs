use crate::convert::*;

use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write, BufRead};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::{Write, Read};
    use std::fs::File;
    use std::collections::HashMap;

    /// Helper function to read edges from a binary file.
    fn read_edges_from_bin<P: AsRef<Path>>(path: P) -> io::Result<Vec<(u32, u32)>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut buffer = [0u8; 8];
        let mut edges = Vec::new();

        while let Ok(_) = reader.read_exact(&mut buffer) {
            let from = u32::from_le_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
            let to = u32::from_le_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]);
            edges.push((from, to));
        }

        Ok(edges)
    }

    /// Test 1.1: Is GFA file to an adjacency matrix in edge list format without data loss and correctly?
    #[test]
    fn test_gfa_to_adjacency_matrix_basic_conversion() {
        // Setup: Create a mock GFA file
        let mut gfa = NamedTempFile::new().expect("Failed to create temporary GFA file");
        writeln!(gfa, "H\tVN:Z:1.0").unwrap();
        writeln!(gfa, "S\t1\tACGT").unwrap();
        writeln!(gfa, "S\t2\tCGTA").unwrap();
        writeln!(gfa, "S\t3\tGTAC").unwrap();
        writeln!(gfa, "L\t1\t+\t2\t-\t100M").unwrap();
        writeln!(gfa, "L\t2\t-\t3\t+\t100M").unwrap();
        writeln!(gfa, "L\t1\t+\t3\t+\t100M").unwrap();

        // Define output path
        let output = NamedTempFile::new().expect("Failed to create temporary output file");

        // Execute conversion
        convert_gfa_to_edge_list(gfa.path(), output.path()).expect("Conversion failed");

        // Verification: Load and check adjacency_matrix.bin
        let edges = read_edges_from_bin(output.path()).expect("Failed to read edges from binary file");

        let expected_edges = vec![
            (0, 1), // Segment "1" -> "2"
            (1, 2), // Segment "2" -> "3"
            (0, 2), // Segment "1" -> "3"
        ];

        assert_eq!(edges, expected_edges);
    }

    /// Test 1.2: Is GFA file with duplicate segments converted correctly without data loss?
    #[test]
    fn test_gfa_to_adjacency_matrix_with_duplicate_segments() {
        // Setup: Create a mock GFA file with duplicate segments
        let mut gfa = NamedTempFile::new().expect("Failed to create temporary GFA file");
        writeln!(gfa, "H\tVN:Z:1.0").unwrap();
        writeln!(gfa, "S\t1\tACGT").unwrap();
        writeln!(gfa, "S\t2\tCGTA").unwrap();
        writeln!(gfa, "S\t2\tCGTA").unwrap(); // Duplicate segment "2"
        writeln!(gfa, "S\t3\tGTAC").unwrap();
        writeln!(gfa, "L\t1\t+\t2\t-\t100M").unwrap();
        writeln!(gfa, "L\t1\t+\t2\t-\t100M").unwrap(); // Duplicate link
        writeln!(gfa, "L\t2\t-\t3\t+\t100M").unwrap();

        // Define output path
        let output = NamedTempFile::new().expect("Failed to create temporary output file");

        // Execute conversion
        convert_gfa_to_edge_list(gfa.path(), output.path()).expect("Conversion failed");

        // Verification: Load and check adjacency_matrix.bin
        let edges = read_edges_from_bin(output.path()).expect("Failed to read edges from binary file");

        // Expected mapping:
        // "1" -> 0
        // "2" -> 1
        // "3" -> 2
        // Duplicate segment "2" should not be assigned a new index

        let expected_edges = vec![
            (0, 1), // "1" -> "2"
            (0, 1), // Duplicate "1" -> "2"
            (1, 2), // "2" -> "3"
        ];

        assert_eq!(edges, expected_edges);
    }

    /// Test 2.1: Can the unique indices be mapped back to GFA segments correctly?
    #[test]
    fn test_unique_indices_mapping() {
        // Setup: Create a mock GFA file
        let mut gfa = NamedTempFile::new().expect("Failed to create temporary GFA file");
        writeln!(gfa, "H\tVN:Z:1.0").unwrap();
        writeln!(gfa, "S\tA\tACGT").unwrap();
        writeln!(gfa, "S\tB\tCGTA").unwrap();
        writeln!(gfa, "S\tC\tGTAC").unwrap();

        // Define output path
        let output = NamedTempFile::new().expect("Failed to create temporary output file");

        // Execute conversion
        convert_gfa_to_edge_list(gfa.path(), output.path()).expect("Conversion failed");

        // Since parse_segments is internal, we'll read the edge list and infer the mapping
        let edges = read_edges_from_bin(output.path()).expect("Failed to read edges from binary file");

        // Expected mapping:
        // "A" -> 0
        // "B" -> 1
        // "C" -> 2

        // Verify edges are as expected
        let expected_edges = vec![
            (0, 1), // "A" -> "B"
            (1, 2), // "B" -> "C"
            // No "A" -> "C" link
        ];

        assert_eq!(edges, expected_edges);
    }

    /// Test 2.2: Can the unique indices handle segment names with special characters?
    #[test]
    fn test_unique_indices_mapping_with_special_characters() {
        // Setup: Create a mock GFA file with special characters in segment names
        let mut gfa = NamedTempFile::new().expect("Failed to create temporary GFA file");
        writeln!(gfa, "H\tVN:Z:1.0").unwrap();
        writeln!(gfa, "S\tA-1\tACGT").unwrap();
        writeln!(gfa, "S\tB_2\tCGTA").unwrap();
        writeln!(gfa, "S\tC+3\tGTAC").unwrap();

        // Define output path
        let output = NamedTempFile::new().expect("Failed to create temporary output file");

        // Execute conversion
        convert_gfa_to_edge_list(gfa.path(), output.path()).expect("Conversion failed");

        // Read edges
        let edges = read_edges_from_bin(output.path()).expect("Failed to read edges from binary file");

        // Expected mapping:
        // "A-1" -> 0
        // "B_2" -> 1
        // "C+3" -> 2

        // Expected edges:
        // "A-1" -> "B_2"
        // "B_2" -> "C+3"
        // "A-1" -> "C+3"

        let expected_edges = vec![
            (0, 1), // "A-1" -> "B_2"
            (1, 2), // "B_2" -> "C+3"
            (0, 2), // "A-1" -> "C+3"
        ];

        assert_eq!(edges, expected_edges);
    }

    /// Test 3.1: Are the edges written to the output file fully correct with respect to the original GFA file?
    #[test]
    fn test_edges_written_correctly() {
        // Setup: Create a mock GFA file with multiple links
        let mut gfa = NamedTempFile::new().expect("Failed to create temporary GFA file");
        writeln!(gfa, "H\tVN:Z:1.0").unwrap();
        writeln!(gfa, "S\tA\tACGT").unwrap();
        writeln!(gfa, "S\tB\tCGTA").unwrap();
        writeln!(gfa, "S\tC\tGTAC").unwrap();
        writeln!(gfa, "S\tD\tTGCA").unwrap();
        writeln!(gfa, "L\tA\t+\tB\t-\t100M").unwrap();
        writeln!(gfa, "L\tB\t-\tC\t+\t100M").unwrap();
        writeln!(gfa, "L\tC\t+\tD\t-\t100M").unwrap();
        writeln!(gfa, "L\tA\t+\tD\t+\t100M").unwrap();

        // Define output path
        let output = NamedTempFile::new().expect("Failed to create temporary output file");

        // Execute conversion
        convert_gfa_to_edge_list(gfa.path(), output.path()).expect("Conversion failed");

        // Read edges
        let edges = read_edges_from_bin(output.path()).expect("Failed to read edges from binary file");

        // Expected mapping:
        // "A" -> 0
        // "B" -> 1
        // "C" -> 2
        // "D" -> 3

        let expected_edges = vec![
            (0, 1), // "A" -> "B"
            (1, 2), // "B" -> "C"
            (2, 3), // "C" -> "D"
            (0, 3), // "A" -> "D"
        ];

        assert_eq!(edges, expected_edges);
    }

    /// Test 3.2: Handling non-existent segments in links gracefully.
    #[test]
    fn test_edges_with_nonexistent_segments() {
        // Setup: Create a mock GFA file with a link referencing a non-existent segment
        let mut gfa = NamedTempFile::new().expect("Failed to create temporary GFA file");
        writeln!(gfa, "H\tVN:Z:1.0").unwrap();
        writeln!(gfa, "S\tA\tACGT").unwrap();
        writeln!(gfa, "S\tB\tCGTA").unwrap();
        // Segment "D" does not exist
        writeln!(gfa, "L\tA\t+\tB\t-\t100M").unwrap();
        writeln!(gfa, "L\tB\t-\tD\t+\t100M").unwrap(); // "D" is nonexistent

        // Define output path
        let output = NamedTempFile::new().expect("Failed to create temporary output file");

        // Execute conversion
        convert_gfa_to_edge_list(gfa.path(), output.path()).expect("Conversion failed");

        // Read edges
        let edges = read_edges_from_bin(output.path()).expect("Failed to read edges from binary file");

        // Expected mapping:
        // "A" -> 0
        // "B" -> 1
        // "D" does not exist, so only one edge should be present

        let expected_edges = vec![
            (0, 1), // "A" -> "B"
        ];

        assert_eq!(edges, expected_edges);
    }

    /// Test 4.1: Is the adjacency matrix symmetric, ensuring bidirectional connections?
    #[test]
    fn test_adjacency_matrix_symmetry() {
        // Setup: Create a mock GFA file with bidirectional links
        let mut gfa = NamedTempFile::new().expect("Failed to create temporary GFA file");
        writeln!(gfa, "H\tVN:Z:1.0").unwrap();
        writeln!(gfa, "S\t1\tACGT").unwrap();
        writeln!(gfa, "S\t2\tCGTA").unwrap();
        writeln!(gfa, "S\t3\tGTAC").unwrap();
        writeln!(gfa, "S\t4\tTGCA").unwrap();
        writeln!(gfa, "L\t1\t+\t2\t-\t100M").unwrap();
        writeln!(gfa, "L\t2\t-\t1\t+\t100M").unwrap(); // Reciprocal link
        writeln!(gfa, "L\t2\t-\t3\t+\t100M").unwrap();
        writeln!(gfa, "L\t3\t+\t2\t-\t100M").unwrap(); // Reciprocal link
        writeln!(gfa, "L\t3\t+\t4\t-\t100M").unwrap();
        writeln!(gfa, "L\t4\t-\t3\t+\t100M").unwrap(); // Reciprocal link

        // Define output path
        let output = NamedTempFile::new().expect("Failed to create temporary output file");

        // Execute conversion
        convert_gfa_to_edge_list(gfa.path(), output.path()).expect("Conversion failed");

        // Read edges
        let edges = read_edges_from_bin(output.path()).expect("Failed to read edges from binary file");

        // Expected mapping:
        // "1" -> 0
        // "2" -> 1
        // "3" -> 2
        // "4" -> 3

        let expected_edges = vec![
            (0, 1), // "1" -> "2"
            (1, 0), // "2" -> "1"
            (1, 2), // "2" -> "3"
            (2, 1), // "3" -> "2"
            (2, 3), // "3" -> "4"
            (3, 2), // "4" -> "3"
        ];

        assert_eq!(edges, expected_edges);

        // Additionally, verify symmetry
        let mut adjacency_map: HashMap<(u32, u32), bool> = HashMap::new();
        for &(from, to) in &edges {
            adjacency_map.insert((from, to), true);
        }

        for &(from, to) in &edges {
            assert!(
                adjacency_map.get(&(to, from)).is_some(),
                "Adjacency matrix is not symmetric: Missing reciprocal link for ({}, {})",
                from,
                to
            );
        }
    }
}
