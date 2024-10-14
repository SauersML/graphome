// tests/test_convert.rs

use graphome::convert;
use graphome::convert::*;

use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::collections::HashSet;
    use std::fs::File;
    use std::io::{Read, Write};
    use tempfile::NamedTempFile;

    /// Helper function to read edges from a binary .gam file.
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
        use std::collections::HashSet;
        use std::io::Write;
        use tempfile::NamedTempFile;

        // GFA content
        let gfa_content = "\
    H\tVN:Z:1.0
    S\tA\tACCTT
    S\tB\tTCAAGG
    S\tC\tCTTGATT
    L\tA\t+\tB\t+\t*
    L\tB\t+\tA\t+\t*
    L\tA\t+\tC\t+\t*
    L\tC\t+\tA\t+\t*
    L\tB\t+\tC\t+\t*
    L\tC\t+\tB\t+\t*
    P\tbasic_path\tA+,B+,C+
        ";

        // Create temporary GFA file
        let mut temp_gfa = NamedTempFile::new().unwrap();
        write!(temp_gfa.as_file_mut(), "{}", gfa_content).unwrap();

        // Create temporary output .gam file
        let temp_gam = NamedTempFile::new().unwrap();

        // Run the conversion
        convert::convert_gfa_to_edge_list(temp_gfa.path(), temp_gam.path()).unwrap();

        // Read the .gam file
        let edges = read_edges_from_bin(temp_gam.path()).unwrap();

        // Define expected edges
        let expected_edges = vec![
            (0, 1), // A -> B
            (1, 0), // B -> A
            (0, 2), // A -> C
            (2, 0), // C -> A
            (1, 2), // B -> C
            (2, 1), // C -> B
        ];
        let expected_set: HashSet<(u32, u32)> = expected_edges.into_iter().collect();
        let actual_set: HashSet<(u32, u32)> = edges.into_iter().collect();

        assert_eq!(
            actual_set, expected_set,
            "Edges do not match expected values."
        );
    }

    /// Test 1.2: Is GFA file with duplicate segments converted correctly without data loss?
    #[test]
    fn test_gfa_to_adjacency_matrix_with_duplicate_segments() {
        use std::collections::HashSet;
        use std::io::Write;
        use tempfile::NamedTempFile;

        // GFA content
        let gfa_content = "\
    H\tVN:Z:1.0
    S\tA\tACCTT
    S\tB\tTCAAGG
    S\tC\tCTTGATT
    L\tA\t+\tB\t+\t*
    L\tB\t+\tA\t+\t*
    L\tA\t+\tC\t+\t*
    L\tC\t+\tA\t+\t*
    L\tB\t+\tC\t+\t*
    L\tC\t+\tB\t+\t*
    P\t0_path\tA+,B+,C+
        ";

        // Create temporary GFA file
        let mut temp_gfa = NamedTempFile::new().unwrap();
        write!(temp_gfa.as_file_mut(), "{}", gfa_content).unwrap();

        // Create temporary output .gam file
        let temp_gam = NamedTempFile::new().unwrap();

        // Run the conversion
        convert::convert_gfa_to_edge_list(temp_gfa.path(), temp_gam.path()).unwrap();

        // Read the .gam file
        let edges = read_edges_from_bin(temp_gam.path()).unwrap();

        // Define expected edges (both directions)
        let expected_edges = vec![
            (0, 1), // A -> B
            (1, 0), // B -> A
            (0, 2), // A -> C
            (2, 0), // C -> A
            (1, 2), // B -> C
            (2, 1), // C -> B
        ];
        let expected_set: HashSet<(u32, u32)> = expected_edges.into_iter().collect();
        let actual_set: HashSet<(u32, u32)> = edges.into_iter().collect();

        assert_eq!(
            actual_set, expected_set,
            "Edges do not match expected values."
        );
    }

    /// Test 2.1: Can the unique indices be mapped back to GFA segments correctly?
    #[test]
    fn test_unique_indices_mapping() {
        use std::collections::HashSet;
        use std::io::Write;
        use tempfile::NamedTempFile;

        // GFA content
        let gfa_content = "\
    H\tVN:Z:1.0
    S\tA\tACCTT
    S\tB\tTCAAGG
    S\tC\tCTTGATT
    L\tA\t+\tB\t+\t*
    L\tB\t+\tA\t+\t*
    L\tB\t+\tC\t+\t*
    L\tC\t+\tB\t+\t*
    P\tunique_path\tA+,B+,C+
        ";

        // Create temporary GFA file
        let mut temp_gfa = NamedTempFile::new().unwrap();
        write!(temp_gfa.as_file_mut(), "{}", gfa_content).unwrap();

        // Create temporary output .gam file
        let temp_gam = NamedTempFile::new().unwrap();

        // Run the conversion
        convert::convert_gfa_to_edge_list(temp_gfa.path(), temp_gam.path()).unwrap();

        // Read the .gam file
        let edges = read_edges_from_bin(temp_gam.path()).unwrap();

        // Define expected edges
        let expected_edges = vec![
            (0, 1), // A -> B
            (1, 0), // B -> A
            (1, 2), // B -> C
            (2, 1), // C -> B
        ];
        let expected_set: HashSet<(u32, u32)> = expected_edges.into_iter().collect();
        let actual_set: HashSet<(u32, u32)> = edges.into_iter().collect();

        assert_eq!(
            actual_set, expected_set,
            "Edges do not match expected values."
        );
    }

    /// Test 2.2: Can the unique indices handle segment names with special characters?
    #[test]
    fn test_unique_indices_mapping_with_special_characters() {
        use std::collections::HashSet;
        use std::io::Write;
        use tempfile::NamedTempFile;

        // GFA content
        let gfa_content = "\
    H\tVN:Z:1.0
    S\tA\tACCTT
    S\tB\tTCAAGG
    S\tC\tCTTGATT
    L\tA\t+\tB\t+\t*
    L\tB\t+\tA\t+\t*
    L\tA\t+\tC\t+\t*
    L\tC\t+\tA\t+\t*
    L\tB\t+\tC\t+\t*
    L\tC\t+\tB\t+\t*
    P\tspecial_chars_path\tA+,B+,C+
        ";

        // Create temporary GFA file
        let mut temp_gfa = NamedTempFile::new().unwrap();
        write!(temp_gfa.as_file_mut(), "{}", gfa_content).unwrap();

        // Create temporary output .gam file
        let temp_gam = NamedTempFile::new().unwrap();

        // Run the conversion
        convert::convert_gfa_to_edge_list(temp_gfa.path(), temp_gam.path()).unwrap();

        // Read the .gam file
        let edges = read_edges_from_bin(temp_gam.path()).unwrap();

        // Define expected edges
        let expected_edges = vec![
            (0, 1), // A -> B
            (1, 0), // B -> A
            (0, 2), // A -> C
            (2, 0), // C -> A
            (1, 2), // B -> C
            (2, 1), // C -> B
        ];
        let expected_set: HashSet<(u32, u32)> = expected_edges.into_iter().collect();
        let actual_set: HashSet<(u32, u32)> = edges.into_iter().collect();

        assert_eq!(
            actual_set, expected_set,
            "Edges do not match expected values."
        );
    }

    /// Test 3.1: Are the edges written to the output file fully correct with respect to the original GFA file?
    #[test]
    fn test_edges_written_correctly() {
        use std::collections::HashSet;
        use std::io::Write;
        use tempfile::NamedTempFile;

        // GFA content
        let gfa_content = "\
    H\tVN:Z:1.0
    S\tA\tACCTT
    S\tB\tCGTA
    S\tC\tGTAC
    S\tD\tTGCA
    L\tA\t+\tB\t+\t*
    L\tB\t+\tA\t+\t*
    L\tA\t+\tC\t+\t*
    L\tC\t+\tA\t+\t*
    L\tC\t+\tD\t+\t*
    L\tD\t+\tC\t+\t*
    L\tB\t+\tD\t+\t*
    L\tD\t+\tB\t+\t*
    P\tedges_path\tA+,B+,C+,D+
        ";

        // Create temporary GFA file
        let mut temp_gfa = NamedTempFile::new().unwrap();
        write!(temp_gfa.as_file_mut(), "{}", gfa_content).unwrap();

        // Create temporary output .gam file
        let temp_gam = NamedTempFile::new().unwrap();

        // Run the conversion
        convert::convert_gfa_to_edge_list(temp_gfa.path(), temp_gam.path()).unwrap();

        // Read the .gam file
        let edges = read_edges_from_bin(temp_gam.path()).unwrap();

        // Define expected edges
        let expected_edges = vec![
            (0, 1), // A -> B
            (1, 0), // B -> A
            (0, 2), // A -> C
            (2, 0), // C -> A
            (2, 3), // C -> D
            (3, 2), // D -> C
            (1, 3), // B -> D
            (3, 1), // D -> B
        ];
        let expected_set: HashSet<(u32, u32)> = expected_edges.into_iter().collect();
        let actual_set: HashSet<(u32, u32)> = edges.into_iter().collect();

        assert_eq!(
            actual_set, expected_set,
            "Edges do not match expected values."
        );
    }

    /// Test 3.2: Handling non-existent segments in links gracefully.
    #[test]
    fn test_edges_with_nonexistent_segments() {
        // Setup: Create a mock GFA file with a link referencing a non-existent segment
        let mut gfa = NamedTempFile::new().expect("Failed to create temporary GFA file");
        write!(gfa.as_file_mut(), "H\tVN:Z:1.0\n").unwrap();
        write!(gfa.as_file_mut(), "S\tA\tACGT\n").unwrap();
        write!(gfa.as_file_mut(), "S\tB\tCGTA\n").unwrap();
        // Segment "D" does not exist
        write!(gfa.as_file_mut(), "L\tA\t+\tB\t+\t*\n").unwrap();
        write!(gfa.as_file_mut(), "L\tB\t+\tD\t+\t*\n").unwrap(); // "D" is nonexistent

        // Define output path
        let temp_gam = NamedTempFile::new().unwrap();

        // Execute conversion
        convert_gfa_to_edge_list(gfa.path(), temp_gam.path()).expect("Conversion failed");

        // Read edges
        let edges = read_edges_from_bin(temp_gam.path()).unwrap();

        // Expected mapping:
        // "A" -> 0
        // "B" -> 1
        // "D" does not exist, so only one edge should be present

        let expected_edges = vec![
            (0, 1), // "A" -> "B"
        ];

        assert_eq!(edges, expected_edges, "Edges do not match expected values.");
    }

    /// Test 4.1: Is the adjacency matrix symmetric, ensuring bidirectional connections?
    #[test]
    fn test_adjacency_matrix_symmetry() {
        use std::collections::HashSet;
        use std::io::Write;
        use tempfile::NamedTempFile;

        // GFA content
        let gfa_content = "\
    H\tVN:Z:1.0
    S\tA\tACCTT
    S\tB\tCGTA
    S\tC\tGTAC
    S\tD\tTGCA
    L\tA\t+\tB\t+\t*
    L\tB\t+\tA\t+\t*
    L\tB\t+\tC\t+\t*
    L\tC\t+\tB\t+\t*
    L\tC\t+\tD\t+\t*
    L\tD\t+\tC\t+\t*
    L\tA\t+\tD\t+\t*
    L\tD\t+\tA\t+\t*
    P\tsymmetry_path\tA+,B+,C+,D+
        ";

        // Create temporary GFA file
        let mut temp_gfa = NamedTempFile::new().unwrap();
        write!(temp_gfa.as_file_mut(), "{}", gfa_content).unwrap();

        // Create temporary output .gam file
        let temp_gam = NamedTempFile::new().unwrap();

        // Run the conversion
        convert::convert_gfa_to_edge_list(temp_gfa.path(), temp_gam.path()).unwrap();

        // Read the .gam file
        let edges = read_edges_from_bin(temp_gam.path()).unwrap();

        // Define expected edges (both directions)
        let expected_edges = vec![
            (0, 1), // A -> B
            (1, 0), // B -> A
            (1, 2), // B -> C
            (2, 1), // C -> B
            (2, 3), // C -> D
            (3, 2), // D -> C
            (0, 3), // A -> D
            (3, 0), // D -> A
        ];
        let expected_set: HashSet<(u32, u32)> = expected_edges.into_iter().collect();
        let actual_set: HashSet<(u32, u32)> = edges.into_iter().collect();

        assert_eq!(
            actual_set, expected_set,
            "Adjacency matrix is not symmetric."
        );
    }
}
