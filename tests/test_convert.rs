use graphome::convert::*;

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
        use std::collections::HashSet;
    
        // Create a temporary GFA file with 'S' and 'L' lines
        let gfa_content = "\
    H	VN:Z:1.0
    S	0	A
    S	1	B
    S	2	C
    L	A	+	B	+	*
    L	A	+	C	+	*
    L	B	+	C	+	*
    P	0_path	A+,B+,C+
    ";
        let temp_gfa = tempfile::NamedTempFile::new().unwrap();
        write!(temp_gfa.as_ref(), "{}", gfa_content).unwrap();
    
        // Output .gam file
        let temp_gam = tempfile::NamedTempFile::new().unwrap();
    
        // Run the conversion
        convert::convert_gfa_to_edge_list(
            temp_gfa.path(),
            temp_gam.path(),
        ).unwrap();
    
        // Read the .gam file
        let edges = read_gam_file(temp_gam.path()).unwrap();
    
        // Define expected edges
        let expected_edges = vec![
            (0, 1),
            (0, 2),
            (1, 2),
        ];
        let expected_set: HashSet<(u32, u32)> = expected_edges.into_iter().collect();
        let actual_set: HashSet<(u32, u32)> = edges.into_iter().collect();
    
        assert_eq!(actual_set, expected_set, "Edges do not match expected values.");
    }

    /// Test 1.2: Is GFA file with duplicate segments converted correctly without data loss?
    #[test]
    fn test_gfa_to_adjacency_matrix_with_duplicate_segments() {
        use std::collections::HashSet;
    
        // Create a temporary GFA file with duplicate 'L' lines
        let gfa_content = "\
    H	VN:Z:1.0
    S	0	A
    S	1	B
    S	2	C
    L	A	+	B	+	*
    L	B	+	C	+	*
    L	A	+	B	+	*
    P	0_path	A+,B+,C+
    ";
        let temp_gfa = tempfile::NamedTempFile::new().unwrap();
        write!(temp_gfa.as_ref(), "{}", gfa_content).unwrap();
    
        // Output .gam file
        let temp_gam = tempfile::NamedTempFile::new().unwrap();
    
        // Run the conversion
        convert::convert_gfa_to_edge_list(
            temp_gfa.path(),
            temp_gam.path(),
        ).unwrap();
    
        // Read the .gam file
        let edges = read_gam_file(temp_gam.path()).unwrap();
    
        // Define expected edges
        let expected_edges = vec![
            (0, 1),
            (0, 1),
            (1, 2),
        ];
        let expected_set: HashSet<(u32, u32)> = expected_edges.into_iter().collect();
        let actual_set: HashSet<(u32, u32)> = edges.into_iter().collect();
    
        assert_eq!(actual_set, expected_set, "Edges do not match expected values.");
    }


    /// Test 2.1: Can the unique indices be mapped back to GFA segments correctly?
    #[test]
    fn test_unique_indices_mapping() {
        use std::collections::HashSet;
    
        // Create a temporary GFA file with 'S' and 'L' lines
        let gfa_content = "\
    H	VN:Z:1.0
    S	0	A
    S	1	B
    S	2	C
    L	A	+	B	+	*
    L	B	+	C	+	*
    P	0_path	A+,B+,C+
    ";
        let temp_gfa = tempfile::NamedTempFile::new().unwrap();
        write!(temp_gfa.as_ref(), "{}", gfa_content).unwrap();
    
        // Output .gam file
        let temp_gam = tempfile::NamedTempFile::new().unwrap();
    
        // Run the conversion
        convert::convert_gfa_to_edge_list(
            temp_gfa.path(),
            temp_gam.path(),
        ).unwrap();
    
        // Read the .gam file
        let edges = read_gam_file(temp_gam.path()).unwrap();
    
        // Define expected edges
        let expected_edges = vec![
            (0, 1),
            (1, 2),
        ];
        let expected_set: HashSet<(u32, u32)> = expected_edges.into_iter().collect();
        let actual_set: HashSet<(u32, u32)> = edges.into_iter().collect();
    
        assert_eq!(actual_set, expected_set, "Edges do not match expected values.");
    }


    /// Test 2.2: Can the unique indices handle segment names with special characters?
    #[test]
    fn test_unique_indices_mapping_with_special_characters() {
        use std::collections::HashSet;
    
        // Create a temporary GFA file with 'S' and 'L' lines including special characters
        let gfa_content = "\
    H	VN:Z:1.0
    S	0	A!
    S	1	B@
    S	2	C#
    L	A!	+	B@	+	*
    L	B@	+	C#	+	*
    L	A!	+	C#	+	*
    P	0_path	A+,B@+,C#+
    ";
        let temp_gfa = tempfile::NamedTempFile::new().unwrap();
        write!(temp_gfa.as_ref(), "{}", gfa_content).unwrap();
    
        // Output .gam file
        let temp_gam = tempfile::NamedTempFile::new().unwrap();
    
        // Run the conversion
        convert::convert_gfa_to_edge_list(
            temp_gfa.path(),
            temp_gam.path(),
        ).unwrap();
    
        // Read the .gam file
        let edges = read_gam_file(temp_gam.path()).unwrap();
    
        // Define expected edges
        let expected_edges = vec![
            (0, 1),
            (1, 2),
            (0, 2),
        ];
        let expected_set: HashSet<(u32, u32)> = expected_edges.into_iter().collect();
        let actual_set: HashSet<(u32, u32)> = edges.into_iter().collect();
    
        assert_eq!(actual_set, expected_set, "Edges do not match expected values.");
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
        use std::collections::HashSet;
    
        // Create a temporary GFA file with 'S' and 'L' lines defining symmetric edges
        let gfa_content = "\
    H	VN:Z:1.0
    S	0	A
    S	1	B
    S	2	C
    S	3	D
    L	A	+	B	+	*
    L	B	+	A	+	*
    L	B	+	C	+	*
    L	C	+	B	+	*
    L	C	+	D	+	*
    L	D	+	C	+	*
    P	0_path	A+,B+,C+,D+
    ";
        let temp_gfa = tempfile::NamedTempFile::new().unwrap();
        write!(temp_gfa.as_ref(), "{}", gfa_content).unwrap();
    
        // Output .gam file
        let temp_gam = tempfile::NamedTempFile::new().unwrap();
    
        // Run the conversion
        convert::convert_gfa_to_edge_list(
            temp_gfa.path(),
            temp_gam.path(),
        ).unwrap();
    
        // Read the .gam file
        let edges = read_gam_file(temp_gam.path()).unwrap();
    
        // Define expected symmetric edges
        let expected_edges = vec![
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 2),
        ];
        let expected_set: HashSet<(u32, u32)> = expected_edges.into_iter().collect();
        let actual_set: HashSet<(u32, u32)> = edges.into_iter().collect();
    
        assert_eq!(actual_set, expected_set, "Adjacency matrix is not symmetric.");
    }
}
