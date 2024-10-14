// tests/test_convert.rs

use std::collections::HashSet;
use std::fs::File;
use std::io::{self, Read, Write};
use tempfile::NamedTempFile;

use graphome::convert::convert_gfa_to_edge_list;

/// Helper function to read edges from the binary edge list file
fn read_edges_from_file(path: &std::path::Path) -> io::Result<HashSet<(u32, u32)>> {
    let mut edges = HashSet::new();
    let mut file = File::open(path)?;
    let mut buffer = [0u8; 8];

    while let Ok(_) = file.read_exact(&mut buffer) {
        let from = u32::from_le_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
        let to = u32::from_le_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]);
        edges.insert((from, to));
    }

    Ok(edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that the adjacency matrix is symmetric after conversion.
    #[test]
    fn test_adjacency_matrix_symmetry() -> io::Result<()> {
        // Create a temporary GFA file with sample data
        let mut gfa_file = NamedTempFile::new()?;
        writeln!(gfa_file, "H\tVN:Z:1.0")?;
        writeln!(gfa_file, "S\tA\t*")?;
        writeln!(gfa_file, "S\tB\t*")?;
        writeln!(gfa_file, "S\tC\t*")?;
        writeln!(gfa_file, "S\tD\t*")?;
        writeln!(gfa_file, "L\tA\t+\tB\t+\t0M")?;
        writeln!(gfa_file, "L\tB\t+\tC\t+\t0M")?;
        writeln!(gfa_file, "L\tC\t+\tD\t+\t0M")?;
        writeln!(gfa_file, "L\tA\t+\tD\t+\t0M")?;

        // Output file for adjacency matrix
        let output_file = NamedTempFile::new()?;

        // Run the conversion
        convert_gfa_to_edge_list(gfa_file.path(), output_file.path())?;

        // Read the output and verify the edges
        let edges = read_edges_from_file(output_file.path())?;
        let expected_edges = HashSet::from([
            (0, 1), (1, 0),
            (1, 2), (2, 1),
            (2, 3), (3, 2),
            (0, 3), (3, 0),
        ]);

        assert_eq!(
            edges, expected_edges,
            "Adjacency matrix is not symmetric."
        );

        Ok(())
    }

    /// Test that unique indices are correctly mapped even with special characters in segment names.
    #[test]
    fn test_unique_indices_mapping_with_special_characters() -> io::Result<()> {
        // Create a temporary GFA file with segments having special characters
        let mut gfa_file = NamedTempFile::new()?;
        writeln!(gfa_file, "H\tVN:Z:1.0")?;
        writeln!(gfa_file, "S\tseg1#@!\t*")?;
        writeln!(gfa_file, "S\tseg2$%^&\t*")?;
        writeln!(gfa_file, "L\tseg1#@!\t+\tseg2$%^&\t+\t0M")?;

        // Output file for adjacency matrix
        let output_file = NamedTempFile::new()?;

        // Run the conversion
        convert_gfa_to_edge_list(gfa_file.path(), output_file.path())?;

        // Read the output and verify the edges
        let edges = read_edges_from_file(output_file.path())?;
        let expected_edges = HashSet::from([
            (0, 1), (1, 0),
        ]);

        assert_eq!(
            edges, expected_edges,
            "Edges do not match expected values for segments with special characters."
        );

        Ok(())
    }

    /// Test that duplicate segments and links are handled correctly.
    #[test]
    fn test_gfa_to_adjacency_matrix_with_duplicate_segments() -> io::Result<()> {
        // Create a temporary GFA file with duplicate segment names
        let mut gfa_file = NamedTempFile::new()?;
        writeln!(gfa_file, "H\tVN:Z:1.0")?;
        writeln!(gfa_file, "S\tseg1\t*")?;
        writeln!(gfa_file, "S\tseg1\t*")?; // Duplicate segment
        writeln!(gfa_file, "S\tseg2\t*")?;
        writeln!(gfa_file, "L\tseg1\t+\tseg2\t+\t0M")?;
        writeln!(gfa_file, "L\tseg2\t+\tseg1\t+\t0M")?; // Duplicate link

        // Output file for adjacency matrix
        let output_file = NamedTempFile::new()?;

        // Run the conversion
        convert_gfa_to_edge_list(gfa_file.path(), output_file.path())?;

        // Read the output and verify the edges
        let edges = read_edges_from_file(output_file.path())?;
        let expected_edges = HashSet::from([
            (0, 1), (1, 0),
        ]);

        assert_eq!(
            edges, expected_edges,
            "Edges do not match expected values with duplicate segments."
        );

        Ok(())
    }

    /// Test that unique indices are correctly mapped for non-duplicate segments.
    #[test]
    fn test_unique_indices_mapping() -> io::Result<()> {
        // Create a temporary GFA file with unique segments
        let mut gfa_file = NamedTempFile::new()?;
        writeln!(gfa_file, "H\tVN:Z:1.0")?;
        writeln!(gfa_file, "S\tseg1\t*")?;
        writeln!(gfa_file, "S\tseg2\t*")?;
        writeln!(gfa_file, "L\tseg1\t+\tseg2\t+\t0M")?;

        // Output file for adjacency matrix
        let output_file = NamedTempFile::new()?;

        // Run the conversion
        convert_gfa_to_edge_list(gfa_file.path(), output_file.path())?;

        // Read the output and verify the edges
        let edges = read_edges_from_file(output_file.path())?;
        let expected_edges = HashSet::from([
            (0, 1), (1, 0),
        ]);

        assert_eq!(
            edges, expected_edges,
            "Edges do not match expected values for unique indices mapping."
        );

        Ok(())
    }

    /// Test the basic conversion from GFA to adjacency matrix.
    #[test]
    fn test_gfa_to_adjacency_matrix_basic_conversion() -> io::Result<()> {
        // Create a temporary GFA file with basic segments and links
        let mut gfa_file = NamedTempFile::new()?;
        writeln!(gfa_file, "H\tVN:Z:1.0")?;
        writeln!(gfa_file, "S\tA\t*")?;
        writeln!(gfa_file, "S\tB\t*")?;
        writeln!(gfa_file, "S\tC\t*")?;
        writeln!(gfa_file, "L\tA\t+\tB\t+\t0M")?;
        writeln!(gfa_file, "L\tB\t+\tC\t+\t0M")?;
        writeln!(gfa_file, "L\tA\t+\tC\t+\t0M")?;

        // Output file for adjacency matrix
        let output_file = NamedTempFile::new()?;

        // Run the conversion
        convert_gfa_to_edge_list(gfa_file.path(), output_file.path())?;

        // Read the output and verify the edges
        let edges = read_edges_from_file(output_file.path())?;
        let expected_edges = HashSet::from([
            (0, 1), (1, 0),
            (1, 2), (2, 1),
            (0, 2), (2, 0),
        ]);

        assert_eq!(
            edges, expected_edges,
            "Edges do not match expected values for basic conversion."
        );

        Ok(())
    }

    /// Test that edges are written correctly for multiple segments and links.
    #[test]
    fn test_edges_written_correctly() -> io::Result<()> {
        // Create a temporary GFA file with multiple segments and links
        let mut gfa_file = NamedTempFile::new()?;
        writeln!(gfa_file, "H\tVN:Z:1.0")?;
        writeln!(gfa_file, "S\t0\t*")?;
        writeln!(gfa_file, "S\t1\t*")?;
        writeln!(gfa_file, "S\t2\t*")?;
        writeln!(gfa_file, "S\t3\t*")?;
        writeln!(gfa_file, "L\t0\t+\t1\t+\t0M")?;
        writeln!(gfa_file, "L\t1\t+\t2\t+\t0M")?;
        writeln!(gfa_file, "L\t2\t+\t3\t+\t0M")?;
        writeln!(gfa_file, "L\t0\t+\t3\t+\t0M")?;

        // Output file for adjacency matrix
        let output_file = NamedTempFile::new()?;

        // Run the conversion
        convert_gfa_to_edge_list(gfa_file.path(), output_file.path())?;

        // Read the output and verify the edges
        let edges = read_edges_from_file(output_file.path())?;
        let expected_edges = HashSet::from([
            (0, 1), (1, 0),
            (1, 2), (2, 1),
            (2, 3), (3, 2),
            (0, 3), (3, 0),
        ]);

        assert_eq!(
            edges, expected_edges,
            "Edges do not match expected values for edges written correctly."
        );

        Ok(())
    }

    /// Test unique indices mapping with more complex segment names.
    #[test]
    fn test_unique_indices_mapping_with_special_characters_more() -> io::Result<()> {
        // Create a temporary GFA file with segments having complex names
        let mut gfa_file = NamedTempFile::new()?;
        writeln!(gfa_file, "H\tVN:Z:1.0")?;
        writeln!(gfa_file, "S\tHG00438#1#JAHBCB010000006.1\t*")?;
        writeln!(gfa_file, "S\tHG00438#1#JAHBCB010000013.1\t*")?;
        writeln!(gfa_file, "S\tHG00438#1#JAHBCB010000015.1\t*")?;
        writeln!(gfa_file, "L\tHG00438#1#JAHBCB010000006.1\t+\tHG00438#1#JAHBCB010000013.1\t+\t0M")?;
        writeln!(gfa_file, "L\tHG00438#1#JAHBCB010000013.1\t+\tHG00438#1#JAHBCB010000015.1\t+\t0M")?;

        // Output file for adjacency matrix
        let output_file = NamedTempFile::new()?;

        // Run the conversion
        convert_gfa_to_edge_list(gfa_file.path(), output_file.path())?;

        // Read the output and verify the edges
        let edges = read_edges_from_file(output_file.path())?;
        let expected_edges = HashSet::from([
            (0, 1), (1, 0),
            (1, 2), (2, 1),
        ]);

        assert_eq!(
            edges, expected_edges,
            "Edges do not match expected values for segments with special characters."
        );

        Ok(())
    }
}
