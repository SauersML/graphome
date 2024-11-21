use graphome::extract::*;
use graphome::eigen::{
    adjacency_matrix_to_ndarray,
    call_eigendecomp,
    compute_ngec,
    print_heatmap,
    print_heatmap_ndarray,
    print_eigenvalues_heatmap
};

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use tempfile::{NamedTempFile, tempdir};
use ndarray::prelude::*;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::collections::HashSet;

use graphome::convert::convert_gfa_to_edge_list;

#[cfg(test)]
mod tests {
    use super::*;
    use graphome::extract;
    use std::fs::File;
    use std::io::Read;
    use std::cmp::min;

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

    /// Helper function to create a mock .gam file with given edges
    fn create_mock_gam_file<P: AsRef<Path>>(path: P, edges: &[(u32, u32)]) -> io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        for &(from, to) in edges {
            writer.write_all(&from.to_le_bytes())?;
            writer.write_all(&to.to_le_bytes())?;
        }
        writer.flush()?;
        Ok(())
    }

    /// Test that the mock .gam file is created correctly
    #[test]
    fn test_create_mock_gam_file() -> io::Result<()> {
        let dir = tempdir()?;
        let gam_path = dir.path().join("mock.gam");

        let mock_edges = vec![(0, 1), (1, 2), (2, 0)];
        create_mock_gam_file(&gam_path, &mock_edges)?;

        // Read the file back and verify its contents
        let mut file = File::open(&gam_path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        // Each edge consists of 8 bytes (4 bytes for `from` and 4 bytes for `to`)
        assert_eq!(buffer.len(), mock_edges.len() * 8);

        for (i, &(from, to)) in mock_edges.iter().enumerate() {
            let offset = i * 8;
            let from_bytes = &buffer[offset..offset + 4];
            let to_bytes = &buffer[offset + 4..offset + 8];
            let from_loaded = u32::from_le_bytes([
                from_bytes[0],
                from_bytes[1],
                from_bytes[2],
                from_bytes[3],
            ]);
            let to_loaded = u32::from_le_bytes([
                to_bytes[0],
                to_bytes[1],
                to_bytes[2],
                to_bytes[3],
            ]);
            assert_eq!(from_loaded, from);
            assert_eq!(to_loaded, to);
        }

        dir.close()?;
        Ok(())
    }

    /// Test that load_adjacency_matrix correctly loads edges from a .gam file
    #[test]
    fn test_load_adjacency_matrix() -> io::Result<()> {
        let dir = tempdir()?;
        let gam_path = dir.path().join("test.gam");

        let mock_edges = vec![(1, 2), (2, 3), (3, 1)];
        create_mock_gam_file(&gam_path, &mock_edges)?;

        let loaded_edges = load_adjacency_matrix(&gam_path, 1, 3)?;

        assert_eq!(loaded_edges.len(), mock_edges.len());
        for &(from, to) in &mock_edges {
            assert!(loaded_edges.contains(&(from, to)));
        }

        dir.close()?;
        Ok(())
    }

    /// Test extracting a submatrix covering all nodes and verifying the Laplacian matches the original adjacency matrix
    #[test]
    fn test_full_range_extraction() -> io::Result<()> {
        // Create a temporary GFA file with sample data
        let mut gfa_file = NamedTempFile::new()?;
        writeln!(gfa_file, "H\tVN:Z:1.0")?;
        writeln!(gfa_file, "S\t1\t*")?;
        writeln!(gfa_file, "S\t2\t*")?;
        writeln!(gfa_file, "S\t3\t*")?;
        writeln!(gfa_file, "L\t1\t+\t2\t+\t50M")?;
        writeln!(gfa_file, "L\t2\t+\t3\t+\t60M")?;
        writeln!(gfa_file, "L\t1\t+\t3\t+\t70M")?;

        // Output file for adjacency matrix
        let output_gam = NamedTempFile::new()?;

        // Run the conversion
        convert_gfa_to_edge_list(gfa_file.path(), output_gam.path())?;

        // Define the full range (all nodes)
        let start_node = 1;
        let end_node = 3;

        // Run the extraction
        let (laplacian, eigvals, eigvecs) = extract_and_analyze(output_gam.path(), start_node, end_node)?;

        // Define expected edges within the range as a Vec
        let _expected_edges: Vec<(u32, u32)> = vec![
            (1, 2), // From node 1 to node 2
            (2, 1), // From node 2 to node 1
            (2, 3), // From node 2 to node 3
            (3, 2), // From node 3 to node 2
            (1, 3), // From node 1 to node 3
            (3, 1), // From node 3 to node 1
        ];

        // Verify the Laplacian matrix
        // For nodes 1, 2, 3 with all mutual connections:
        // Degrees: 2, 2, 2
        // Laplacian should be:
        // [2, -1, -1]
        // [-1, 2, -1]
        // [-1, -1, 2]
        let expected_laplacian = array![
            [2.0, -1.0, -1.0],
            [-1.0, 2.0, -1.0],
            [-1.0, -1.0, 2.0],
        ];
        assert_eq!(
            laplacian, expected_laplacian,
            "Laplacian matrix does not match expected values for full range extraction."
        );

        // Expected eigenvalues
        let expected_eigenvalues = vec![0.0, 3.0, 3.0];

        // The order doesn't affect the comparison
        let mut sorted_eigenvalues = eigvals.clone();
        sorted_eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut sorted_expected = expected_eigenvalues.clone();
        sorted_expected.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Define a tolerance for floating-point comparison
        let tolerance = 1e-6;

        // Compare each eigenvalue within the tolerance
        for (computed, expected) in sorted_eigenvalues.iter().zip(sorted_expected.iter()) {
            assert!(
                (*computed - *expected).abs() < tolerance,
                "Eigenvalue mismatch: computed {} vs expected {}",
                computed,
                expected
            );
        }

        // Print heatmaps for manual inspection
        println!("Laplacian Matrix Heatmap:");
        print_heatmap(&laplacian.view());

        println!("Eigenvectors Heatmap:");
        let eigenvecs_subset = eigvecs.slice(s![.., 0..min(500, eigvecs.ncols())]); // Display at max first 500
        print_heatmap_ndarray(&eigenvecs_subset.to_owned());

        println!("Eigenvalues Heatmap:");
        print_eigenvalues_heatmap(&eigvals);

        Ok(())
    }

    /// Test extracting a submatrix for a subset of nodes
    #[test]
    fn test_partial_range_extraction() -> io::Result<()> {
        // Create a temporary GFA file with sample data
        let mut gfa_file = NamedTempFile::new()?;
        writeln!(gfa_file, "H\tVN:Z:1.0")?;
        writeln!(gfa_file, "S\t1\t*")?;
        writeln!(gfa_file, "S\t2\t*")?;
        writeln!(gfa_file, "S\t3\t*")?;
        writeln!(gfa_file, "S\t4\t*")?;
        writeln!(gfa_file, "L\t1\t+\t2\t+\t50M")?;
        writeln!(gfa_file, "L\t2\t+\t3\t+\t60M")?;
        writeln!(gfa_file, "L\t3\t+\t4\t+\t70M")?;
        writeln!(gfa_file, "L\t1\t+\t4\t+\t80M")?;
        // Output file for adjacency matrix
        let output_gam = NamedTempFile::new()?;
        // Run the conversion
        convert_gfa_to_edge_list(gfa_file.path(), output_gam.path())?;
        // Define a partial range (nodes 2 to 3)
        let start_node = 2;
        let end_node = 3;

        // Run the extraction
        let (laplacian, eigvals, _eigvecs) = extract_and_analyze(output_gam.path(), start_node, end_node)?;

        // Define expected edges within the range as a Vec
        let _expected_edges: Vec<(u32, u32)> = vec![
            (2, 3), // From node 2 to node 3
            (3, 2), // From node 3 to node 2
        ];
        // Verify the Laplacian matrix
        // For nodes 2 and 3, the adjacency is:
        // 2 connected to 3
        // 3 connected to 2
        // Thus, the Laplacian should be:
        // [1, -1]
        // [-1, 1]
        let expected_laplacian = array![
            [1.0, -1.0],
            [-1.0, 1.0],
        ];
        assert_eq!(
            laplacian, expected_laplacian,
            "Laplacian matrix does not match expected values for partial range extraction."
        );
        // Expected eigenvalues for the Laplacian matrix [1, -1; -1, 1] are [0.0, 2.0]
        let expected_eigenvalues = vec![0.0, 2.0];
        // The order doesn't affect the comparison
        let mut sorted_eigenvalues = eigvals.clone();
        sorted_eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut sorted_expected = expected_eigenvalues.clone();
        sorted_expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Define a tolerance for floating-point comparison
        let tolerance = 1e-6;
        // Compare each eigenvalue within the tolerance
        for (computed, expected) in sorted_eigenvalues.iter().zip(sorted_expected.iter()) {
            assert!(
                (*computed - *expected).abs() < tolerance,
                "Eigenvalue mismatch: computed {} vs expected {}",
                computed,
                expected
            );
        }

        Ok(())
    }



    /// Test that adjacency_matrix_to_ndarray correctly converts edges to adjacency matrix
    #[test]
    fn test_adjacency_matrix_to_ndarray_correct_conversion() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let start_node = 0;
        let end_node = 2;

        let adj_matrix = adjacency_matrix_to_ndarray(&edges, start_node, end_node);

        let expected = array![
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ];

        assert_eq!(adj_matrix, expected);
    }

    /// Test that the adjacency matrix is symmetric
    #[test]
    fn test_adjacency_matrix_is_symmetric() {
        let edges = vec![(0, 1), (1, 2), (2, 0), (0, 2)];
        let adj_matrix = adjacency_matrix_to_ndarray(&edges, 0, 2);

        for i in 0..adj_matrix.nrows() {
            for j in 0..adj_matrix.ncols() {
                assert_eq!(adj_matrix[[i, j]], adj_matrix[[j, i]]);
            }
        }
    }

    /// Test that degree matrix is correctly computed from adjacency matrix
    #[test]
    fn test_degree_matrix_computation() {
        let adj_matrix = array![
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
        ];

        let degrees = adj_matrix.sum_axis(Axis(1));
        let degree_matrix = Array2::<f64>::from_diag(degrees);

        let expected_degrees = array![1.0, 3.0, 2.0, 2.0];
        let expected_degree_matrix = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
        ];

        assert_eq!(degrees, expected_degrees);
        assert_eq!(degree_matrix, expected_degree_matrix);
    }

    /// Test that Laplacian matrix is correctly computed from degree and adjacency matrices
    #[test]
    fn test_laplacian_computation() {
        let adj_matrix = array![
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
        ];

        let degrees = adj_matrix.sum_axis(Axis(1));
        let degree_matrix = Array2::<f64>::from_diag(degrees);
        let laplacian = degree_matrix - &adj_matrix;

        let expected_laplacian = array![
            [1.0, -1.0, 0.0, 0.0],
            [-1.0, 3.0, -1.0, -1.0],
            [0.0, -1.0, 2.0, -1.0],
            [0.0, -1.0, -1.0, 2.0],
        ];

        assert_eq!(laplacian, expected_laplacian);
    }

    /// Test that eigendecomposition produces non-negative eigenvalues for Laplacian
    #[test]
    fn test_eigendecomposition_non_negative_eigenvalues() {
        let laplacian = DMatrix::<f64>::from_row_slice(
            3,
            3,
            &[2.0, -1.0, -1.0, -1.0, 2.0, -1.0, -1.0, -1.0, 2.0],
        );

        let symmetric_eigen = SymmetricEigen::new(laplacian);

        let eigvals = symmetric_eigen.eigenvalues;

        for lambda in eigvals.iter().cloned() {
            assert!(
                lambda >= -1e-6,
                "Eigenvalue {} is negative beyond tolerance.",
                lambda
            );
        }
    }

    /// Test the correctness of the eigendecomposition process by verifying that L * v = λ * v for each eigenpair
    #[test]
    fn test_eigendecomposition_correctness() -> io::Result<()> {
        // Define a small Laplacian matrix manually
        let laplacian = DMatrix::from_row_slice(
            3,
            3,
            &[
                2.0, -1.0, -1.0,
                -1.0, 2.0, -1.0,
                -1.0, -1.0, 2.0,
            ],
        );

        // Perform eigendecomposition
        let symmetric_eigen = SymmetricEigen::new(laplacian.clone());

        let eigvals = symmetric_eigen.eigenvalues;
        let eigvecs = symmetric_eigen.eigenvectors;

        // Iterate through each eigenpair and verify L * v = λ * v
        for i in 0..eigvals.len() {
            let lambda = eigvals[i];
            let v = eigvecs.column(i);

            // Compute L * v
            let lv = &laplacian * &v;

            // Compute lambda * v
            let lambda_v: DVector<f64> = v * lambda; // Explicitly specifying the type for lambda_v

            // Allow a small tolerance for floating-point comparisons
            let tolerance = 1e-6;
            for j in 0..v.len() {
                assert!(
                    (lv[j] - lambda_v[j]).abs() < tolerance,
                    "Eigendecomposition incorrect for eigenpair {}: L*v[{}] = {}, lambda*v[{}] = {}",
                    i,
                    j,
                    lv[j],
                    j,
                    lambda_v[j]
                );
            }
        }

        Ok(())
    }
}
