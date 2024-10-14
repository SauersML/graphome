// tests/test_extract.rs

use graphome::extract::*;

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use ndarray::prelude::*;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use tempfile::{NamedTempFile, tempdir};

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Read;

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
            let from_loaded = u32::from_le_bytes([from_bytes[0], from_bytes[1], from_bytes[2], from_bytes[3]]);
            let to_loaded = u32::from_le_bytes([to_bytes[0], to_bytes[1], to_bytes[2], to_bytes[3]]);
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
        let degree_matrix = Array2::<f64>::from_diag(&degrees);

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
        let degree_matrix = Array2::<f64>::from_diag(&degrees);
        let laplacian = &degree_matrix - &adj_matrix;

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
            assert!(lambda >= -1e-6); // Allowing a small negative due to floating point precision
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
            let lv = laplacian * &v;
            
            // Compute λ * v
            let lambda_v: DVector<f64> = v * lambda;
            
            // Allow a small tolerance for floating-point comparisons
            let tolerance = 1e-6;
            for j in 0..v.len() {
                assert!(
                    (lv[j] - lambda_v[j]).abs() < tolerance,
                    "Eigendecomposition incorrect for eigenpair {}: L*v[{}] = {}, λ*v[{}] = {}",
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

    /// Test that saving the adjacency matrix to CSV works correctly
    #[test]
    fn test_save_matrix_to_csv() -> io::Result<()> {
        // Create a temporary matrix
        let matrix = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0])
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        // Output file
        let output_file = NamedTempFile::new()?;

        // Save matrix to CSV
        save_matrix_to_csv(&matrix, output_file.path())?;

        // Read the output file
        let mut file = File::open(output_file.path())?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        // Expected output with floats
        let expected = "0.0,1.0\n1.0,0.0\n";

        assert_eq!(
            contents, expected,
            "Saved matrix does not match expected float values."
        );

        Ok(())
    }

    /// Test that saving the nalgebra matrix to CSV works correctly
    #[test]
    fn test_save_nalgebra_matrix_to_csv() -> io::Result<()> {
        // Create a temporary matrix
        let matrix = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);

        // Output file
        let output_file = NamedTempFile::new()?;

        // Save matrix to CSV
        save_nalgebra_matrix_to_csv(&matrix, output_file.path())?;

        // Read the output file
        let mut file = File::open(output_file.path())?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        // Expected output with floats
        let expected = "0.0,1.0\n1.0,0.0\n";

        assert_eq!(
            contents, expected,
            "Saved nalgebra matrix does not match expected float values."
        );

        Ok(())
    }

    /// Test that saving the nalgebra vector to CSV works correctly
    #[test]
    fn test_save_nalgebra_vector_to_csv() -> io::Result<()> {
        // Create a temporary vector
        let vector = DVector::from_vec(vec![0.0, 3.0, 3.0]);

        // Output file
        let output_file = NamedTempFile::new()?;

        // Save vector to CSV
        save_nalgebra_vector_to_csv(&vector, output_file.path())?;

        // Read the output file
        let mut file = File::open(output_file.path())?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        // Expected output with floats
        let expected = "0.0,3.0,3.0\n";

        assert_eq!(
            contents, expected,
            "Saved vector does not match expected float values."
        );

        Ok(())
    }
}
