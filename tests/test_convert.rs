// tests/test_convert.rs

use graphome::extract::*;
use graphome::eigen_print::{
    adjacency_matrix_to_ndarray,
    save_nalgebra_matrix_to_csv,
    save_nalgebra_vector_to_csv,
};

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use tempfile::{NamedTempFile, tempdir};
use ndarray::prelude::*;
use std::io::{self, BufWriter, Write, Read};
use std::path::Path;
use std::fs::File;

use graphome::convert::convert_gfa_to_edge_list;
use graphome::extract;

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
    // Create a temporary directory for all test files
    let test_dir = tempdir()?;

    // Create a temporary GFA file with sample data in the test directory
    let gfa_path = test_dir.path().join("test.gfa");
    {
        let mut gfa_file = File::create(&gfa_path)?;
        writeln!(gfa_file, "H\tVN:Z:1.0")?;
        writeln!(gfa_file, "S\t1\t*")?;
        writeln!(gfa_file, "S\t2\t*")?;
        writeln!(gfa_file, "S\t3\t*")?;
        writeln!(gfa_file, "L\t1\t+\t2\t+\t50M")?;
        writeln!(gfa_file, "L\t2\t+\t3\t+\t60M")?;
        writeln!(gfa_file, "L\t1\t+\t3\t+\t70M")?;
    }

    // Output file for adjacency matrix (the .gam file) in the test directory
    let output_gam = test_dir.path().join("test.gam");

    // Run the conversion
    convert_gfa_to_edge_list(&gfa_path, &output_gam)?;

    // Define the full range (all nodes)
    let start_node = 0;
    let end_node = 2;

    // Run the extraction
    extract::extract_and_analyze_submatrix(
        &output_gam,
        start_node,
        end_node,
    )?;

    // Build paths to the output files within the test_dir
    let laplacian_csv = test_dir.path().join("laplacian.csv");
    let eigenvalues_csv = test_dir.path().join("eigenvalues.csv");
    let eigenvectors_csv = test_dir.path().join("eigenvectors.csv");

    // Load the Laplacian matrix from CSV
    let laplacian = load_csv_as_matrix(&laplacian_csv)?;

    // Verify the Laplacian matrix
    let expected_laplacian = array![
        [2.0, -1.0, -1.0],
        [-1.0, 2.0, -1.0],
        [-1.0, -1.0, 2.0],
    ];
    assert_eq!(
        laplacian, expected_laplacian,
        "Laplacian matrix does not match expected values for full range extraction."
    );

    // Load eigenvalues
    let eigenvalues = load_csv_as_vector(&eigenvalues_csv)?;

    // Expected eigenvalues
    let expected_eigenvalues = vec![0.0, 3.0, 3.0];

    // The order doesn't affect the comparison
    let mut sorted_eigenvalues = eigenvalues.clone();
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

    // Load eigenvectors if needed
    let _eigenvectors = load_csv_as_matrix(&eigenvectors_csv)?;

    // Clean up
    test_dir.close()?;
    Ok(())
}

/// Test extracting a submatrix for a subset of nodes and ensuring the Laplacian and eigendecomposition are computed correctly
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
    // Create output directory first
    let output_dir = tempdir()?;
    // Output file for adjacency matrix in the output directory
    let output_gam = output_dir.path().join("test.gam");
    // Run the conversion
    convert_gfa_to_edge_list(gfa_file.path(), &output_gam)?;
    // Define a partial range (nodes 2 to 3)
    let start_node = 1;
    let end_node = 2;
    // Run the extraction
    extract::extract_and_analyze_submatrix(
        output_gam.as_path(),
        start_node,
        end_node,
    )?;
    // Define expected edges within the range as a Vec
    let _expected_edges: Vec<(u32, u32)> = vec![
        (1, 2), // From node 2 to node 3
        (2, 1), // From node 3 to node 2
    ];
    // Load the Laplacian matrix from CSV
    let laplacian_csv = output_dir.path().join("laplacian.csv");
    
    let laplacian = load_csv_as_matrix(&laplacian_csv)?;
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
    // Load the eigenvalues from CSV
    let eigenvalues_csv = output_dir.path().join("eigenvalues.csv");
    let eigenvalues = load_csv_as_vector(&eigenvalues_csv)?;
    // Expected eigenvalues for the Laplacian matrix [1, -1; -1, 1] are [0.0, 2.0]
    let expected_eigenvalues = vec![0.0, 2.0];
    // The order doesn't affect the comparison
    let mut sorted_eigenvalues = eigenvalues.clone();
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
    // Load eigenvectors
    let eigenvectors_csv = output_dir.path().join("eigenvectors.csv");
    let eigenvectors = load_csv_as_matrix(&eigenvectors_csv)?;
    // Since eigenvectors can vary in sign and orientation, we'll focus on verifying the eigenvalues' correctness
    // Alternatively, you can implement additional checks for eigenvectors if necessary
    Ok(())
}

/// Helper function to load a CSV file as an ndarray::Array2<f64>
fn load_csv_as_matrix<P: AsRef<Path>>(path: P) -> io::Result<Array2<f64>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)?;
    let mut matrix = Vec::new();
    for result in rdr.records() {
        let record = result?;
        let row = record
            .iter()
            .map(|s| s.parse::<f64>().unwrap())
            .collect::<Vec<f64>>();
        matrix.push(row);
    }
    let nrows = matrix.len();
    let ncols = if nrows > 0 { matrix[0].len() } else { 0 };
    Array2::from_shape_vec((nrows, ncols), matrix.into_iter().flatten().collect())
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
}

/// Helper function to load a CSV file as a nalgebra::DVector<f64>
fn load_csv_as_vector<P: AsRef<Path>>(path: P) -> io::Result<Vec<f64>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)?;
    let mut vector = Vec::new();
    for result in rdr.records() {
        let record = result?;
        for field in record.iter() {
            vector.push(field.parse::<f64>().unwrap());
        }
    }
    Ok(vector)
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


/// Counts the number of edges in a `.gam` file by checking its size in bytes.
/// Each edge is represented by two `u32` values = 8 bytes.
fn count_edges_in_gam<P: AsRef<Path>>(gam_path: P) -> std::io::Result<usize> {
    let metadata = std::fs::metadata(gam_path)?;
    let file_size = metadata.len();
    // Each edge is 8 bytes
    Ok((file_size / 8) as usize)
}

/// Test that a GFA with no links results in zero edges
#[test]
fn test_no_links_expected_zero_edges() -> std::io::Result<()> {
    let test_dir = tempdir()?;
    let gfa_path = test_dir.path().join("no_links.gfa");

    {
        let mut gfa_file = File::create(&gfa_path)?;
        writeln!(gfa_file, "H\tVN:Z:1.0")?;
        writeln!(gfa_file, "S\t1\t*")?;
        writeln!(gfa_file, "S\t2\t*")?;
        // No L lines
    }

    let output_gam = test_dir.path().join("no_links.gam");
    convert_gfa_to_edge_list(&gfa_path, &output_gam)?;

    let num_edges = count_edges_in_gam(&output_gam)?;
    assert_eq!(num_edges, 0, "Expected 0 edges for a GFA with no links.");

    Ok(())
}

/// Test that a GFA with exactly one link line produces exactly one edge
#[test]
fn test_one_link_expected_one_edge() -> std::io::Result<()> {
    let test_dir = tempdir()?;
    let gfa_path = test_dir.path().join("one_link.gfa");

    {
        let mut gfa_file = File::create(&gfa_path)?;
        writeln!(gfa_file, "H\tVN:Z:1.0")?;
        writeln!(gfa_file, "S\t1\t*")?;
        writeln!(gfa_file, "S\t2\t*")?;
        writeln!(gfa_file, "L\t1\t+\t2\t+\t50M")?;
    }

    let output_gam = test_dir.path().join("one_link.gam");
    convert_gfa_to_edge_list(&gfa_path, &output_gam)?;

    let num_edges = count_edges_in_gam(&output_gam)?;
    // Expect exactly 1 edge since we have 1 link line and no reverse edges are written
    assert_eq!(num_edges, 1, "Expected 1 edge for a GFA with one link line.");

    Ok(())
}

/// Test that a GFA with multiple links produces the exact number of edges
#[test]
fn test_three_links_expected_three_edges() -> std::io::Result<()> {
    let test_dir = tempdir()?;
    let gfa_path = test_dir.path().join("three_links.gfa");

    {
        let mut gfa_file = File::create(&gfa_path)?;
        writeln!(gfa_file, "H\tVN:Z:1.0")?;
        writeln!(gfa_file, "S\t1\t*")?;
        writeln!(gfa_file, "S\t2\t*")?;
        writeln!(gfa_file, "S\t3\t*")?;
        writeln!(gfa_file, "L\t1\t+\t2\t+\t50M")?;
        writeln!(gfa_file, "L\t2\t+\t3\t+\t60M")?;
        writeln!(gfa_file, "L\t1\t+\t3\t+\t70M")?;
    }

    let output_gam = test_dir.path().join("three_links.gam");
    convert_gfa_to_edge_list(&gfa_path, &output_gam)?;

    let num_edges = count_edges_in_gam(&output_gam)?;
    // We have 3 link lines, each producing 1 edge as currently coded
    let expected_edges = 3;
    assert_eq!(
        num_edges, expected_edges,
        "Expected {} edges for a GFA with three link lines.",
        expected_edges
    );

    Ok(())
}

/// Test with segments but no links, reconfirming zero edges
#[test]
fn test_segments_no_links_reconfirm_zero() -> std::io::Result<()> {
    let test_dir = tempdir()?;
    let gfa_path = test_dir.path().join("segments_no_links.gfa");

    {
        let mut gfa_file = File::create(&gfa_path)?;
        writeln!(gfa_file, "H\tVN:Z:1.0")?;
        writeln!(gfa_file, "S\tA\t*")?;
        writeln!(gfa_file, "S\tB\t*")?;
        writeln!(gfa_file, "S\tC\t*")?;
        // Still no links
    }

    let output_gam = test_dir.path().join("segments_no_links.gam");
    convert_gfa_to_edge_list(&gfa_path, &output_gam)?;

    let num_edges = count_edges_in_gam(&output_gam)?;
    assert_eq!(
        num_edges, 0,
        "Expected 0 edges for a GFA with segments but no links."
    );

    Ok(())
}
