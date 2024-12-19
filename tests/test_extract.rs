// tests/test_extract.rs

use graphome::extract::*;
use graphome::eigen_print::{
    adjacency_matrix_to_ndarray,
    save_nalgebra_matrix_to_csv,
    save_nalgebra_vector_to_csv,
};

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use tempfile::{NamedTempFile, tempdir};
use ndarray::prelude::*;
use std::io::{self, Write, Read, BufWriter};
use std::path::Path;
use std::collections::HashSet;
use std::fs::File;

use graphome::convert::convert_gfa_to_edge_list;
use graphome::extract;





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
    // Create a temporary directory
    let dir = tempdir()?;

    // Create a temporary GFA file with sample data
    let gfa_path = dir.path().join("test.gfa");
    let mut gfa_file = File::create(&gfa_path)?;
    writeln!(gfa_file, "H\tVN:Z:1.0")?;
    writeln!(gfa_file, "S\t1\t*")?;
    writeln!(gfa_file, "S\t2\t*")?;
    writeln!(gfa_file, "S\t3\t*")?;
    writeln!(gfa_file, "L\t1\t+\t2\t+\t50M")?;
    writeln!(gfa_file, "L\t2\t+\t3\t+\t60M")?;
    writeln!(gfa_file, "L\t1\t+\t3\t+\t70M")?;

    // Output file for adjacency matrix
    let output_gam_path = dir.path().join("output.gam");

    // Run the conversion
    convert_gfa_to_edge_list(&gfa_path, &output_gam_path)?;

    // Define the full range (all nodes)
    let start_node = 0;
    let end_node = 2;

    // Run the extraction
    extract::extract_and_analyze_submatrix(
        &output_gam_path,
        start_node,
        end_node,
    )?;

    // Paths to the expected output files
    let laplacian_csv = dir.path().join("laplacian.csv");
    let eigenvalues_csv = dir.path().join("eigenvalues.csv");
    let eigenvectors_csv = dir.path().join("eigenvectors.csv");

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

    // Load eigenvectors (optional)
    let eigenvectors = load_csv_as_matrix(&eigenvectors_csv)?;
    // Additional tests on eigenvectors can be added here

    // Close the temporary directory (optional)
    dir.close()?;
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

/// Test extracting a submatrix for a subset of nodes
#[test]
fn test_partial_range_extraction() -> io::Result<()> {
    // Create a temporary directory
    let dir = tempdir()?;

    // Create a temporary GFA file with sample data
    let gfa_path = dir.path().join("test.gfa");
    let mut gfa_file = File::create(&gfa_path)?;
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
    let output_gam_path = dir.path().join("output.gam");

    // Run the conversion
    convert_gfa_to_edge_list(&gfa_path, &output_gam_path)?;

    // Define a partial range (nodes 2 to 3)
    let start_node = 1;
    let end_node = 2;

    // Run the extraction
    extract::extract_and_analyze_submatrix(
        &output_gam_path,
        start_node,
        end_node,
    )?;

    // Paths to the expected output files
    let laplacian_csv = dir.path().join("laplacian.csv");
    let eigenvalues_csv = dir.path().join("eigenvalues.csv");
    let eigenvectors_csv = dir.path().join("eigenvectors.csv");

    // Load the Laplacian matrix from CSV
    let laplacian = load_csv_as_matrix(&laplacian_csv)?;

    // Verify the Laplacian matrix
    let expected_laplacian = array![
        [1.0, -1.0],
        [-1.0, 1.0],
    ];

    assert_eq!(
        laplacian, expected_laplacian,
        "Laplacian matrix does not match expected values for partial range extraction."
    );

    // Load eigenvalues
    let eigenvalues = load_csv_as_vector(&eigenvalues_csv)?;

    // Expected eigenvalues
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

    // Load eigenvectors (optional)
    let eigenvectors = load_csv_as_matrix(&eigenvectors_csv)?;
    // Additional tests on eigenvectors can be added here

    // Close the temporary directory (optional)
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

















/// Helper function to create a mock .gam file with given edges (u32, u32)
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

/// Helper function to read edges from a .gam file into a HashSet
fn read_edges_from_gam<P: AsRef<Path>>(path: P) -> io::Result<HashSet<(u32, u32)>> {
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

/// Helper function to compare two floating point matrices with exact equality.
/// This is suitable here because the Laplacian should have exact integer-like values.
fn assert_matrix_eq(a: &Array2<f64>, b: &Array2<f64>) {
    assert_eq!(a.shape(), b.shape(), "Matrices have different shapes.");
    for ((i, j), &val_a) in a.indexed_iter() {
        let val_b = b[[i, j]];
        assert!(
            (val_a - val_b).abs() < 1e-12,
            "Matrix values differ at ({}, {}): {} vs {}",
            i,
            j,
            val_a,
            val_b
        );
    }
}

/// Empty Graph (No edges) - makes sure no edges appear unexpectedly
#[test]
fn test_no_edges_expect_none() -> io::Result<()> {
    let dir = tempdir()?;
    let gam_path = dir.path().join("no_edges.gam");

    // No edges written
    create_mock_gam_file(&gam_path, &[])?;

    // Load adjacency matrix in some range (e.g., 0..2)
    let edges = load_adjacency_matrix(&gam_path, 0, 2)?;
    assert!(edges.is_empty(), "Expected no edges, got some edges");

    // Construct Laplacian and verify it's all zeros
    let laplacian = fast_laplacian_from_gam(&gam_path, 0, 2)?;
    let expected = Array2::<f64>::zeros((3, 3));
    assert_matrix_eq(&laplacian, &expected);

    Ok(())
}

/// Single Edge In-Range
#[test]
fn test_single_edge_in_range() -> io::Result<()> {
    let dir = tempdir()?;
    let gam_path = dir.path().join("single_edge.gam");

    // Write a single edge from node 1 to node 2
    create_mock_gam_file(&gam_path, &[(1, 2)])?;

    // Load adjacency matrix with range covering both 1 and 2
    let edges = load_adjacency_matrix(&gam_path, 1, 2)?;
    assert_eq!(edges.len(), 1, "Expected exactly one edge");
    assert!(edges.contains(&(1, 2)), "Expected edge (1,2) not found");

    // Laplacian for nodes [1,2]: one edge means:
    // Node 1 degree = 1, Node 2 degree = 1
    // Laplacian = [[1, -1],
    //              [-1, 1]]
    let laplacian = fast_laplacian_from_gam(&gam_path, 1, 2)?;
    let expected = array![
        [1.0, -1.0],
        [-1.0, 1.0],
    ];
    assert_matrix_eq(&laplacian, &expected);

    Ok(())
}

/// Multiple Edges with Partial Range Exclusion
#[test]
fn test_partial_range_exclusion() -> io::Result<()> {
    // Nodes: 1,2,3,4
    // Edges: (1,2), (2,3), (3,4), (1,4)
    // Subrange: [1,2]
    // Only (1,2) should remain, since (2,3), (3,4), (1,4) are out-of-range.
    
    let dir = tempdir()?;
    let gam_path = dir.path().join("partial_exclusion.gam");
    let all_edges = vec![(1,2), (2,3), (3,4), (1,4)];
    create_mock_gam_file(&gam_path, &all_edges)?;

    // Load adjacency with subrange 1..2
    let edges = load_adjacency_matrix(&gam_path, 1, 2)?;
    assert_eq!(edges.len(), 1, "Only (1,2) should remain");
    assert!(edges.contains(&(1,2)), "Edge (1,2) expected");

    // Laplacian should reflect only edge (1,2):
    // Laplacian = [[1, -1],
    //              [-1, 1]]
    let laplacian = fast_laplacian_from_gam(&gam_path, 1, 2)?;
    let expected = array![
        [1.0, -1.0],
        [-1.0, 1.0],
    ];
    assert_matrix_eq(&laplacian, &expected);

    Ok(())
}

/// Edges on Subrange Boundaries
#[test]
fn test_edges_on_subrange_boundaries() -> io::Result<()> {
    // Graph: nodes 0,1,2,3
    // Edges: (0,1), (1,2), (2,3)
    // Subrange: [1,2]
    // Within [1,2], only (1,2) is fully in range.
    // (0,1) and (2,3) cross boundaries and must be excluded.

    let dir = tempdir()?;
    let gam_path = dir.path().join("boundary_edges.gam");
    let all_edges = vec![(0,1), (1,2), (2,3)];
    create_mock_gam_file(&gam_path, &all_edges)?;

    // Load adjacency with [1,2]
    let edges = load_adjacency_matrix(&gam_path, 1, 2)?;
    assert_eq!(edges.len(), 1, "Only (1,2) should be included");
    assert!(edges.contains(&(1,2)), "Expected edge (1,2) not found");

    // Laplacian for nodes [1,2]:
    // With one edge: same pattern as before
    let laplacian = fast_laplacian_from_gam(&gam_path, 1, 2)?;
    let expected = array![
        [1.0, -1.0],
        [-1.0, 1.0],
    ];
    assert_matrix_eq(&laplacian, &expected);

    Ok(())
}

/// Test E: No Duplicate or Extra Edges
#[test]
fn test_no_duplicate_or_extra_edges() -> io::Result<()> {    
    let dir = tempdir()?;
    let gam_path = dir.path().join("duplicates.gam");
    // Insert the same edge (1,2) three times.
    let duplicated_edges = vec![(1,2), (1,2), (1,2)];
    create_mock_gam_file(&gam_path, &duplicated_edges)?;

    // Even though duplicates exist, no new out-of-range edges should appear.
    // Subrange [1,2] includes (1,2), so we get all duplicates.
    let edges = load_adjacency_matrix(&gam_path, 1, 2)?;
    // We should see all three edges (1,2) as loaded. The code doesn't remove duplicates.
    // This makes sure no "extra different edges" appear.
    assert_eq!(edges.len(), 3, "Expected 3 occurrences of (1,2)");
    assert!(edges.iter().all(|&(f,t)| f==1 && t==2), "Only (1,2) edges should be present");
    
    // The Laplacian is computed from these edges. Each edge (1,2) counts.
    // If we treat each edge equally, the degree of node 1 = 3, node 2 = 3, and off-diagonal = -3.
    // Laplacian = [[3, -3],
    //              [-3, 3]]
    let laplacian = fast_laplacian_from_gam(&gam_path, 1, 2)?;
    let expected = array![
        [3.0, -3.0],
        [-3.0, 3.0],
    ];
    assert_matrix_eq(&laplacian, &expected);

    Ok(())
}


/// We write a known set of edges to a .gam file, and then load them back.
/// The loaded edges must match exactly the input set - no additional edges,
/// no duplicates (beyond what was in the file), and no edges that we did not specify.
#[test]
fn test_no_artificial_extras_or_duplicates() -> std::io::Result<()> {
    let dir = tempfile::tempdir()?;
    let gam_path = dir.path().join("no_extras.gam");

    // Define a set of edges with no duplicates
    let original_edges = vec![(0,1), (1,2), (2,2), (3,4)];
    // Write these edges to the .gam file
    {
        let file = std::fs::File::create(&gam_path)?;
        let mut writer = std::io::BufWriter::new(file);
        for &(from, to) in &original_edges {
            writer.write_all(&from.to_le_bytes())?;
            writer.write_all(&to.to_le_bytes())?;
        }
        writer.flush()?;
    }

    // Now load them back with a range that covers all these nodes:
    // The largest node index here is 4, so let's cover 0..4
    let loaded_edges = graphome::extract::load_adjacency_matrix(&gam_path, 0, 4)?;

    // Convert loaded edges into a sorted vector to compare easily
    let mut loaded_sorted = loaded_edges.clone();
    loaded_sorted.sort_unstable();

    let mut original_sorted = original_edges.clone();
    original_sorted.sort_unstable();

    // Check that the loaded edges exactly match the original edges
    assert_eq!(
        loaded_sorted, original_sorted,
        "Loaded edges do not match the original edges. \
         No extra edges or duplicates should appear."
    );

    // If the code attempted to create edges not listed in the file,
    // or somehow generated duplicates not in the original set, 
    // this assertion would fail.

    Ok(())
}
