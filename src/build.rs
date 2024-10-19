extern crate cmake;
extern crate curl;

use cmake::Config;
use curl::easy::Easy;
use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    let lapack_url = "https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.0.tar.gz"; // LAPACK source URL
    let lapack_dir = "lapack"; // Folder to extract LAPACK

    // Download LAPACK source if it doesn't exist
    if !Path::new(lapack_dir).exists() {
        download_and_extract(lapack_url, lapack_dir).unwrap();
    }

    // Build LAPACK statically using CMake
    let dst = Config::new(lapack_dir)
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("LAPACKE", "ON")
        .build();

    // Tell Rust to link with the static LAPACK library
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=lapack");
    println!("cargo:rustc-link-lib=static=blas");
}

// Function to download and extract LAPACK source
fn download_and_extract(url: &str, output_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut easy = Easy::new();
    easy.url(url)?;

    // Download the tar.gz file
    let mut response = Vec::new();
    {
        let mut transfer = easy.transfer();
        transfer.write_function(|data| {
            response.extend_from_slice(data);
            Ok(data.len())
        })?;
        transfer.perform()?;
    }

    // Write the downloaded content to a file
    let tarball = format!("{}.tar.gz", output_dir);
    fs::write(&tarball, &response)?;

    // Extract the tar.gz file
    Command::new("tar")
        .args(&["-xzf", &tarball])
        .output()
        .expect("Failed to extract tar.gz");

    Ok(())
}
