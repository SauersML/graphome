extern crate cmake;
extern crate curl;

use cmake::Config;
use curl::easy::Easy;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;
use std::process::Command;

fn main() {
    println!("Build started");
    let lapack_url = "https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.0.tar.gz";
    let lapack_dir = "lapack"; // Directory to store LAPACK files
    let tarball = format!("{}.tar.gz", lapack_dir);

    // Download and extract LAPACK if it doesn't exist
    if !Path::new(lapack_dir).exists() {
        download_and_extract(lapack_url, &tarball, lapack_dir).expect("Failed to download and extract LAPACK");
    }

    // Build LAPACK statically using CMake
    let dst = Config::new(lapack_dir)
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("LAPACKE", "ON")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON") // -fPIC for static linking
        .build();

    // Tell Cargo to link with the static LAPACK library and ensure linking order
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=lapacke");
    println!("cargo:rustc-link-lib=static=lapack");
    println!("cargo:rustc-link-lib=static=blas");
}

// Function to download and extract LAPACK source
fn download_and_extract(url: &str, tarball: &str, output_dir: &str) -> io::Result<()> {
    download_file(url, tarball)?;
    extract_tarball(tarball)?;
    Ok(())
}

// Function to download a file using curl
fn download_file(url: &str, file_path: &str) -> io::Result<()> {
    let mut easy = Easy::new();
    easy.url(url).unwrap();

    let mut response = Vec::new();
    {
        let mut transfer = easy.transfer();
        transfer.write_function(|data| {
            response.extend_from_slice(data);
            Ok(data.len())
        }).unwrap();
        transfer.perform().unwrap();
    }

    let mut file = File::create(file_path)?;
    file.write_all(&response)?;
    Ok(())
}

// Function to extract a tar.gz file using `tar`
fn extract_tarball(tarball: &str) -> io::Result<()> {
    let output = Command::new("tar")
        .args(&["-xzf", tarball])
        .output()?;

    if !output.status.success() {
        Err(io::Error::new(io::ErrorKind::Other, "Failed to extract tarball"))
    } else {
        Ok(())
    }
}
