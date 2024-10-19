extern crate cmake;
extern crate curl;

use cmake::Config;
use curl::easy::Easy;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;
use std::process::Command;

/// Entry point for the build script.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("Build started");

    // Define versions
    let lapack_version = "v3.12.0";
    let openblas_version = "v0.3.28";

    // URLs for downloading sources
    let lapack_url = format!(
        "https://github.com/Reference-LAPACK/lapack/archive/refs/tags/{}.tar.gz",
        lapack_version
    );
    let openblas_url = format!(
        "https://github.com/xianyi/OpenBLAS/archive/refs/tags/{}.tar.gz",
        openblas_version
    );

    // Extracted directory names based on version
    let lapack_dir = format!("lapack-{}", lapack_version.trim_start_matches('v'));
    let openblas_dir = format!("OpenBLAS-{}", openblas_version.trim_start_matches('v'));

    // Tarballs
    let lapack_tarball = format!("{}.tar.gz", lapack_dir);
    let openblas_tarball = format!("{}.tar.gz", openblas_dir);

    // Download and extract OpenBLAS
    if !Path::new(&openblas_dir).exists() {
        download_and_extract(&openblas_url, &openblas_tarball, &openblas_dir)
            .expect("Failed to download and extract OpenBLAS");
    } else {
        println!("cargo:warning=OpenBLAS already exists: {}", openblas_dir);
    }

    // Build OpenBLAS
    let openblas_dst = Config::new(&openblas_dir)
        .define("BUILD_SHARED_LIBS", "OFF") // Build static libraries
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON") // Enable -fPIC
        .define("NO_SHARED", "ON")
        .define("NO_STATIC", "OFF")
        .build();

    // Download and extract LAPACK
    if !Path::new(&lapack_dir).exists() {
        download_and_extract(&lapack_url, &lapack_tarball, &lapack_dir)
            .expect("Failed to download and extract LAPACK");
    } else {
        println!("cargo:warning=LAPACK already exists: {}", lapack_dir);
    }

    // Build LAPACK with OpenBLAS
    let lapack_dst = Config::new(&lapack_dir)
        .define("CMAKE_Fortran_COMPILER", "gfortran") // Ensure using gfortran
        .define("BUILD_SHARED_LIBS", "OFF") // Build static libraries
        .define("LAPACKE", "ON") // Build LAPACKE (C interface)
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON") // Enable -fPIC
        .define(
            "BLAS_LIBRARIES",
            format!("{}/lib/libopenblas.a", openblas_dst.display()).as_str(),
        )
        .define("BLAS", "OpenBLAS")
        .build();

    // Link LAPACK libraries after OpenBLAS
    println!("cargo:rustc-link-search=native={}/lib", lapack_dst.display());
    println!("cargo:rustc-link-lib=static=lapacke");
    println!("cargo:rustc-link-lib=static=lapack");
    println!("cargo:rustc-link-lib=static=openblas");

    // Link Fortran runtime libraries
    println!("cargo:rustc-link-lib=dylib=gfortran");

    // Optional: Clean up tarballs to save space
    fs::remove_file(&openblas_tarball).ok();
    fs::remove_file(&lapack_tarball).ok();

    // Debugging: List contents of LAPACK lib directory
    println!("cargo:warning=Lapack build directory: {}", lapack_dst.display());

    let lapack_lib_dir = lapack_dst.join("lib");
    if lapack_lib_dir.exists() {
        for entry in fs::read_dir(&lapack_lib_dir).expect("Failed to read LAPACK lib directory") {
            let entry = entry.expect("Failed to get directory entry");
            println!("cargo:warning=Found in LAPACK lib: {:?}", entry.path());
        }
    } else {
        println!(
            "cargo:warning=LAPACK lib directory does not exist: {:?}",
            lapack_lib_dir
        );
    }
}

/// Downloads and extracts a tarball.
fn download_and_extract(url: &str, tarball: &str, output_dir: &str) -> io::Result<()> {
    download_file(url, tarball)?;
    extract_tarball(tarball)?;
    Ok(())
}

/// Downloads a file from the specified URL using `curl`.
fn download_file(url: &str, file_path: &str) -> io::Result<()> {
    println!("Downloading {} to {}", url, file_path);
    let mut easy = Easy::new();
    easy.url(url).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    let mut response = Vec::new();
    {
        let mut transfer = easy.transfer();
        transfer
            .write_function(|data| {
                response.extend_from_slice(data);
                Ok(data.len())
            })
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        transfer.perform().map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    }

    let mut file = File::create(file_path)?;
    file.write_all(&response)?;
    println!("Downloaded {}", file_path);
    Ok(())
}

/// Extracts a tar.gz file using the `tar` command.
fn extract_tarball(tarball: &str) -> io::Result<()> {
    println!("Extracting {}", tarball);
    let output = Command::new("tar")
        .args(&["-xzf", tarball])
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to extract tarball {}: {}", tarball, stderr),
        ));
    }

    println!("Extracted {}", tarball);
    Ok(())
}
