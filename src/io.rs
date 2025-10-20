use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use aws_config::meta::region::RegionProviderChain;
use aws_sdk_s3::config::Region;
use aws_sdk_s3::Client;
use futures_util::TryStreamExt;
use reqwest::blocking::Client as HttpClient;
use tempfile::NamedTempFile;
use tokio::runtime::Builder as RuntimeBuilder;
use url::Url;

/// Represents a readable handle to either a local file, an HTTP(S) resource, or an S3 object.
pub enum InputReader {
    Local(BufReader<File>),
    Http(BufReader<reqwest::blocking::Response>),
    S3 {
        temp: NamedTempFile,
        reader: BufReader<File>,
    },
}

impl Read for InputReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            InputReader::Local(reader) => reader.read(buf),
            InputReader::Http(reader) => reader.read(buf),
            InputReader::S3 { reader, .. } => reader.read(buf),
        }
    }
}

impl BufRead for InputReader {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        match self {
            InputReader::Local(reader) => reader.fill_buf(),
            InputReader::Http(reader) => reader.fill_buf(),
            InputReader::S3 { reader, .. } => reader.fill_buf(),
        }
    }

    fn consume(&mut self, amt: usize) {
        match self {
            InputReader::Local(reader) => reader.consume(amt),
            InputReader::Http(reader) => reader.consume(amt),
            InputReader::S3 { reader, .. } => reader.consume(amt),
        }
    }
}

impl Seek for InputReader {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        match self {
            InputReader::Local(reader) => reader.seek(pos),
            InputReader::Http(_) => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "HTTP streams do not support seeking",
            )),
            InputReader::S3 { reader, .. } => reader.seek(pos),
        }
    }
}

/// A handle to a local file path. Remote resources are materialized to a temporary file
/// and kept alive while the handle exists.
pub struct MaterializedPath {
    path: PathBuf,
    temp: Option<NamedTempFile>,
}

impl MaterializedPath {
    /// Returns the local path that can be used by APIs requiring filesystem access.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

struct S3Location {
    bucket: String,
    key: String,
    region: Option<String>,
}

/// Open a resource that may be a local file path, HTTP(S) URL, or S3 URL.
///
/// HTTP(S) resources are streamed with a blocking client, while `s3://` URLs are fetched using the
/// official AWS Rust SDK.
pub fn open(path_or_url: &str) -> io::Result<InputReader> {
    if let Ok(url) = Url::parse(path_or_url) {
        match url.scheme() {
            "s3" => {
                let loc = parse_s3_scheme(&url)?;
                return open_s3(&loc);
            }
            "http" | "https" => {
                return open_http(&url);
            }
            _ => {}
        }
    }

    let file = File::open(path_or_url)?;
    Ok(InputReader::Local(BufReader::new(file)))
}

/// Materialize a resource to the local filesystem and return a handle to the temporary file.
///
/// Local paths are returned directly without additional work.
pub fn materialize(path_or_url: &str) -> io::Result<MaterializedPath> {
    if let Ok(url) = Url::parse(path_or_url) {
        match url.scheme() {
            "s3" => {
                let loc = parse_s3_scheme(&url)?;
                return download_s3(&loc);
            }
            "http" | "https" => {
                return download_http(&url);
            }
            _ => {}
        }
    }

    Ok(MaterializedPath {
        path: Path::new(path_or_url).to_path_buf(),
        temp: None,
    })
}

fn open_http(url: &Url) -> io::Result<InputReader> {
    let client = HttpClient::builder().build().map_err(to_io_error)?;
    let response = client
        .get(url.as_str())
        .send()
        .map_err(to_io_error)?
        .error_for_status()
        .map_err(to_io_error)?;
    Ok(InputReader::Http(BufReader::new(response)))
}

fn download_http(url: &Url) -> io::Result<MaterializedPath> {
    let client = HttpClient::builder().build().map_err(to_io_error)?;
    let mut response = client
        .get(url.as_str())
        .send()
        .map_err(to_io_error)?
        .error_for_status()
        .map_err(to_io_error)?;
    let mut temp = NamedTempFile::new()?;
    io::copy(&mut response, temp.as_file_mut())?;
    temp.as_file_mut().seek(SeekFrom::Start(0))?;
    let path = temp.path().to_path_buf();
    Ok(MaterializedPath {
        path,
        temp: Some(temp),
    })
}

fn open_s3(loc: &S3Location) -> io::Result<InputReader> {
    let mut materialized = download_s3(loc)?;
    let mut temp = materialized
        .temp
        .take()
        .expect("S3 downloads must produce a temporary file");
    let file = temp.reopen()?;
    // Ensure the reader starts at the beginning.
    temp.as_file_mut().seek(SeekFrom::Start(0))?;
    Ok(InputReader::S3 {
        temp,
        reader: BufReader::new(file),
    })
}

fn download_s3(loc: &S3Location) -> io::Result<MaterializedPath> {
    let rt = RuntimeBuilder::new_current_thread()
        .enable_all()
        .build()
        .map_err(to_io_error)?;
    let bucket = loc.bucket.clone();
    let key = loc.key.clone();
    let region = loc.region.clone();
    let mut temp = NamedTempFile::new()?;
    rt.block_on(async {
        let region_provider = match region {
            Some(region_name) => RegionProviderChain::first_try(Region::new(region_name))
                .or_default_provider()
                .or_else("us-east-1"),
            None => RegionProviderChain::default_provider().or_else("us-east-1"),
        };
        let config = aws_config::from_env().region(region_provider).load().await;
        let client = Client::new(&config);
        let mut obj = client
            .get_object()
            .bucket(&bucket)
            .key(&key)
            .send()
            .await
            .map_err(to_io_error)?;
        while let Some(bytes) = obj.body.try_next().await.map_err(to_io_error)? {
            temp.as_file_mut().write_all(&bytes)?;
        }
        Ok::<(), io::Error>(())
    })?;
    temp.as_file_mut().seek(SeekFrom::Start(0))?;
    let path = temp.path().to_path_buf();
    Ok(MaterializedPath {
        path,
        temp: Some(temp),
    })
}

fn parse_s3_scheme(url: &Url) -> io::Result<S3Location> {
    let bucket = url
        .host_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "missing S3 bucket"))?
        .to_string();
    let key = url.path().trim_start_matches('/').to_string();
    let region = url.query_pairs().find_map(|(k, v)| {
        if k == "region" {
            Some(v.into_owned())
        } else {
            None
        }
    });
    Ok(S3Location {
        bucket,
        key,
        region,
    })
}

fn to_io_error<E: std::fmt::Display>(err: E) -> io::Error {
    io::Error::new(io::ErrorKind::Other, err.to_string())
}
