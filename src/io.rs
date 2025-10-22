use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use flate2::read::GzDecoder;

use aws_config::meta::region::RegionProviderChain;
use aws_sdk_s3::config::Region;
use aws_sdk_s3::Client;
use aws_smithy_types::byte_stream::ByteStream;
use bytes::Bytes;
use reqwest::blocking::Client as HttpClient;
use tempfile::NamedTempFile;
use tokio::runtime::{Builder as RuntimeBuilder, Runtime};
use url::Url;

/// Represents a readable handle to either a local file, an HTTP(S) resource, or an S3 object.
pub enum InputReader {
    Local(BufReader<File>),
    Http(BufReader<reqwest::blocking::Response>),
    S3(S3StreamReader),
}

impl Read for InputReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            InputReader::Local(reader) => reader.read(buf),
            InputReader::Http(reader) => reader.read(buf),
            InputReader::S3(reader) => reader.read(buf),
        }
    }
}

impl BufRead for InputReader {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        match self {
            InputReader::Local(reader) => reader.fill_buf(),
            InputReader::Http(reader) => reader.fill_buf(),
            InputReader::S3(reader) => reader.fill_buf(),
        }
    }

    fn consume(&mut self, amt: usize) {
        match self {
            InputReader::Local(reader) => reader.consume(amt),
            InputReader::Http(reader) => reader.consume(amt),
            InputReader::S3(reader) => reader.consume(amt),
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
            InputReader::S3(_) => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "S3 streams do not support seeking",
            )),
        }
    }
}

/// A handle to a local file path. Remote resources are materialized to a temporary file
/// and kept alive while the handle exists.
pub struct MaterializedPath {
    path: PathBuf,
    #[allow(dead_code)]
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

/// Blocking adapter around the AWS SDK's streaming body for S3 objects.
pub struct S3StreamReader {
    runtime: Runtime,
    stream: ByteStream,
    buffer: Bytes,
    pos: usize,
    eof: bool,
}

impl S3StreamReader {
    fn new(runtime: Runtime, stream: ByteStream) -> Self {
        Self {
            runtime,
            stream,
            buffer: Bytes::new(),
            pos: 0,
            eof: false,
        }
    }

    fn ensure_buffered(&mut self) -> io::Result<()> {
        if self.pos < self.buffer.len() || self.eof {
            return Ok(());
        }

        loop {
            let fut = self.stream.try_next();
            match self.runtime.block_on(fut).map_err(to_io_error)? {
                Some(chunk) => {
                    if chunk.is_empty() {
                        // Skip empty chunks and poll the stream again.
                        continue;
                    }
                    self.buffer = chunk;
                    self.pos = 0;
                    return Ok(());
                }
                None => {
                    self.buffer = Bytes::new();
                    self.pos = 0;
                    self.eof = true;
                    return Ok(());
                }
            }
        }
    }
}

impl Read for S3StreamReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        self.ensure_buffered()?;
        if self.pos >= self.buffer.len() {
            return Ok(0);
        }

        let available = &self.buffer[self.pos..];
        let len = available.len().min(buf.len());
        buf[..len].copy_from_slice(&available[..len]);
        self.pos += len;
        if self.pos >= self.buffer.len() {
            self.buffer = Bytes::new();
            self.pos = 0;
        }
        Ok(len)
    }
}

impl BufRead for S3StreamReader {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        self.ensure_buffered()?;
        Ok(&self.buffer[self.pos..])
    }

    fn consume(&mut self, amt: usize) {
        self.pos = (self.pos + amt).min(self.buffer.len());
        if self.pos >= self.buffer.len() {
            self.buffer = Bytes::new();
            self.pos = 0;
        }
    }
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
    let runtime = RuntimeBuilder::new_current_thread()
        .enable_all()
        .build()
        .map_err(to_io_error)?;
    let bucket = loc.bucket.clone();
    let key = loc.key.clone();
    let region = loc.region.clone();
    let body = runtime.block_on(async move {
        let region_provider = match region {
            Some(region_name) => RegionProviderChain::first_try(Region::new(region_name))
                .or_default_provider()
                .or_else("us-east-1"),
            None => RegionProviderChain::default_provider().or_else("us-east-1"),
        };
        let config = aws_config::defaults(aws_config::BehaviorVersion::latest()).region(region_provider).load().await;
        let client = Client::new(&config);
        let obj = client
            .get_object()
            .bucket(&bucket)
            .key(&key)
            .send()
            .await
            .map_err(to_io_error)?;
        Ok::<ByteStream, io::Error>(obj.body)
    })?;
    Ok(InputReader::S3(S3StreamReader::new(runtime, body)))
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
        let config = aws_config::defaults(aws_config::BehaviorVersion::latest()).region(region_provider).load().await;
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
    io::Error::other(err.to_string())
}

/// A streaming GFA reader that can efficiently extract specific nodes
/// without downloading the entire file
pub struct GfaReader {
    source: String,
}

impl GfaReader {
    /// Open a GFA file (local, HTTP, or S3)
    pub fn new(path_or_url: &str) -> Self {
        Self {
            source: path_or_url.to_string(),
        }
    }

    /// Extract sequences for specific node IDs
    /// This streams through the GFA file and stops early when all nodes are found
    pub fn extract_sequences(
        &self,
        node_ids: &HashSet<String>,
    ) -> io::Result<HashMap<String, String>> {
        let reader = open(&self.source)?;
        let mut sequences = HashMap::new();
        let mut found_count = 0;
        let target_count = node_ids.len();

        eprintln!(
            "[INFO] Streaming GFA from {} looking for {} nodes",
            self.source, target_count
        );

        // Check if file is gzipped based on extension
        let is_gzipped = self.source.ends_with(".gz");
        
        let mut line_count = 0;
        
        if is_gzipped {
            // Decompress on the fly
            let decoder = GzDecoder::new(reader);
            let buf_reader = BufReader::new(decoder);
            
            for line_result in buf_reader.lines() {
                let line = line_result?;
                line_count += 1;

                if !line.starts_with('S') {
                    continue;
                }

                let parts: Vec<&str> = line.split('\t').collect();
                if parts.len() < 3 {
                    continue;
                }

                let node_id = parts[1];
                if node_ids.contains(node_id) {
                    let sequence = parts[2];
                    if sequence != "*" {
                        sequences.insert(node_id.to_string(), sequence.to_string());
                        found_count += 1;

                        if found_count % 10 == 0 || found_count == target_count {
                            eprintln!(
                                "[INFO] Found {}/{} target nodes (scanned {} lines)",
                                found_count, target_count, line_count
                            );
                        }

                        // Early exit optimization
                        if found_count >= target_count {
                            eprintln!(
                                "[INFO] Found all {} target nodes, stopping scan at line {}",
                                target_count, line_count
                            );
                            break;
                        }
                    }
                }
            }
        } else {
            // Read uncompressed
            for line_result in reader.lines() {
                let line = line_result?;
                line_count += 1;

                if !line.starts_with('S') {
                    continue;
                }

                let parts: Vec<&str> = line.split('\t').collect();
                if parts.len() < 3 {
                    continue;
                }

                let node_id = parts[1];
                if node_ids.contains(node_id) {
                    let sequence = parts[2];
                    if sequence != "*" {
                        sequences.insert(node_id.to_string(), sequence.to_string());
                        found_count += 1;

                        if found_count % 10 == 0 || found_count == target_count {
                            eprintln!(
                                "[INFO] Found {}/{} target nodes (scanned {} lines)",
                                found_count, target_count, line_count
                            );
                        }

                        // Early exit optimization
                        if found_count >= target_count {
                            eprintln!(
                                "[INFO] Found all {} target nodes, stopping scan at line {}",
                                target_count, line_count
                            );
                            break;
                        }
                    }
                }
            }
        }

        if found_count < target_count {
            eprintln!(
                "[WARNING] Only found {}/{} target nodes after scanning {} lines",
                found_count, target_count, line_count
            );
        }

        Ok(sequences)
    }
}
