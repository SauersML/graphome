use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use flate2::read::GzDecoder;

use aws_sdk_s3::config::Region;
use aws_sdk_s3::Client;
use aws_smithy_types::byte_stream::ByteStream;
use bytes::Bytes;
use indicatif::{HumanBytes, ProgressBar};
use reqwest::blocking::Client as HttpClient;
use tempfile::NamedTempFile;
use tokio::runtime::{Builder as RuntimeBuilder, Runtime};
use url::Url;

use crate::progress::{byte_progress_bar, count_progress_bar};
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
    temp: Option<NamedTempFile>,
}

impl MaterializedPath {
    /// Returns the local path that can be used by APIs requiring filesystem access.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

static RETAINED_MATERIALIZED: OnceLock<Mutex<Vec<MaterializedPath>>> = OnceLock::new();

/// Retains a materialized file for the remainder of the process so that temporary files
/// downloaded from remote locations remain accessible after the handle goes out of scope.
pub fn retain_materialized(materialized: MaterializedPath) -> PathBuf {
    let path = materialized.path.clone();
    if materialized.temp.is_some() {
        let cache = RETAINED_MATERIALIZED.get_or_init(|| Mutex::new(Vec::new()));
        let mut guard = cache.lock().expect("materialized cache mutex poisoned");
        guard.push(materialized);
    }
    path
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
    total_read: u64,
    progress: ProgressBar,
    progress_done: bool,
    label: String,
}

impl S3StreamReader {
    fn new(runtime: Runtime, stream: ByteStream, total_size: Option<u64>, label: String) -> Self {
        let progress = byte_progress_bar(format!("Streaming {label}"), total_size);
        progress.set_message("from S3".to_string());
        Self {
            runtime,
            stream,
            buffer: Bytes::new(),
            pos: 0,
            eof: false,
            total_read: 0,
            progress,
            progress_done: false,
            label,
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
                    self.total_read += chunk.len() as u64;
                    self.progress.set_position(self.total_read);
                    self.buffer = chunk;
                    self.pos = 0;
                    return Ok(());
                }
                None => {
                    self.buffer = Bytes::new();
                    self.pos = 0;
                    self.eof = true;
                    if !self.progress_done {
                        self.progress.finish_with_message(format!(
                            "Completed streaming {label} ({})",
                            HumanBytes(self.total_read),
                            label = self.label,
                        ));
                        self.progress_done = true;
                    }
                    return Ok(());
                }
            }
        }
    }
}

impl Drop for S3StreamReader {
    fn drop(&mut self) {
        if !self.progress_done {
            self.progress.abandon_with_message(format!(
                "Streaming {} interrupted at {}",
                self.label,
                HumanBytes(self.total_read)
            ));
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
    let total = response.content_length();
    let label = format!("HTTP {}", url);
    let pb = byte_progress_bar(label, total);
    pb.set_message("downloading".to_string());
    let mut downloaded = 0u64;
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    loop {
        let bytes_read = response.read(&mut buffer).map_err(to_io_error)?;
        if bytes_read == 0 {
            break;
        }
        temp.as_file_mut().write_all(&buffer[..bytes_read])?;
        downloaded += bytes_read as u64;
        pb.set_position(downloaded);
    }
    pb.finish_with_message(format!(
        "Completed HTTP download ({})",
        HumanBytes(downloaded)
    ));
    temp.as_file_mut().seek(SeekFrom::Start(0))?;
    let path = temp.path().to_path_buf();
    Ok(MaterializedPath {
        path,
        temp: Some(temp),
    })
}

fn proxy_required_for_s3() -> bool {
    const PROXY_ENV_VARS: [&str; 4] = ["HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"];
    PROXY_ENV_VARS
        .iter()
        .filter_map(|name| env::var(name).ok())
        .any(|value| !value.trim().is_empty())
}

fn public_s3_https_url(bucket: &str, key: &str) -> io::Result<Url> {
    let mut url =
        Url::parse(&format!("https://{bucket}.s3.amazonaws.com/")).map_err(to_io_error)?;
    url.set_path(key);
    Ok(url)
}

fn open_s3(loc: &S3Location) -> io::Result<InputReader> {
    let resource_label = format!("s3://{}/{}", loc.bucket, loc.key);

    if proxy_required_for_s3() {
        let url = public_s3_https_url(&loc.bucket, &loc.key)?;
        eprintln!(
            "[INFO] Detected HTTPS proxy configuration; streaming {} via public S3 HTTPS endpoint",
            resource_label
        );
        return open_http(&url);
    }

    let runtime = RuntimeBuilder::new_current_thread()
        .enable_all()
        .build()
        .map_err(to_io_error)?;
    let bucket = loc.bucket.clone();
    let key = loc.key.clone();
    let region = loc.region.clone();
    let (body, total) = runtime.block_on(async move {
        // For public buckets, use no credentials
        // Region will be auto-detected from the bucket's location
        let mut config_loader =
            aws_config::defaults(aws_config::BehaviorVersion::latest()).no_credentials();

        if let Some(region_name) = region {
            config_loader = config_loader.region(Region::new(region_name));
        }

        let config = config_loader.load().await;

        // Create S3 client - it should handle cross-region redirects automatically
        let client = Client::new(&config);
        let obj = client
            .get_object()
            .bucket(&bucket)
            .key(&key)
            .send()
            .await
            .map_err(to_io_error)?;
        let total = obj
            .content_length()
            .and_then(|len| if len >= 0 { Some(len as u64) } else { None });
        Ok::<(ByteStream, Option<u64>), io::Error>((obj.body, total))
    })?;
    Ok(InputReader::S3(S3StreamReader::new(
        runtime,
        body,
        total,
        resource_label,
    )))
}

fn download_s3(loc: &S3Location) -> io::Result<MaterializedPath> {
    if proxy_required_for_s3() {
        let url = public_s3_https_url(&loc.bucket, &loc.key)?;
        eprintln!(
            "[INFO] Detected HTTPS proxy configuration; downloading s3://{}/{} via public S3 HTTPS endpoint",
            loc.bucket, loc.key
        );
        return download_http(&url);
    }

    let rt = RuntimeBuilder::new_current_thread()
        .enable_all()
        .build()
        .map_err(to_io_error)?;
    let label = format!("s3://{}/{}", loc.bucket, loc.key);
    let bucket = loc.bucket.clone();
    let key = loc.key.clone();
    let region = loc.region.clone();
    let mut temp = NamedTempFile::new()?;
    let download_result = rt.block_on(async {
        // For public buckets, use no credentials
        // Region will be auto-detected from the bucket's location
        let mut config_loader =
            aws_config::defaults(aws_config::BehaviorVersion::latest()).no_credentials();

        if let Some(region_name) = region {
            config_loader = config_loader.region(Region::new(region_name));
        }

        let config = config_loader.load().await;

        // Create S3 client - it should handle cross-region redirects automatically
        let client = Client::new(&config);
        // Try to get the object, handling region redirects
        let mut obj = match client.get_object().bucket(&bucket).key(&key).send().await {
            Ok(obj) => obj,
            Err(e) => {
                // Check if this is a redirect error with region information
                let error_str = format!("{:?}", e);
                if error_str.contains("PermanentRedirect")
                    && error_str.contains("x-amz-bucket-region")
                {
                    // Extract region from error (it's in the headers)
                    // The error message contains: "x-amz-bucket-region": HeaderValue { _private: H1("us-west-2") }
                    if let Some(start) = error_str.find("x-amz-bucket-region") {
                        if let Some(region_start) = error_str[start..].find("H1(\"") {
                            if let Some(region_end) =
                                error_str[start + region_start + 4..].find("\"")
                            {
                                let detected_region = &error_str[start + region_start + 4
                                    ..start + region_start + 4 + region_end];
                                eprintln!(
                                    "[INFO] Bucket is in region {}, retrying...",
                                    detected_region
                                );

                                // Recreate client with correct region
                                let config =
                                    aws_config::defaults(aws_config::BehaviorVersion::latest())
                                        .no_credentials()
                                        .region(Region::new(detected_region.to_string()))
                                        .load()
                                        .await;
                                let client = Client::new(&config);

                                // Retry the request
                                client
                                    .get_object()
                                    .bucket(&bucket)
                                    .key(&key)
                                    .send()
                                    .await
                                    .map_err(to_io_error)?
                            } else {
                                return Err(to_io_error(e));
                            }
                        } else {
                            return Err(to_io_error(e));
                        }
                    } else {
                        return Err(to_io_error(e));
                    }
                } else {
                    return Err(to_io_error(e));
                }
            }
        };
        let total = obj
            .content_length()
            .and_then(|len| if len >= 0 { Some(len as u64) } else { None });
        let pb = byte_progress_bar(format!("Downloading {label}"), total);
        pb.set_message("from S3".to_string());
        let mut downloaded = 0u64;
        let mut result = Ok(());
        while let Some(bytes) = obj.body.try_next().await.map_err(to_io_error)? {
            if let Err(err) = temp.as_file_mut().write_all(&bytes) {
                result = Err(err);
                break;
            }
            downloaded += bytes.len() as u64;
            pb.set_position(downloaded);
        }
        match result {
            Ok(()) => {
                pb.finish_with_message(format!(
                    "Completed S3 download ({})",
                    HumanBytes(downloaded)
                ));
                Ok::<(), io::Error>(())
            }
            Err(err) => {
                pb.abandon_with_message(format!(
                    "S3 download failed after {}",
                    HumanBytes(downloaded)
                ));
                Err(err)
            }
        }
    });
    download_result?;
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
        let progress_label = format!("Streaming {}", self.source);
        let pb = count_progress_bar(progress_label, "lines", None);
        pb.set_message(format!("0/{target_count} nodes"));

        if is_gzipped {
            // Decompress on the fly
            let decoder = GzDecoder::new(reader);
            let buf_reader = BufReader::new(decoder);

            for line_result in buf_reader.lines() {
                let line = match line_result {
                    Ok(line) => line,
                    Err(err) => {
                        pb.abandon_with_message(format!(
                            "Streaming failed after {} lines",
                            line_count
                        ));
                        return Err(err);
                    }
                };
                line_count += 1;
                pb.inc(1);

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
                            pb.set_message(format!("{}/{} nodes", found_count, target_count));
                        }

                        // Early exit optimization
                        if found_count >= target_count {
                            eprintln!(
                                "[INFO] Found all {} target nodes, stopping scan at line {}",
                                target_count, line_count
                            );
                            pb.set_message(format!(
                                "{}/{} nodes — early stop",
                                found_count, target_count
                            ));
                            break;
                        }
                    }
                }
            }
        } else {
            // Read uncompressed
            for line_result in reader.lines() {
                let line = match line_result {
                    Ok(line) => line,
                    Err(err) => {
                        pb.abandon_with_message(format!(
                            "Streaming failed after {} lines",
                            line_count
                        ));
                        return Err(err);
                    }
                };
                line_count += 1;
                pb.inc(1);

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
                            pb.set_message(format!("{}/{} nodes", found_count, target_count));
                        }

                        // Early exit optimization
                        if found_count >= target_count {
                            eprintln!(
                                "[INFO] Found all {} target nodes, stopping scan at line {}",
                                target_count, line_count
                            );
                            pb.set_message(format!(
                                "{}/{} nodes — early stop",
                                found_count, target_count
                            ));
                            break;
                        }
                    }
                }
            }
        }

        pb.finish_with_message(format!(
            "Scanned {} lines — found {}/{} nodes",
            line_count, found_count, target_count
        ));

        if found_count < target_count {
            eprintln!(
                "[WARNING] Only found {}/{} target nodes after scanning {} lines",
                found_count, target_count, line_count
            );
        }

        Ok(sequences)
    }
}
