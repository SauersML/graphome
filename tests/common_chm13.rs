use reqwest::blocking::Client;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant};
use tempfile::NamedTempFile;

const CHM13_GBZ_URL: &str = "https://s3-us-west-2.amazonaws.com/human-pangenomics/pangenomes/freeze/release2/minigraph-cactus/hprc-v2.0-mc-chm13.gbz";
const CHM13_GBZ_NAME: &str = "hprc-v2.0-mc-chm13.gbz";
const LOCK_TIMEOUT: Duration = Duration::from_secs(60 * 60);
const LOCK_STALE_AGE: Duration = Duration::from_secs(2 * 60 * 60);

static CHM13_GBZ_PATH: OnceLock<Result<PathBuf, String>> = OnceLock::new();

pub fn chm13_fixture_path_if_present() -> Option<PathBuf> {
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let primary = project_root.join("data/hprc").join(CHM13_GBZ_NAME);
    if primary.exists() {
        return Some(primary);
    }
    let fallback = project_root.join("data").join(CHM13_GBZ_NAME);
    if fallback.exists() {
        return Some(fallback);
    }
    None
}

pub fn should_run_large_integration() -> bool {
    std::env::var("RUN_LARGE_INTEGRATION")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            normalized == "1" || normalized == "true" || normalized == "yes"
        })
        .unwrap_or(false)
}

pub fn ensure_chm13_gbz() -> Result<PathBuf, Box<dyn std::error::Error>> {
    match CHM13_GBZ_PATH.get_or_init(resolve_or_download_chm13_gbz) {
        Ok(path) => Ok(path.clone()),
        Err(err) => Err(std::io::Error::other(err.clone()).into()),
    }
}

fn resolve_or_download_chm13_gbz() -> Result<PathBuf, String> {
    if let Some(existing) = chm13_fixture_path_if_present() {
        return Ok(existing);
    }

    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let primary = project_root.join("data/hprc").join(CHM13_GBZ_NAME);
    let fallback = project_root.join("data").join(CHM13_GBZ_NAME);
    if !should_run_large_integration() {
        return Err(
            "CHM13 fixture missing; set RUN_LARGE_INTEGRATION=1 to enable auto-download".into(),
        );
    }

    fs::create_dir_all(primary.parent().ok_or("invalid cache path")?)
        .map_err(|e| format!("failed to create data directory: {e}"))?;

    {
        let lock_guard = acquire_download_lock(&primary)?;
        assert!(
            !lock_guard.path.as_os_str().is_empty(),
            "download lock path must be non-empty"
        );
        if primary.exists() {
            return Ok(primary);
        }
        if fallback.exists() {
            return Ok(fallback);
        }

        let client = Client::builder()
            .timeout(Duration::from_secs(60 * 60))
            .build()
            .map_err(|e| format!("failed to create HTTP client: {e}"))?;

        let mut response = client
            .get(CHM13_GBZ_URL)
            .send()
            .and_then(reqwest::blocking::Response::error_for_status)
            .map_err(|e| format!("failed to download CHM13 GBZ: {e}"))?;

        let parent = primary.parent().ok_or("invalid primary parent")?;
        let mut tmp = NamedTempFile::new_in(parent)
            .map_err(|e| format!("failed to create temp file in {:?}: {}", parent, e))?;
        std::io::copy(&mut response, &mut tmp).map_err(|e| format!("download copy failed: {e}"))?;
        tmp.flush()
            .map_err(|e| format!("flush failed for {:?}: {}", tmp.path(), e))?;
        tmp.as_file()
            .sync_all()
            .map_err(|e| format!("sync failed for {:?}: {}", tmp.path(), e))?;

        if !primary.exists() {
            tmp.persist(&primary).map_err(|e| {
                format!(
                    "failed to move temporary file into {:?}: {}",
                    primary, e.error
                )
            })?;
        }
    }
    Ok(primary)
}

struct LockFile {
    path: PathBuf,
}

impl Drop for LockFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn acquire_download_lock(primary: &Path) -> Result<LockFile, String> {
    let lock_path = primary.with_extension("gbz.lock");
    let start = Instant::now();
    loop {
        match OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&lock_path)
        {
            Ok(mut file) => {
                let _ = writeln!(file, "pid={}", std::process::id());
                return Ok(LockFile { path: lock_path });
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                if let Ok(meta) = fs::metadata(&lock_path) {
                    if let Ok(modified) = meta.modified() {
                        if modified.elapsed().unwrap_or_default() > LOCK_STALE_AGE {
                            let _ = fs::remove_file(&lock_path);
                            continue;
                        }
                    }
                }

                if start.elapsed() > LOCK_TIMEOUT {
                    return Err(format!(
                        "timed out waiting for CHM13 fixture lock {:?}",
                        lock_path
                    ));
                }
                thread::sleep(Duration::from_millis(250));
            }
            Err(err) => {
                return Err(format!(
                    "failed to acquire CHM13 fixture lock {:?}: {}",
                    lock_path, err
                ));
            }
        }
    }
}
