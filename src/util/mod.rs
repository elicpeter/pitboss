//! Small utilities shared across modules.

pub mod paths;

use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};

/// Atomically write `bytes` to `path`.
///
/// The contents are written to a sibling temp file and `fsync`ed before being
/// renamed onto the target. A crash mid-write leaves either the old contents or
/// the new contents — never a half-written file. Used for every mutation of
/// `plan.md`, `deferred.md`, and `state.json`.
pub fn write_atomic(path: impl AsRef<Path>, bytes: &[u8]) -> Result<()> {
    let path = path.as_ref();
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .with_context(|| format!("write_atomic: path {:?} has no file name", path))?;

    let mut tmp_name = OsString::from(".");
    tmp_name.push(file_name);
    tmp_name.push(".tmp");
    let tmp_path = dir.join(&tmp_name);

    {
        let mut tmp = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)
            .with_context(|| format!("write_atomic: opening {:?}", tmp_path))?;
        tmp.write_all(bytes)
            .with_context(|| format!("write_atomic: writing {:?}", tmp_path))?;
        tmp.sync_all()
            .with_context(|| format!("write_atomic: fsync {:?}", tmp_path))?;
    }

    fs::rename(&tmp_path, path)
        .with_context(|| format!("write_atomic: rename {:?} -> {:?}", tmp_path, path))?;

    // Best-effort: fsync the parent directory so the rename is durable. Not all
    // filesystems support opening a directory for fsync; if it fails we don't
    // surface the error because the rename itself already succeeded.
    let _ = File::open(dir).and_then(|d| d.sync_all());

    Ok(())
}

#[cfg(test)]
mod write_atomic_tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn writes_bytes_to_target() {
        let dir = tempdir().unwrap();
        let target = dir.path().join("hello.txt");
        write_atomic(&target, b"hello world").unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"hello world");
    }

    #[test]
    fn overwrites_existing_file() {
        let dir = tempdir().unwrap();
        let target = dir.path().join("file.txt");
        fs::write(&target, b"old").unwrap();
        write_atomic(&target, b"new").unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"new");
    }

    #[test]
    fn does_not_leave_temp_file_on_success() {
        let dir = tempdir().unwrap();
        let target = dir.path().join("file.txt");
        write_atomic(&target, b"data").unwrap();
        let leftover: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with(".file.txt.tmp"))
            .collect();
        assert!(leftover.is_empty(), "temp file should be renamed away");
    }

    #[test]
    fn crash_resistance_preserves_old_contents_when_temp_left_behind() {
        // Simulates a crash *between* the temp-write and the rename: the temp
        // file exists but the target retains its original contents. A
        // subsequent successful write_atomic must still produce the new
        // contents (and remove the temp file).
        let dir = tempdir().unwrap();
        let target = dir.path().join("doc.md");
        fs::write(&target, b"original").unwrap();

        // Pre-place a stale temp file to mimic an interrupted prior run.
        let stale_tmp = dir.path().join(".doc.md.tmp");
        fs::write(&stale_tmp, b"garbage from crashed previous run").unwrap();

        // Original is still intact since the rename never happened.
        assert_eq!(fs::read(&target).unwrap(), b"original");

        // Now a successful atomic write must clobber the stale temp and land
        // the new payload.
        write_atomic(&target, b"recovered").unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"recovered");
        assert!(!stale_tmp.exists(), "stale temp file should be replaced");
    }
}
