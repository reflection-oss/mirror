use std::io::Write;

use assert_matches::assert_matches;
use pretty_assertions::assert_eq;

use super::*;

fn file_storage(directory: &tempfile::TempDir) -> (CaptureStorage, std::path::PathBuf) {
    let path = directory.path().join("capture");
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .expect("create capture file");
    (CaptureStorage::File { file, written: 0 }, path)
}

#[test]
fn a_failed_write_moves_to_memory_without_its_partial_bytes() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let (mut storage, path) = file_storage(&directory);
    storage.write("caf".as_bytes());
    // A write that failed partway, cutting `é` in two.
    std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .and_then(|mut file| file.write_all(&"é".as_bytes()[..1]))
        .expect("append partial write");

    storage.fall_back_to_memory("é au lait".as_bytes());
    storage.write(b"!");

    assert_matches!(storage, CaptureStorage::Memory(_));
    assert_eq!(
        String::from_utf8(storage.read_all().expect("read kept text")),
        Ok("café au lait!".to_string())
    );
}

#[test]
fn a_file_that_cannot_be_written_moves_to_memory() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("capture");
    std::fs::write(&path, b"").expect("create capture file");
    let file = std::fs::File::open(&path).expect("open read-only");
    let mut storage = CaptureStorage::File { file, written: 0 };

    storage.write(b"kept");

    assert_eq!(storage.read_all().expect("read kept text"), b"kept");
}

#[test]
fn text_that_cannot_be_read_back_reports_its_loss() {
    let directory = tempfile::tempdir().expect("temporary directory");
    // A directory can be opened but neither written nor read, so the bytes
    // it claims to hold cannot be recovered.
    let file = std::fs::File::open(directory.path()).expect("open directory");
    let mut storage = CaptureStorage::File { file, written: 4 };

    storage.write(b"lost");
    storage.write(b"also lost");

    assert_matches!(storage.read_all(), Err(error) if error.to_string().starts_with("captured output was lost"));
}

#[test]
fn clearing_lost_text_starts_over_in_memory() {
    let mut storage = CaptureStorage::Failed {
        kind: std::io::ErrorKind::Other,
        message: "disk failed".to_string(),
    };

    storage.clear();
    storage.write(b"fresh");

    assert_eq!(storage.read_all().expect("read kept text"), b"fresh");
}
