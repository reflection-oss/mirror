//! Finding the llama.cpp server binary: `llama-server`, or the unified `llama` binary.

use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};

/// Find llama.cpp's server: an explicit path, else the user's `PATH`, else standard install
/// locations.
///
/// Every directory is searched for `llama-server` before any is searched for the unified
/// `llama` binary, so an unrelated tool named `llama` is used only when there is no
/// `llama-server`.
///
/// A discovered file, and the directory holding it, must belong to the user or root and not
/// be writable by others, so a shared directory on `PATH` cannot supply or replace the
/// binary. An explicit path is trusted.
pub fn executable(configured: Option<PathBuf>, search_path: &str, home: &Path) -> Option<PathBuf> {
    discover(
        configured,
        std::env::split_paths(search_path)
            .chain([home.join(".local/bin")])
            .chain(STANDARD_DIRECTORIES.map(PathBuf::from)),
    )
}

/// Where package managers put llama-server, searched after the user's own directories.
const STANDARD_DIRECTORIES: [&str; 4] = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"];

/// The unified llama.cpp binary, which runs the server as `llama serve`.
const UNIFIED_BINARY: &str = "llama";

fn discover(
    configured: Option<PathBuf>,
    directories: impl IntoIterator<Item = PathBuf>,
) -> Option<PathBuf> {
    if let Some(path) = configured {
        return runnable(&path);
    }
    let directories: Vec<PathBuf> = directories.into_iter().collect();
    ["llama-server", UNIFIED_BINARY]
        .into_iter()
        .find_map(|name| {
            directories
                .iter()
                .filter_map(|directory| runnable(&directory.join(name)))
                .find(|path| trusted(path))
        })
}

/// The arguments that select the server ahead of its own flags.
pub fn serve_prefix(binary: &Path) -> &'static [&'static str] {
    if binary
        .file_name()
        .is_some_and(|name| name == UNIFIED_BINARY)
    {
        &["serve"]
    } else {
        &[]
    }
}

/// The canonical path of an absolute executable file.
fn runnable(path: &Path) -> Option<PathBuf> {
    let metadata = path.metadata().ok()?;
    if !path.is_absolute() || !metadata.is_file() || metadata.permissions().mode() & 0o111 == 0 {
        return None;
    }
    path.canonicalize().ok()
}

/// Whether only the user or root can change the canonical file at `path`.
fn trusted(path: &Path) -> bool {
    path.parent().is_some_and(owner_only_writable) && owner_only_writable(path)
}

fn owner_only_writable(path: &Path) -> bool {
    path.metadata().is_ok_and(|metadata| {
        metadata.permissions().mode() & 0o022 == 0
            && [nix::unistd::geteuid().as_raw(), 0].contains(&metadata.uid())
    })
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    fn fake_server(directory: &Path, mode: u32) -> PathBuf {
        fake_binary(directory, "llama-server", mode)
    }

    fn fake_binary(directory: &Path, name: &str, mode: u32) -> PathBuf {
        std::fs::create_dir_all(directory).expect("directory");
        std::fs::set_permissions(directory, std::fs::Permissions::from_mode(0o755))
            .expect("directory mode");
        let path = directory.join(name);
        std::fs::write(&path, b"#!/bin/sh\n").expect("binary");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).expect("mode");
        path.canonicalize().expect("canonical")
    }

    #[test]
    fn the_first_binary_others_cannot_write_or_replace_is_found_in_search_order() {
        let root = tempfile::tempdir().expect("root");
        fake_server(&root.path().join("shared"), 0o775);
        let open = root.path().join("open");
        fake_server(&open, 0o755);
        std::fs::set_permissions(&open, std::fs::Permissions::from_mode(0o777)).expect("mode");
        let first = fake_server(&root.path().join("first"), 0o755);
        fake_server(&root.path().join("second"), 0o755);

        let found = discover(
            None,
            ["shared", "open", "first", "second"].map(|name| root.path().join(name)),
        );

        assert_eq!(found, Some(first));
    }

    #[test]
    fn an_explicit_path_is_used_without_the_ownership_check() {
        let root = tempfile::tempdir().expect("root");
        let explicit = fake_server(&root.path().join("custom"), 0o775);

        assert_eq!(discover(Some(explicit.clone()), []), Some(explicit));
    }

    #[test]
    fn a_binary_in_the_local_bin_directory_is_found() {
        let root = tempfile::tempdir().expect("root");
        let local = fake_server(&root.path().join(".local/bin"), 0o755);

        assert_eq!(executable(None, "", root.path()), Some(local));
    }

    #[test]
    fn the_unified_binary_is_found_when_no_llama_server_exists() {
        let root = tempfile::tempdir().expect("root");
        let directory = root.path().join("bin");
        let unified = fake_binary(&directory, "llama", 0o755);

        assert_eq!(discover(None, [directory]), Some(unified));
    }

    #[test]
    fn a_llama_server_in_a_later_directory_wins_over_the_unified_binary() {
        let root = tempfile::tempdir().expect("root");
        fake_binary(&root.path().join("first"), "llama", 0o755);
        let server = fake_server(&root.path().join("second"), 0o755);

        let found = discover(None, ["first", "second"].map(|name| root.path().join(name)));

        assert_eq!(found, Some(server));
    }

    #[test]
    fn only_the_unified_binary_is_served_with_a_subcommand() {
        assert_eq!(serve_prefix(Path::new("/bin/llama")), &["serve"]);
        assert_eq!(serve_prefix(Path::new("/bin/llama-server")), &[] as &[&str]);
    }
}
