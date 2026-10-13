//! Project-root and ancestor-directory discovery.

use std::path::Path;
use std::path::PathBuf;

use crate::is_repository_root;

/// Return directories from the nearest project root through `cwd`.
pub fn directories(cwd: &Path) -> Vec<PathBuf> {
    let walked = cwd.ancestors().map(Path::to_path_buf).collect::<Vec<_>>();
    match walked
        .iter()
        .position(|directory| is_repository_root(directory))
    {
        Some(index) => walked[..=index].iter().rev().cloned().collect(),
        None => vec![cwd.to_path_buf()],
    }
}

/// Whether `cwd` or an ancestor contains the project-root marker.
pub fn is_repository(cwd: &Path) -> bool {
    cwd.ancestors().any(is_repository_root)
}
