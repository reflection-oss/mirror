//! Where a session's files live, as a capability the runtime supplies.

use std::io;
use std::path::Component;
use std::path::Path;
use std::path::PathBuf;

use futures_util::future::BoxFuture;

/// The checkout a session operates on: which repository it is, how to name a
/// file inside it, and what is currently at a path.
///
/// Tools are already confined to one root; this is that root seen by a
/// feature that needs facts about it without reading the loop host's disk.
/// The local worker reads its own filesystem; a hosted session asks the
/// sandbox that holds the checkout.
pub trait Workspace: Send + Sync {
    /// The repository's stable name: its normalized origin URL, or the
    /// canonical toplevel path when there is no remote. `None` outside any
    /// repository.
    fn repository(&self) -> Option<&str>;

    /// `path` expressed relative to the checkout root with forward slashes,
    /// the form path globs match against. `None` outside the repository.
    fn relative(&self, path: &str) -> Option<String>;

    /// Whether each checkout-relative path is missing, a directory, a file or
    /// unreadable, as of now.
    fn inspect<'a>(&'a self, paths: &'a [String]) -> BoxFuture<'a, io::Result<Vec<PathState>>>;
}

/// What `Workspace::inspect` found at one path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathState {
    Missing,
    Directory,
    File,
    Unreadable,
}

/// `path` relative to `toplevel` with forward slashes, the form path globs
/// match against; relative input is taken against `workspace`. Purely
/// lexical: `.` and `..` are folded, symlinks are not followed and nothing is
/// read from disk, so it behaves the same over a sandbox as on this machine.
/// `None` when the path climbs above the filesystem root, lies outside
/// `toplevel` or is `toplevel` itself, or has a segment that is not UTF-8.
pub fn relative_within(toplevel: &Path, workspace: &Path, path: &Path) -> Option<String> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::Prefix(_) => return None,
            Component::RootDir | Component::Normal(_) => normalized.push(component),
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() {
                    return None;
                }
            }
        }
    }
    let relative = normalized.strip_prefix(toplevel).ok()?;
    let mut segments = Vec::new();
    for component in relative.components() {
        match component {
            // A lossy replacement could make two distinct filenames share
            // one glob scope; omit paths that cannot be represented exactly.
            Component::Normal(segment) => segments.push(segment.to_str()?),
            _ => return None,
        }
    }
    if segments.is_empty() {
        return None;
    }
    Some(segments.join("/"))
}
