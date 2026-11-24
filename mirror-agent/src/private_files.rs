//! Directories only the current user can reach.

use std::fs::{self, DirBuilder};
use std::io;
use std::os::unix::fs::{DirBuilderExt, MetadataExt};
use std::path::Path;

use nix::unistd::geteuid;

/// Create `path` as a private directory if it is missing, then check it.
pub fn private_directory(path: &Path) -> io::Result<()> {
    match DirBuilder::new().recursive(true).mode(0o700).create(path) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error),
    }
    check_private_directory(path)
}

/// Require a real directory (not a symlink) owned by this user that no one else can enter.
pub fn check_private_directory(path: &Path) -> io::Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_dir() || metadata.uid() != geteuid().as_raw() || metadata.mode() & 0o077 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!(
                "{} must be a private directory (0700) owned by this user, without symlinks",
                path.display()
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn only_a_real_directory_no_one_else_can_enter_is_private() {
        let root = tempfile::tempdir().expect("root");
        let directory = root.path().join("state");
        private_directory(&directory).expect("created private");
        let link = root.path().join("link");
        std::os::unix::fs::symlink(&directory, &link).expect("symlink");

        assert_matches!(check_private_directory(&link), Err(_));
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o750)).expect("loosen");
        assert_eq!(
            private_directory(&directory).map_err(|error| error.kind()),
            Err(io::ErrorKind::PermissionDenied)
        );
    }
}
