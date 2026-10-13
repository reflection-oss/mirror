use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use super::repository_checkout::{git, validate_directory};
use super::validate_relative;

pub(super) struct RepositorySeed {
    git_directory: PathBuf,
    commit: String,
}

impl RepositorySeed {
    pub(super) async fn find(parent: &Path, url: &str) -> Option<Self> {
        let name = clone_directory(url)?;
        validate_directory(parent, &format!("{name}/.git/objects/pack")).ok()?;
        let seed = parent.join(name);
        let git_directory = seed.join(".git");
        if !is_self_contained(&git_directory) {
            return None;
        }
        let origin = git(&seed, &["config", "--get", "remote.origin.url"])
            .await
            .ok()?;
        if repository_identity(&origin) != repository_identity(url) {
            return None;
        }
        let commit = git(
            &seed,
            &["rev-parse", "--verify", "--quiet", "HEAD^{commit}"],
        )
        .await
        .ok()?;
        Some(Self {
            git_directory,
            commit,
        })
    }

    pub(super) fn share_objects(self, checkout: &Path) -> io::Result<String> {
        let target = checkout.join(".git/objects");
        for entry in fs::read_dir(self.git_directory.join("objects"))? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            let pack = name == "pack";
            let fanout = name.len() == 2 && name.bytes().all(|byte| byte.is_ascii_hexdigit());
            if (pack || fanout) && entry.file_type()?.is_dir() {
                share_directory(&entry.path(), &target.join(name), pack)?;
            }
        }
        let shallow = self.git_directory.join("shallow");
        if fs::symlink_metadata(&shallow).is_ok_and(|metadata| metadata.is_file()) {
            fs::copy(shallow, checkout.join(".git/shallow"))?;
        }
        Ok(self.commit)
    }
}

fn is_self_contained(git_directory: &Path) -> bool {
    let objects = git_directory.join("objects");
    fs::symlink_metadata(objects.join("info/alternates")).is_err()
        && fs::read_dir(objects.join("pack")).is_ok_and(|mut entries| {
            entries.all(|entry| {
                entry.is_ok_and(|entry| {
                    !entry.file_name().as_encoded_bytes().ends_with(b".promisor")
                })
            })
        })
}

fn share_directory(source: &Path, target: &Path, pack: bool) -> io::Result<()> {
    fs::create_dir_all(target)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let object = if pack {
            name.starts_with("pack-")
        } else {
            name.bytes().all(|byte| byte.is_ascii_hexdigit())
        };
        if object && entry.file_type()?.is_file() {
            let destination = target.join(name);
            if fs::hard_link(entry.path(), &destination).is_err() {
                fs::copy(entry.path(), &destination)?;
            }
        }
    }
    Ok(())
}

fn clone_directory(url: &str) -> Option<String> {
    let url = url::Url::parse(url).ok()?;
    let name = url
        .path_segments()?
        .rev()
        .find(|segment| !segment.is_empty())?;
    let name = name.strip_suffix(".git").unwrap_or(name);
    if name.is_empty() || name == "repository" || validate_relative(name).is_err() {
        return None;
    }
    Some(name.to_string())
}

fn repository_identity(url: &str) -> &str {
    let url = url.trim_end_matches('/');
    url.strip_suffix(".git").unwrap_or(url)
}
