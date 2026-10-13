//! Claude Code auto-memory: a directory of Markdown notes with YAML
//! frontmatter (`name`, `description`, `metadata.type`) and a `MEMORY.md`
//! index. Superseded notes live under `old/` and are not loaded.

use std::path::Path;
use std::path::PathBuf;
use std::sync::LazyLock;

use regex::Regex;
use serde::Deserialize;

use super::ImportedNote;
use crate::memory::slug_key;
use crate::memory::MemoryResult;
use crate::memory::MemoryType;

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Frontmatter {
    name: Option<String>,
    description: Option<String>,
    metadata: Option<Metadata>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Metadata {
    r#type: Option<String>,
}

/// Every note under `directory` in stable path order, index and `old/`
/// excluded. A note's observation time is its file's modification time.
pub fn claude_code_notes(directory: &Path) -> MemoryResult<Vec<ImportedNote>> {
    let mut files = Vec::new();
    collect_markdown(directory, &mut files)?;
    files.sort();
    let mut notes = Vec::new();
    for path in files {
        let text = std::fs::read_to_string(&path)?;
        let (front, body) = split_frontmatter(&text);
        let front: Frontmatter = front
            .and_then(|yaml| serde_yaml::from_str(yaml).ok())
            .unwrap_or_default();
        let body = body.trim();
        if body.is_empty() {
            continue;
        }
        let stem = path
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_default();
        let name = front
            .name
            .filter(|name| !name.trim().is_empty())
            .unwrap_or(stem);
        let kind = match front
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.r#type.as_deref())
        {
            Some("feedback") => MemoryType::Preference,
            Some("user" | "project") => MemoryType::Fact,
            _ => MemoryType::Reference,
        };
        notes.push(ImportedNote {
            key: slug_key(&name),
            kind,
            situation: front.description.unwrap_or_default(),
            content: body.to_string(),
            keywords: wiki_links(body),
            observed_at: modified_at(&path),
            file: path
                .strip_prefix(directory)
                .unwrap_or(&path)
                .to_string_lossy()
                .into_owned(),
        });
    }
    Ok(notes)
}

pub(super) fn has_notes(directory: &Path) -> MemoryResult<bool> {
    let mut files = Vec::new();
    collect_markdown(directory, &mut files)?;
    Ok(!files.is_empty())
}

/// The checkout a Claude Code project directory stands for, when its encoded
/// name (`-Users-me-src-repo`) maps back to exactly one existing path.
pub fn project_path(project_dir: &Path) -> Option<PathBuf> {
    let name = project_dir.file_name()?.to_string_lossy();
    let segments: Vec<&str> = name.trim_start_matches('-').split('-').collect();
    let mut matches = Vec::new();
    resolve_segments(Path::new("/"), &segments, &mut matches);
    (matches.len() == 1).then(|| matches.remove(0))
}

/// A hyphen can stand for either a separator or a literal hyphen. Stop once
/// two existing paths prove that the encoding cannot identify the checkout.
fn resolve_segments(base: &Path, segments: &[&str], matches: &mut Vec<PathBuf>) {
    if matches.len() > 1 {
        return;
    }
    if segments.is_empty() {
        matches.push(base.to_path_buf());
        return;
    }
    for take in 1..=segments.len() {
        let candidate = base.join(segments[..take].join("-"));
        if candidate.is_dir() {
            resolve_segments(&candidate, &segments[take..], matches);
        }
    }
}

fn collect_markdown(directory: &Path, out: &mut Vec<PathBuf>) -> MemoryResult<()> {
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let file_type = entry.file_type()?;
        if name.starts_with('.') || file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            if name != "old" {
                collect_markdown(&path, out)?;
            }
        } else if file_type.is_file() && name.ends_with(".md") && name != "MEMORY.md" {
            out.push(path);
        }
    }
    Ok(())
}

pub(super) fn modified_at(path: &Path) -> Option<u64> {
    let modified = std::fs::metadata(path).ok()?.modified().ok()?;
    let since_epoch = modified.duration_since(std::time::UNIX_EPOCH).ok()?;
    Some(since_epoch.as_secs())
}

/// Split leading `---` fenced YAML from the body.
fn split_frontmatter(text: &str) -> (Option<&str>, &str) {
    let Some(rest) = text
        .strip_prefix("---\n")
        .or_else(|| text.strip_prefix("---\r\n"))
    else {
        return (None, text);
    };
    let mut cursor = 0;
    for line in rest.split_inclusive('\n') {
        if line.trim_end() == "---" {
            return (Some(&rest[..cursor]), &rest[cursor + line.len()..]);
        }
        cursor += line.len();
    }
    (None, text)
}

fn wiki_links(body: &str) -> Vec<String> {
    static LINK: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]").expect("link pattern"));
    let mut links = Vec::new();
    for capture in LINK.captures_iter(body) {
        let link = capture[1].trim().to_string();
        if !link.is_empty() && !links.contains(&link) {
            links.push(link);
        }
    }
    links
}
