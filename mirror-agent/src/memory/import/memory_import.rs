//! One import run: read notes from other harnesses' files, turn each into a
//! record under the user's authority, and commit idempotently. Each source is
//! a snapshot: what it holds now is what stays active from it.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::Path;
use std::path::PathBuf;

use super::claude_code_notes;
use super::codex_notes;
use crate::memory::create_record;
use crate::memory::invalid;
use crate::memory::now;
use crate::memory::replaceable;
use crate::memory::MemoryActor;
use crate::memory::MemoryChange;
use crate::memory::MemoryCommit;
use crate::memory::MemoryDraft;
use crate::memory::MemoryError;
use crate::memory::MemoryEventOp;
use crate::memory::MemoryQuery;
use crate::memory::MemoryRecord;
use crate::memory::MemoryResult;
use crate::memory::MemoryType;
use crate::memory::SqliteMemoryDatabase;
use crate::memory::MAX_KEYWORDS;
use crate::memory::MAX_SITUATION_BYTES;
use crate::RepositoryIdentity;

/// Reads notes from every source and commits them to one store.
pub struct MemoryImport<'a> {
    database: &'a mut SqliteMemoryDatabase,
}

/// Where notes come from. Each variant is one harness's on-disk layout.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ImportSource {
    /// A Claude Code auto-memory directory: `MEMORY.md` plus one Markdown
    /// file per memory. `repository` is the project's checkout: its notes are
    /// scoped to that repository, or apply everywhere when it is not one.
    ClaudeCode {
        directory: PathBuf,
        repository: Option<PathBuf>,
    },
    /// A Codex `memories` directory: `memory_summary.md` and `MEMORY.md`.
    Codex {
        directory: PathBuf,
        repository: Option<PathBuf>,
    },
}

/// One note as a reader produced it, before validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportedNote {
    pub key: String,
    pub kind: MemoryType,
    /// When the note applies; a reader falls back to naming its origin.
    pub situation: String,
    pub content: String,
    pub keywords: Vec<String>,
    /// The file's modification time.
    pub observed_at: Option<u64>,
    /// The file the text came from, relative to the source.
    pub file: String,
}

/// What one run did, and every note the store refused.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImportReport {
    pub saved: usize,
    pub unchanged: usize,
    /// Records from earlier runs that their sources no longer hold.
    pub retired: usize,
    /// (key, reason) for every note that was not imported.
    pub rejected: Vec<(String, String)>,
}

enum Outcome {
    Saved,
    Unchanged,
}

impl<'a> MemoryImport<'a> {
    pub fn new(database: &'a mut SqliteMemoryDatabase) -> Self {
        Self { database }
    }

    /// Import every source.
    pub fn run(&mut self, sources: &[ImportSource]) -> MemoryResult<ImportReport> {
        let mut report = ImportReport::default();
        for source in sources {
            let notes = match source.read() {
                Ok(notes) => notes,
                Err(error) => {
                    report
                        .rejected
                        .push((source.to_string(), format!("could not read: {error}")));
                    continue;
                }
            };
            let identity = source.repository().and_then(RepositoryIdentity::discover);
            let scope_repository = identity
                .as_ref()
                .map_or(String::new(), |identity| identity.repository().to_string());
            let prefix = source.prefix();
            let mut seen_keys: BTreeMap<String, usize> = BTreeMap::new();
            let mut keys = BTreeSet::new();
            for note in notes {
                let key = unique_key(&note_key(source, &note), &mut seen_keys);
                keys.insert(key.clone());
                match self.commit(note, key.clone(), &scope_repository, &prefix) {
                    Ok(Outcome::Saved) => report.saved += 1,
                    Ok(Outcome::Unchanged) => report.unchanged += 1,
                    Err(error) => report.rejected.push((key, reason(error))),
                }
            }
            report.retired += self.reconcile(&prefix, &keys, &scope_repository)?;
        }
        Ok(report)
    }

    /// Store one note, or leave it when it is already there in any state: an
    /// import never undoes a retirement. A key another writer holds is
    /// refused.
    fn commit(
        &mut self,
        note: ImportedNote,
        key: String,
        scope_repository: &str,
        prefix: &str,
    ) -> MemoryResult<Outcome> {
        let observed = note.observed_at.unwrap_or_else(now);
        let mut record = create_record(
            note.into_draft(key, scope_repository, prefix),
            MemoryActor::Import,
            observed,
        )?;
        record.id = import_id(&record);
        if self.database.get(&record.id)?.is_some() {
            return Ok(Outcome::Unchanged);
        }
        let mut changes = vec![MemoryChange::Insert(record.clone())];
        let active = self.database.search(&MemoryQuery {
            repository: Some(scope_repository.to_string()),
            ..MemoryQuery::default()
        })?;
        for existing in replaceable(&active, &record.key, scope_repository) {
            if existing.id == record.id {
                continue;
            }
            if existing.actor != MemoryActor::Import || !existing.source.starts_with(prefix) {
                return Err(invalid("another memory already has this key"));
            }
            // The same note, edited: the old text goes, the key stays.
            changes.push(MemoryChange::retire(
                existing.clone(),
                MemoryEventOp::Retire,
            ));
        }
        self.database.commit(&MemoryCommit {
            actor: MemoryActor::Import,
            session_id: None,
            changes,
        })?;
        Ok(Outcome::Saved)
    }

    /// Retire what this source produced earlier and no longer holds: every
    /// active record the importer wrote from `prefix` in this scope whose key
    /// `keys` does not name.
    fn reconcile(
        &mut self,
        prefix: &str,
        keys: &BTreeSet<String>,
        scope_repository: &str,
    ) -> MemoryResult<usize> {
        let active = self.database.search(&MemoryQuery {
            repository: Some(scope_repository.to_string()),
            ..MemoryQuery::default()
        })?;
        let gone: Vec<MemoryChange> = active
            .into_iter()
            .filter(|record| {
                record.actor == MemoryActor::Import
                    && record.repository == scope_repository
                    && record.source.starts_with(prefix)
                    && !keys.contains(&record.key)
            })
            .map(|record| MemoryChange::retire(record, MemoryEventOp::Retire))
            .collect();
        let retired = gone.len();
        if !gone.is_empty() {
            self.database.commit(&MemoryCommit {
                actor: MemoryActor::Import,
                session_id: None,
                changes: gone,
            })?;
        }
        Ok(retired)
    }
}

impl ImportSource {
    /// Parse `kind=path` or a bare path whose layout is detected.
    pub fn parse(spec: &str, repository: Option<&Path>) -> MemoryResult<Self> {
        let (kind, path) = match spec.split_once('=') {
            Some((kind, path)) if !path.is_empty() && !kind.contains('/') => (Some(kind), path),
            _ => (None, spec),
        };
        let path = PathBuf::from(expand_home(path));
        let repository = repository.map(Path::to_path_buf);
        Ok(match kind {
            Some("claude-code" | "claude") => Self::ClaudeCode {
                directory: path,
                repository,
            },
            Some("codex") => Self::Codex {
                directory: path,
                repository,
            },
            Some(other) => {
                return Err(invalid(format!(
                    "unknown memory source kind {other:?}; use claude-code or codex"
                )))
            }
            None => Self::detect(&path, repository)?,
        })
    }

    /// Recognize a directory by its layout.
    pub fn detect(path: &Path, repository: Option<PathBuf>) -> MemoryResult<Self> {
        if !path.is_dir() {
            return Err(invalid(format!("{} is not a directory", path.display())));
        }
        let directory = path.to_path_buf();
        if path.join("memory_summary.md").is_file() {
            return Ok(Self::Codex {
                directory,
                repository,
            });
        }
        if claude_code_notes::has_notes(path)? {
            return Ok(Self::ClaudeCode {
                directory,
                repository,
            });
        }
        if path.join("MEMORY.md").is_file() {
            return Ok(Self::Codex {
                directory,
                repository,
            });
        }
        Err(invalid(format!(
            "{} does not look like a Claude Code or Codex memory location",
            path.display()
        )))
    }

    /// The default locations each harness writes to on this machine, when
    /// present. Notes from a Claude Code project outside any known checkout
    /// apply in every repository.
    pub fn discover(repository: Option<&Path>) -> Vec<Self> {
        let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
            return Vec::new();
        };
        let mut sources = Vec::new();
        if let Ok(projects) = std::fs::read_dir(home.join(".claude").join("projects")) {
            let mut directories: Vec<PathBuf> = projects
                .flatten()
                .map(|entry| entry.path())
                .filter(|project| project.join("memory").is_dir())
                .collect();
            directories.sort();
            for project in directories {
                let repository = repository.map(Path::to_path_buf).or_else(|| {
                    claude_code_notes::project_path(&project).filter(|candidate| candidate.is_dir())
                });
                sources.push(Self::ClaudeCode {
                    directory: project.join("memory"),
                    repository,
                });
            }
        }
        let codex = home.join(".codex").join("memories");
        if codex.join("memory_summary.md").is_file() || codex.join("MEMORY.md").is_file() {
            sources.push(Self::Codex {
                directory: codex,
                repository: repository.map(Path::to_path_buf),
            });
        }
        sources
    }

    pub fn read(&self) -> MemoryResult<Vec<ImportedNote>> {
        match self {
            Self::ClaudeCode { directory, .. } => claude_code_notes::claude_code_notes(directory),
            Self::Codex { directory, .. } => codex_notes::codex_notes(directory),
        }
    }

    fn harness(&self) -> &'static str {
        match self {
            Self::ClaudeCode { .. } => "Claude Code",
            Self::Codex { .. } => "Codex",
        }
    }

    fn directory(&self) -> &Path {
        match self {
            Self::ClaudeCode { directory, .. } | Self::Codex { directory, .. } => directory,
        }
    }

    fn repository(&self) -> Option<&Path> {
        match self {
            Self::ClaudeCode { repository, .. } | Self::Codex { repository, .. } => {
                repository.as_deref()
            }
        }
    }

    /// How every record from this source starts its `source`: the harness
    /// and the directory, so a later run can tell this source's records apart.
    fn prefix(&self) -> String {
        let directory = std::fs::canonicalize(self.directory())
            .unwrap_or_else(|_| self.directory().to_path_buf());
        format!("{} [{}] ", self.harness(), directory.display())
    }
}

impl std::fmt::Display for ImportSource {
    /// The harness and where its notes were written: the project for Claude
    /// Code, the notes directory for Codex.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let location = match self {
            Self::ClaudeCode {
                directory,
                repository,
            } => repository.as_deref().unwrap_or(directory),
            Self::Codex { directory, .. } => directory,
        };
        write!(formatter, "{} {}", self.harness(), home_relative(location))
    }
}

impl ImportedNote {
    fn into_draft(self, key: String, repository: &str, prefix: &str) -> MemoryDraft {
        let situation = if self.situation.trim().is_empty() {
            "from the user's imported notes".to_string()
        } else {
            truncate_bytes(&clean(&self.situation), MAX_SITUATION_BYTES)
        };
        let mut keywords: Vec<String> = self
            .keywords
            .into_iter()
            .map(|keyword| truncate_bytes(keyword.trim(), 64))
            .filter(|keyword| !keyword.is_empty())
            .collect();
        keywords.dedup();
        keywords.truncate(MAX_KEYWORDS);
        MemoryDraft {
            r#type: self.kind,
            key,
            repository: repository.to_string(),
            path_globs: Vec::new(),
            situation,
            keywords,
            content: clean(&self.content),
            source: truncate_bytes(&format!("{prefix}{}", self.file), 1_024),
        }
    }
}

/// A note's handle: its slug plus a short hash of what identifies it within
/// the source. A Claude file is one note, so its path identifies it across
/// edits; a Codex bullet has no file of its own, so its text does.
fn note_key(source: &ImportSource, note: &ImportedNote) -> String {
    let identity = match source {
        ImportSource::ClaudeCode { .. } => note.file.clone(),
        ImportSource::Codex { .. } => format!("{}\0{}", note.file, clean(&note.content)),
    };
    let digest = ring::digest::digest(&ring::digest::SHA256, identity.as_bytes());
    let suffix: String = digest
        .as_ref()
        .iter()
        .take(6)
        .map(|byte| format!("{byte:02x}"))
        .collect();
    let base = note.key.trim_end_matches('-');
    let end = base.len().min(64 - 1 - suffix.len());
    format!("{}-{suffix}", base[..end].trim_end_matches('-'))
}

fn unique_key(key: &str, seen: &mut BTreeMap<String, usize>) -> String {
    let count = seen.entry(key.to_string()).or_default();
    *count += 1;
    if *count == 1 {
        key.to_string()
    } else {
        format!("{key}-{count}")
    }
}

/// The same note in the same scope always gets the same id, so a rerun finds
/// what it produced before, whatever that record's status is now.
fn import_id(record: &MemoryRecord) -> String {
    let mut context = ring::digest::Context::new(&ring::digest::SHA256);
    for part in [
        record.source.as_str(),
        &record.key,
        &record.repository,
        &record.situation,
        &record.keywords.join("\n"),
        &record.content,
        record.r#type.as_str(),
    ] {
        context.update(part.as_bytes());
        context.update(b"\0");
    }
    let digest = context.finish();
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&digest.as_ref()[..16]);
    uuid::Builder::from_random_bytes(bytes)
        .into_uuid()
        .to_string()
}

/// Why a note was skipped, without the error's `memory:` prefix.
fn reason(error: MemoryError) -> String {
    match error {
        MemoryError::Invalid(message) => message,
        error => error.to_string(),
    }
}

/// `path` with the home directory written as `~`.
fn home_relative(path: &Path) -> String {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    match home
        .as_deref()
        .and_then(|home| path.strip_prefix(home).ok())
    {
        Some(rest) if rest.as_os_str().is_empty() => "~".to_string(),
        Some(rest) => format!("~/{}", rest.display()),
        None => path.display().to_string(),
    }
}

/// Keep newlines and tabs, drop other control characters and carriage returns.
fn clean(text: &str) -> String {
    text.replace("\r\n", "\n")
        .chars()
        .filter(|character| !character.is_control() || *character == '\n' || *character == '\t')
        .collect::<String>()
        .trim()
        .to_string()
}

fn truncate_bytes(text: &str, max_bytes: usize) -> String {
    let mut end = 0;
    for (offset, character) in text.char_indices() {
        if offset + character.len_utf8() > max_bytes {
            break;
        }
        end = offset + character.len_utf8();
    }
    text[..end].to_string()
}

fn expand_home(path: &str) -> String {
    match (path.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => PathBuf::from(home)
            .join(rest)
            .to_string_lossy()
            .into_owned(),
        _ => path.to_string(),
    }
}

#[cfg(test)]
#[path = "memory_import_test.rs"]
mod tests;
