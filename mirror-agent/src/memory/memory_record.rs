//! One validated memory record; bounds and phrasing are enforced here, never by the model.

use std::sync::LazyLock;

use regex::Regex;

use super::invalid;
use super::MemoryActor;
use super::MemoryError;
use super::MemoryResult;

/// What kind of knowledge a record holds; decides admission and injection policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum MemoryType {
    #[default]
    Preference,
    Fact,
    Procedure,
    Pitfall,
    Reference,
}

/// Whether selection may see the record.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MemoryStatus {
    #[default]
    Active,
    Retired,
}

/// One memory as stored: a row, not a blob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryRecord {
    /// Stable UUID.
    pub id: String,
    /// Kebab-case handle a writer uses to replace or forget the record.
    pub key: String,
    /// Starts at 1 and increments on every change to this record.
    pub version: u64,
    pub r#type: MemoryType,
    pub status: MemoryStatus,
    /// Who wrote the current version; protects the person's words from
    /// background passes.
    pub actor: MemoryActor,
    /// The repository this applies to, as `RepositoryIdentity` names it;
    /// empty means any repository.
    pub repository: String,
    /// Toplevel-relative file patterns naming where this memory applies.
    pub path_globs: Vec<String>,
    /// One sentence: when a future session should use this record; indexed.
    pub situation: String,
    /// Terms that help lexical retrieval find the record.
    pub keywords: Vec<String>,
    /// The memory itself, as shown to the model.
    pub content: String,
    /// Where an imported record came from: the harness, its directory and
    /// the file. Empty for what was written here.
    pub source: String,
    /// When the content was observed; fixed for the life of the record.
    pub observed_at: u64,
    /// When the current version was written; set by the store.
    pub updated_at: u64,
}

/// Everything a writer may supply. The runtime owns identity, version, status and time.
#[derive(Debug, Clone, Default)]
pub struct MemoryDraft {
    pub r#type: MemoryType,
    pub key: String,
    pub repository: String,
    pub path_globs: Vec<String>,
    pub situation: String,
    pub keywords: Vec<String>,
    pub content: String,
    pub source: String,
}

// High-precision credential formats, not a general password or PII detector.
const CREDENTIAL_PATTERN: &str = concat!(
    r"(?s:-----BEGIN[A-Z ]*PRIVATE KEY-----.*?(?:-----END[A-Z ]*PRIVATE KEY-----|$))",
    r"|\bAKIA[0-9A-Z]{16}\b",
    r"|\bgh[pousr]_[A-Za-z0-9]{36,}\b",
    r"|\bgithub_pat_[A-Za-z0-9_]{40,}\b",
    r"|\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
    r"|\bAIza[0-9A-Za-z_-]{35}\b",
    r"|\bsk-[A-Za-z0-9]{32,}\b",
    r"|\bsk-(?:proj|svcacct|admin)-[A-Za-z0-9_-]{20,}\b",
);

static CREDENTIAL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(CREDENTIAL_PATTERN).expect("credential pattern compiles"));

/// Stored text is kept whole; what a recall shows is bounded separately at
/// render time.
const MAX_CONTENT_BYTES: usize = 65_536;
const MAX_PITFALL_BYTES: usize = 1_024;
pub const MAX_SITUATION_BYTES: usize = 512;
const MAX_KEY_BYTES: usize = 64;
pub const MAX_KEYWORDS: usize = 32;
const MAX_GLOBS: usize = 16;
const MAX_SOURCE_BYTES: usize = 1_024;

impl MemoryType {
    pub const ALL: [MemoryType; 5] = [
        Self::Preference,
        Self::Fact,
        Self::Procedure,
        Self::Pitfall,
        Self::Reference,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Preference => "preference",
            Self::Fact => "fact",
            Self::Procedure => "procedure",
            Self::Pitfall => "pitfall",
            Self::Reference => "reference",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == name)
    }
}

impl MemoryStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Retired => "retired",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "active" => Some(Self::Active),
            "retired" => Some(Self::Retired),
            _ => None,
        }
    }
}

/// Build a validated, active first-version record written by `actor` and
/// observed at `observed_at`.
pub fn create(
    draft: MemoryDraft,
    actor: MemoryActor,
    observed_at: u64,
) -> MemoryResult<MemoryRecord> {
    let record = MemoryRecord {
        id: uuid::Uuid::new_v4().to_string(),
        key: draft.key,
        version: 1,
        r#type: draft.r#type,
        status: MemoryStatus::Active,
        actor,
        repository: draft.repository,
        path_globs: draft.path_globs,
        situation: draft.situation,
        keywords: draft.keywords,
        content: draft.content,
        source: draft.source,
        observed_at,
        updated_at: observed_at,
    };
    validate(&record)?;
    Ok(record)
}

/// Check every bound and phrasing rule a stored record must satisfy.
pub fn validate(record: &MemoryRecord) -> MemoryResult<()> {
    if uuid::Uuid::parse_str(&record.id).is_err() {
        return Err(invalid("memory id is not a UUID"));
    }
    if record.version == 0 {
        return Err(invalid("memory version starts at 1"));
    }
    if !is_kebab(&record.key) || record.key.len() > MAX_KEY_BYTES {
        return Err(invalid(
            "memory key must be a kebab-case slug of at most 64 bytes",
        ));
    }
    if record.situation.len() > MAX_SITUATION_BYTES || has_control_chars(&record.situation) {
        return Err(invalid(
            "memory situation is at most 512 bytes of plain text",
        ));
    }
    if record.situation.trim().is_empty() {
        return Err(invalid(
            "a memory needs a situation: when a future session should use it",
        ));
    }
    if record.path_globs.len() > MAX_GLOBS {
        return Err(invalid("memory record has more than 16 path globs"));
    }
    if !record.path_globs.is_empty() && record.repository.is_empty() {
        return Err(invalid("path globs require a repository"));
    }
    for pattern in &record.path_globs {
        validate_glob(pattern)?;
    }
    if record.keywords.len() > MAX_KEYWORDS
        || record
            .keywords
            .iter()
            .any(|keyword| keyword.trim().is_empty() || keyword.len() > 64)
    {
        return Err(invalid(
            "memory keywords: at most 32 entries of at most 64 bytes",
        ));
    }
    let content_limit = if record.r#type == MemoryType::Pitfall {
        MAX_PITFALL_BYTES
    } else {
        MAX_CONTENT_BYTES
    };
    if record.content.trim().is_empty() || record.content.len() > content_limit {
        return Err(invalid(format!(
            "memory content must be non-empty and at most {content_limit} bytes"
        )));
    }
    if has_control_chars(&record.content) {
        return Err(invalid("memory content contains control characters"));
    }
    if record.observed_at == 0 {
        return Err(invalid("memory observed_at must be set"));
    }
    if record.source.len() > MAX_SOURCE_BYTES || has_control_chars(&record.source) {
        return Err(invalid(format!(
            "memory source is at most {MAX_SOURCE_BYTES} bytes of plain text"
        )));
    }
    let mut screened = vec![record.content.as_str(), record.situation.as_str()];
    screened.extend(record.keywords.iter().map(String::as_str));
    reject_credentials(&screened)
}

/// Whether a background writer may change `record`: active and not written
/// by the person.
pub fn is_unprotected(record: &MemoryRecord) -> bool {
    record.status == MemoryStatus::Active && !record.actor.is_person()
}

/// The active records among `records` with `key` in exactly `repository`. A
/// global record (empty repository) and a repository-scoped record never
/// match each other; callers decide which of them they may replace with
/// `is_unprotected`.
pub fn replaceable<'a>(
    records: impl IntoIterator<Item = &'a MemoryRecord>,
    key: &str,
    repository: &str,
) -> Vec<&'a MemoryRecord> {
    records
        .into_iter()
        .filter(|record| {
            record.status == MemoryStatus::Active
                && record.key == key
                && record.repository == repository
        })
        .collect()
}

/// Compile one toplevel-relative glob. Root-level files and `**` prefixes are valid.
pub fn validate_glob(pattern: &str) -> MemoryResult<globset::Glob> {
    if pattern.len() > 256
        || pattern.is_empty()
        || pattern.starts_with('/')
        || pattern.contains("//")
        || pattern.split('/').any(|segment| segment == "..")
    {
        return Err(invalid(format!(
            "path glob {pattern:?} must be relative to the repository root"
        )));
    }
    globset::GlobBuilder::new(pattern)
        .literal_separator(true)
        .build()
        .map_err(|error| invalid(format!("path glob {pattern:?}: {error}")))
}

/// Whether toplevel-relative `path` matches `pattern`; an invalid pattern matches nothing.
pub fn glob_matches(pattern: &str, path: &str) -> bool {
    validate_glob(pattern).is_ok_and(|glob| glob.compile_matcher().is_match(path))
}

pub(crate) fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn is_kebab(key: &str) -> bool {
    !key.is_empty()
        && !key.starts_with('-')
        && !key.ends_with('-')
        && !key.contains("--")
        && key
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
}

fn has_control_chars(text: &str) -> bool {
    text.chars()
        .any(|character| character.is_control() && character != '\n' && character != '\t')
}

/// Refuse any text that looks like a secret.
pub fn reject_credentials(texts: &[&str]) -> MemoryResult<()> {
    if texts.iter().any(|text| CREDENTIAL.is_match(text)) {
        return Err(MemoryError::Credential);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    use super::*;

    pub(crate) fn draft() -> MemoryDraft {
        MemoryDraft {
            r#type: MemoryType::Pitfall,
            key: "pkill-self-match".into(),
            repository: "github.com/example/project".into(),
            path_globs: vec!["scripts/**/*.sh".into()],
            situation: "killing a background process from the bash tool".into(),
            keywords: vec!["pkill".into(), "exit 144".into()],
            content: "pkill -f matches the tool shell itself; bracketing the pattern avoids it."
                .into(),
            source: String::new(),
        }
    }

    #[test]
    fn builds_a_first_version_active_record() {
        let record = create(draft(), MemoryActor::User, 1_000).expect("valid record");
        assert_eq!(record.version, 1);
        assert_eq!(record.status, MemoryStatus::Active);
        assert_eq!(record.actor, MemoryActor::User);
        assert_eq!(record.observed_at, 1_000);
    }

    #[test]
    fn a_memory_needs_a_situation() {
        let mut unconditional = draft();
        unconditional.situation.clear();
        assert_matches!(
            create(unconditional, MemoryActor::User, 1),
            Err(MemoryError::Invalid(_))
        );
    }

    #[test]
    fn accepts_natural_wording_for_every_memory_type() {
        let mut imperative = draft();
        imperative.content = "Always bracket the pkill pattern.".into();
        assert_matches!(create(imperative.clone(), MemoryActor::User, 1), Ok(_));
        imperative.r#type = MemoryType::Procedure;
        assert_matches!(create(imperative, MemoryActor::User, 1), Ok(_));
    }

    #[test]
    fn accepts_common_repository_globs() {
        for (pattern, matches, misses) in [
            ("**/*.rs", "src/main.rs", "src/main.py"),
            ("src/**/*.rs", "src/bin/main.rs", "tests/main.rs"),
            ("scripts/*.sh", "scripts/run.sh", "scripts/nested/run.sh"),
            ("Cargo.toml", "Cargo.toml", "crates/app/Cargo.toml"),
        ] {
            let mut scoped = draft();
            scoped.path_globs = vec![pattern.into()];
            assert_matches!(create(scoped, MemoryActor::User, 1), Ok(_));
            assert!(
                glob_matches(pattern, matches),
                "{pattern} must match {matches}"
            );
            assert!(
                !glob_matches(pattern, misses),
                "{pattern} must not match {misses}"
            );
        }
    }

    #[test]
    fn rejects_escaping_globs_credentials_and_long_pitfalls() {
        let mut escaping = draft();
        escaping.path_globs = vec!["../outside/*.rs".into()];
        assert_matches!(
            create(escaping, MemoryActor::User, 1),
            Err(MemoryError::Invalid(_))
        );
        let mut leaked = draft();
        // Assembled at runtime so no token-shaped literal sits in the source.
        leaked.content = format!("token {}_{}", "ghp", "a".repeat(36));
        assert_matches!(
            create(leaked, MemoryActor::User, 1),
            Err(MemoryError::Credential)
        );
        let mut long = draft();
        long.content = "x".repeat(MAX_PITFALL_BYTES + 1);
        assert_matches!(create(long, MemoryActor::User, 1), Err(_));
    }

    #[test]
    fn background_writers_may_only_change_what_the_workload_wrote() {
        let person = create(draft(), MemoryActor::Import, 1).expect("record");
        let workload = create(draft(), MemoryActor::Extractor, 1).expect("record");
        let mut retired = create(draft(), MemoryActor::Consolidator, 1).expect("record");
        retired.status = MemoryStatus::Retired;
        assert!(!is_unprotected(&person));
        assert!(is_unprotected(&workload));
        assert!(!is_unprotected(&retired));
    }

    #[test]
    fn replaceable_matches_key_and_exact_repository_only() {
        let scoped = create(draft(), MemoryActor::User, 1).expect("record");
        let mut global_draft = draft();
        global_draft.repository.clear();
        global_draft.path_globs.clear();
        let global = create(global_draft, MemoryActor::User, 1).expect("record");
        let records = [scoped.clone(), global.clone()];
        assert_eq!(
            replaceable(&records, "pkill-self-match", "github.com/example/project"),
            vec![&scoped]
        );
        assert_eq!(replaceable(&records, "pkill-self-match", ""), vec![&global]);
        assert!(replaceable(&records, "other-key", "").is_empty());
    }
}
