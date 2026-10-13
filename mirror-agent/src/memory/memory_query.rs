//! What one read asks the store for: scope predicates, and either a lexical
//! search or a scan of the scoped records.

use super::MemoryType;

/// Scope predicates every result must pass, plus what to look for within them.
///
/// With `text`, the store runs its lexical search and returns the matches best
/// first, already held to its relevance floor. Without `text`, it returns the
/// scoped active records newest first.
#[derive(Debug, Clone, Default)]
pub struct MemoryQuery {
    /// `None` outside any repository: only repository-agnostic records qualify.
    pub repository: Option<String>,
    /// Empty admits every type.
    pub types: Vec<MemoryType>,
    /// Free text to search for; `None` scans the scope instead.
    pub text: Option<String>,
    /// Ids never returned, applied before `limit`.
    pub exclude: Vec<String>,
    /// The most results, or distinct contents when `distinct_content` is set;
    /// zero means no limit. A distinct search returns at most two records per
    /// content: the best match and its shortest alternative.
    pub limit: usize,
    /// Count distinct normalized content toward `limit`, so duplicate records
    /// cannot hide a later, different match.
    pub distinct_content: bool,
    /// Only records with path globs, the candidates for attaching by file.
    pub with_globs_only: bool,
}
