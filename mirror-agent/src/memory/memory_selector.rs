//! Admission channels for one moment: the store's lexical hits for the user
//! message and glob auto-attach for paths in context. Output is bounded by caps
//! and a token budget.

use std::collections::HashSet;

use crate::compaction::chars_for_tokens;

use super::glob_matches;
use super::render_record;
use super::should_query;
use super::MemoryQuery;
use super::MemoryRecord;
use super::MemoryResult;
use super::MemoryStore;
use super::RECALL_PREAMBLE;

/// One selection moment's inputs. Absent fields disable the channel they feed.
#[derive(Debug, Clone)]
pub struct SelectionRequest<'a> {
    /// The latest user message; `None` skips the lexical channel.
    pub message: Option<&'a str>,
    /// Repository-relative paths new to the context this moment.
    pub new_paths: &'a [String],
    /// Ids already surfaced or read in this context window; never re-admitted.
    pub surfaced: &'a HashSet<String>,
    /// Characters of memory already surfaced in this context window.
    pub surfaced_chars: usize,
    pub now: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Selected {
    pub record: MemoryRecord,
    pub channel: Channel,
    pub matched_path: Option<String>,
    pub rendered: String,
}

/// Why a record was admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Channel {
    Lexical,
    Glob,
    GlobLexical,
    Search,
}

impl Channel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lexical => "lexical",
            Self::Glob => "glob",
            Self::GlobLexical => "glob+lexical",
            Self::Search => "search",
        }
    }
}

/// Records each channel may admit per moment.
const LEXICAL_CAP: usize = 5;
const GLOB_CAP: usize = 3;
/// The recall budget, as a token estimate.
const RECALL_TOKENS: usize = 4_096;
/// Characters surfaced per context window before the lexical channel stops.
const CONTEXT_CHARS: usize = 61_440;

/// The store query for the lexical channel, or `None` when the moment has no
/// message worth searching for or the window's budget is spent.
pub fn lexical_query(scope: &MemoryQuery, request: &SelectionRequest<'_>) -> Option<MemoryQuery> {
    let message = request
        .message
        .filter(|text| request.surfaced_chars < CONTEXT_CHARS && should_query(text))?;
    Some(MemoryQuery {
        text: Some(message.to_string()),
        exclude: request.surfaced.iter().cloned().collect(),
        // At most GLOB_CAP lexical hits can overlap the already-admitted glob
        // records; fetch enough to still fill the distinct lexical quota.
        limit: LEXICAL_CAP + GLOB_CAP,
        distinct_content: true,
        ..scope.clone()
    })
}

/// The store query for the glob channel's candidates, or `None` when no path
/// entered the context this moment.
pub fn glob_query(scope: &MemoryQuery, request: &SelectionRequest<'_>) -> Option<MemoryQuery> {
    (!request.new_paths.is_empty()).then(|| MemoryQuery {
        with_globs_only: true,
        exclude: request.surfaced.iter().cloned().collect(),
        ..scope.clone()
    })
}

/// Ask `store` for both channels' candidates within `scope` and select from
/// them. One read per channel the moment enables; nothing else is loaded.
pub async fn select_from(
    store: &dyn MemoryStore,
    scope: &MemoryQuery,
    request: &SelectionRequest<'_>,
) -> MemoryResult<Vec<Selected>> {
    let glob_candidates = match glob_query(scope, request) {
        Some(query) => store.search(query).await?,
        None => Vec::new(),
    };
    let lexical = match lexical_query(scope, request) {
        Some(query) => store.search(query).await?,
        None => Vec::new(),
    };
    Ok(select(&lexical, &glob_candidates, request))
}

/// Run both channels over the store's answers and fit the union into the
/// budget. `lexical` is ranked best first and already held to the backend's
/// relevance floor; `glob_candidates` are the scoped records with path globs.
pub fn select(
    lexical: &[MemoryRecord],
    glob_candidates: &[MemoryRecord],
    request: &SelectionRequest<'_>,
) -> Vec<Selected> {
    let mut admitted: Vec<(&MemoryRecord, Channel, Option<String>)> = Vec::new();

    // Channel B first: a matching file in context is sufficient on its own.
    let mut glob_hits = Vec::new();
    for record in glob_candidates {
        if request.surfaced.contains(&record.id) {
            continue;
        }
        if let Some((path, specificity)) = first_glob_match(&record.path_globs, request.new_paths) {
            glob_hits.push((record, path, specificity));
        }
    }
    glob_hits.sort_by(|a, b| {
        b.2.cmp(&a.2)
            .then(b.0.observed_at.cmp(&a.0.observed_at))
            .then(a.0.id.cmp(&b.0.id))
    });
    let mut glob_contents = HashSet::new();
    for (record, path, _) in glob_hits {
        let content = normalized_content(record);
        if !glob_contents.contains(&content) && glob_contents.len() >= GLOB_CAP {
            continue;
        }
        // Keep an alternative copy: it may fit the recall when the
        // higher-ranked record's metadata makes that record too large.
        glob_contents.insert(content);
        admitted.push((record, Channel::Glob, Some(path)));
    }

    // Channel A: the store's ranked hits, distinct by content, capped.
    let mut lexical_contents: HashSet<String> = admitted
        .iter()
        .map(|(record, _, _)| normalized_content(record))
        .collect();
    let mut taken = 0;
    for record in lexical {
        if request.surfaced.contains(&record.id) {
            continue;
        }
        if let Some(entry) = admitted
            .iter_mut()
            .find(|(admitted, _, _)| admitted.id == record.id)
        {
            entry.1 = Channel::GlobLexical;
            continue;
        }
        let content = normalized_content(record);
        if !lexical_contents.contains(&content) && taken >= LEXICAL_CAP {
            continue;
        }
        if lexical_contents.insert(content) {
            taken += 1;
        }
        admitted.push((record, Channel::Lexical, None));
    }

    // Fit: dedupe on content, then greedy into the recall budget.
    let budget_chars = chars_for_tokens(RECALL_TOKENS);
    let mut used_chars = RECALL_PREAMBLE.len();
    let mut seen_content = HashSet::new();
    let mut selected = Vec::new();
    for (record, channel, matched_path) in admitted {
        let content = normalized_content(record);
        if seen_content.contains(&content) {
            continue;
        }
        let rendered = render_record(record, channel, matched_path.as_deref(), request.now);
        if used_chars + 2 + rendered.len() > budget_chars {
            continue;
        }
        used_chars += 2 + rendered.len();
        seen_content.insert(content);
        selected.push(Selected {
            record: record.clone(),
            channel,
            matched_path,
            rendered,
        });
    }
    selected
}

fn normalized_content(record: &MemoryRecord) -> String {
    record.content.trim().to_lowercase()
}

/// The first new path a record's globs match, with a specificity rank: more
/// literal segments and fewer wildcards rank higher.
fn first_glob_match(globs: &[String], paths: &[String]) -> Option<(String, (usize, isize))> {
    let mut best: Option<(String, (usize, isize))> = None;
    for pattern in globs {
        for path in paths {
            if glob_matches(pattern, path) {
                let specificity = (
                    pattern
                        .split('/')
                        .filter(|segment| !segment.contains(['*', '?', '[']))
                        .count(),
                    -(pattern.matches('*').count() as isize),
                );
                if best
                    .as_ref()
                    .is_none_or(|(_, current)| specificity > *current)
                {
                    best = Some((path.clone(), specificity));
                }
            }
        }
    }
    best
}

#[cfg(test)]
#[path = "memory_selector_test.rs"]
mod tests;
