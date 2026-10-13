//! `search_memory`: the agent's own way into memory when automatic selection
//! missed something. Read-only, and an ordinary tool.

use std::sync::Arc;

use serde::Deserialize;
use serde::Serialize;
use serde_json::Map;
use serde_json::Value;

use super::glob_matches;
use super::now;
use super::render_record;
use super::Channel;
use super::MemoryQuery;
use super::MemoryRecord;
use super::MemoryStatus;
use super::MemoryStore;
use super::MemoryType;
use crate::tools::object_schema;
use crate::tools::Tool;
use crate::tools::ToolError;
use crate::tools::ToolOutput;
use crate::ExecutionContext;
use crate::Workspace;

/// Search one owner's memory within the current repository's scope.
pub struct SearchMemoryTool {
    store: Arc<dyn MemoryStore>,
    workspace: Arc<dyn Workspace>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchMemoryArgs {
    #[serde(default)]
    pub query: String,
    #[serde(default)]
    pub types: Vec<String>,
    /// Paths as you would pass them to the read tool; returns memories tied to them.
    #[serde(default)]
    pub paths: Vec<String>,
    /// A record id whose complete text to return, instead of searching.
    #[serde(default)]
    pub id: String,
}

const MAX_HEADERS: usize = 10;
const BODIES: usize = 3;
/// The last line of every successful result, after all record text, so the
/// records a search showed count as shown and header-like text inside a
/// memory cannot forge it.
pub(super) const RESULT_IDS_PREFIX: &str = "Returned memory ids: ";

impl SearchMemoryTool {
    pub const NAME: &'static str = "search_memory";

    pub fn new(store: Arc<dyn MemoryStore>, workspace: Arc<dyn Workspace>) -> Self {
        Self { store, workspace }
    }

    fn scope(&self, types: Vec<MemoryType>) -> MemoryQuery {
        MemoryQuery {
            repository: self.workspace.repository().map(str::to_string),
            types,
            ..MemoryQuery::default()
        }
    }

    async fn search(&self, args: &SearchMemoryArgs) -> Result<ToolOutput, ToolError> {
        let types = args
            .types
            .iter()
            .map(|name| {
                MemoryType::parse(name).ok_or_else(|| {
                    ToolError::InvalidArguments(format!(
                        "unknown memory type {name:?}; use preference, fact, procedure, pitfall or reference"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let paths = self.repository_paths(&args.paths)?;
        let scope = self.scope(types);
        let mut hits: Vec<(MemoryRecord, Option<String>)> = Vec::new();
        if !paths.is_empty() {
            let candidates = self
                .store
                .search(MemoryQuery {
                    with_globs_only: true,
                    ..scope.clone()
                })
                .await
                .map_err(execution)?;
            for record in candidates {
                if let Some(path) = paths.iter().find(|path| {
                    record
                        .path_globs
                        .iter()
                        .any(|pattern| glob_matches(pattern, path))
                }) {
                    hits.push((record, Some(path.clone())));
                }
            }
        }
        if !args.query.trim().is_empty() {
            let matches = self
                .store
                .search(MemoryQuery {
                    text: Some(args.query.clone()),
                    limit: MAX_HEADERS,
                    ..scope
                })
                .await
                .map_err(execution)?;
            for record in matches {
                if !hits.iter().any(|(hit, _)| hit.id == record.id) {
                    hits.push((record, None));
                }
            }
        }
        hits.truncate(MAX_HEADERS);
        if hits.is_empty() {
            return Ok(ToolOutput::success(
                "No memories match. Nothing has been saved about this yet.",
            ));
        }
        Ok(ToolOutput::success(render(&hits)))
    }

    /// The complete text of the active record `id`, when it applies here.
    async fn full_text(&self, id: &str) -> Result<ToolOutput, ToolError> {
        let record = self.store.get(id).await.map_err(execution)?;
        let repository = self.workspace.repository().unwrap_or_default();
        let applies = |record: &MemoryRecord| {
            record.status == MemoryStatus::Active
                && (record.repository.is_empty() || record.repository == repository)
        };
        let Some(record) = record.filter(applies) else {
            return Ok(ToolOutput::error(format!(
                "No active memory with id {id} applies here."
            )));
        };
        let rendered = render_record(&record, Channel::Search, None, now());
        let header = rendered.lines().next().unwrap_or_default();
        Ok(ToolOutput::success(format!(
            "{header}\n{}\n{RESULT_IDS_PREFIX}{}",
            record.content, record.id
        )))
    }

    /// `paths` as the workspace names them for glob matching.
    fn repository_paths(&self, paths: &[String]) -> Result<Vec<String>, ToolError> {
        if paths.is_empty() {
            return Ok(Vec::new());
        }
        if self.workspace.repository().is_none() {
            return Err(ToolError::InvalidArguments(
                "path search requires a repository checkout".to_string(),
            ));
        }
        paths
            .iter()
            .map(|raw| {
                self.workspace.relative(raw).ok_or_else(|| {
                    ToolError::InvalidArguments(format!(
                        "search path {raw:?} does not resolve within the repository"
                    ))
                })
            })
            .collect()
    }
}

impl Tool for SearchMemoryTool {
    type Args = SearchMemoryArgs;

    fn name(&self) -> &str {
        Self::NAME
    }

    fn description(&self) -> &str {
        "Search the user's private memory (preferences, facts, procedures, pitfalls, references) when the memories shown automatically do not cover what you need. Returns matching records; a record's complete text comes back when you pass its id."
    }

    fn parameters(&self) -> Map<String, Value> {
        object_schema(
            Map::from_iter([
                (
                    "query".to_string(),
                    serde_json::json!({"type": "string", "maxLength": 2048, "description": "Words a matching memory would contain: tool names, error strings, file names, topics."}),
                ),
                (
                    "types".to_string(),
                    serde_json::json!({"type": "array", "maxItems": 5, "items": {"type": "string", "enum": ["preference", "fact", "procedure", "pitfall", "reference"]}, "description": "Only return memories of these kinds"}),
                ),
                (
                    "paths".to_string(),
                    serde_json::json!({"type": "array", "maxItems": 8, "items": {"type": "string", "maxLength": 512}, "description": "Paths as you would pass them to the read tool; also returns memories tied to them."}),
                ),
                (
                    "id".to_string(),
                    serde_json::json!({"type": "string", "maxLength": 64, "description": "The id of one memory shown earlier; returns its complete text instead of searching."}),
                ),
            ]),
            &[],
        )
    }

    async fn run(
        &self,
        args: SearchMemoryArgs,
        _context: &ExecutionContext,
    ) -> Result<ToolOutput, ToolError> {
        if args.query.len() > 2_048
            || args.paths.len() > 8
            || args.paths.iter().any(|path| path.len() > 512)
            || args.types.len() > 5
        {
            return Err(ToolError::InvalidArguments(
                "query is at most 2048 bytes, with at most 8 paths of 512 bytes and 5 types"
                    .to_string(),
            ));
        }
        if !args.id.trim().is_empty() {
            return self.full_text(args.id.trim()).await;
        }
        if args.query.trim().is_empty() && args.paths.is_empty() {
            return Err(ToolError::InvalidArguments(
                "give a query or path to search, or the id of a memory to read".to_string(),
            ));
        }
        self.search(&args).await
    }
}

/// Every hit's header and the first few bodies, then the ids line.
fn render(hits: &[(MemoryRecord, Option<String>)]) -> String {
    let today = now();
    let mut out = format!(
        "{} matching memor{}:\n",
        hits.len(),
        if hits.len() == 1 { "y" } else { "ies" }
    );
    for (index, (record, path)) in hits.iter().enumerate() {
        let rendered = render_record(record, Channel::Search, path.as_deref(), today);
        let shown = if index < BODIES {
            rendered.as_str()
        } else {
            rendered.lines().next().unwrap_or_default()
        };
        out.push_str(&format!("\n{shown}\n"));
    }
    out.push_str(RESULT_IDS_PREFIX);
    out.push_str(
        &hits
            .iter()
            .map(|(record, _)| record.id.as_str())
            .collect::<Vec<_>>()
            .join(","),
    );
    out
}

/// The ids a successful `search_memory` result reported on its last line.
pub(super) fn search_result_ids(content: &str) -> impl Iterator<Item = String> + '_ {
    content
        .lines()
        .last()
        .and_then(|line| line.strip_prefix(RESULT_IDS_PREFIX))
        .into_iter()
        .flat_map(|ids| ids.split(','))
        .filter_map(|id| uuid::Uuid::parse_str(id).ok().map(|uuid| uuid.to_string()))
}

fn execution(error: impl std::fmt::Display) -> ToolError {
    ToolError::Execution(error.to_string())
}

#[cfg(test)]
#[path = "search_memory_tool_test.rs"]
mod tests;
