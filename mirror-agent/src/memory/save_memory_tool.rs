//! `save_memory`: the user's explicit remember, forget, pin and unpin, applied
//! under user authority in the same turn.

use std::path::PathBuf;
use std::sync::Arc;

use serde::Deserialize;
use serde::Serialize;
use serde_json::Map;
use serde_json::Value;

use super::create_record;
use super::invalid;
use super::now;
use super::pinned_memories;
use super::slug_key;
use super::MemoryActor;
use super::MemoryChange;
use super::MemoryCommit;
use super::MemoryDraft;
use super::MemoryEventOp;
use super::MemoryQuery;
use super::MemoryRecord;
use super::MemoryResult;
use super::MemoryStatus;
use super::MemoryStore;
use super::MemoryType;
use crate::tools::object_schema;
use crate::tools::Tool;
use crate::tools::ToolError;
use crate::tools::ToolOutput;
use crate::ExecutionContext;
use crate::Workspace;

/// Applies the user's remember, forget, pin and unpin, stamped with the
/// conversation. Records go to the store; pins go to `PINNED_MEMORIES.md`.
pub struct SaveMemoryTool {
    store: Arc<dyn MemoryStore>,
    workspace: Arc<dyn Workspace>,
    conversation_id: String,
    /// The owner's `PINNED_MEMORIES.md`, which pin and unpin edit.
    pinned_memories: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SaveAction {
    Remember,
    Forget,
    Pin,
    Unpin,
}

/// Where a remembered memory applies.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SaveScope {
    /// Every repository, and sessions outside any repository.
    #[default]
    Global,
    /// Only the repository the session works in.
    Repository,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SaveMemoryArgs {
    pub action: SaveAction,
    /// The memory type; required for `remember`.
    #[serde(default)]
    pub kind: String,
    /// The memory text: what to remember, or the bullet to pin or unpin.
    #[serde(default)]
    pub content: String,
    /// Kebab-case handle. Optional for `remember`; names the target otherwise.
    #[serde(default)]
    pub key: String,
    /// The id of an existing memory: the target of forget, or the record to
    /// copy into the pinned memories.
    #[serde(default)]
    pub id: String,
    /// When the memory applies; defaults to "general".
    #[serde(default)]
    pub situation: String,
    /// Key of an existing memory this one replaces.
    #[serde(default)]
    pub replaces: String,
    /// Where the memory applies; path globs imply the current repository.
    #[serde(default)]
    pub scope: SaveScope,
    #[serde(default)]
    pub path_globs: Vec<String>,
    #[serde(default)]
    pub keywords: Vec<String>,
}

impl SaveMemoryTool {
    pub const NAME: &'static str = "save_memory";

    pub fn new(
        store: Arc<dyn MemoryStore>,
        workspace: Arc<dyn Workspace>,
        conversation_id: impl Into<String>,
        pinned_memories: PathBuf,
    ) -> Self {
        Self {
            store,
            workspace,
            conversation_id: conversation_id.into(),
            pinned_memories,
        }
    }

    fn scope(&self) -> MemoryQuery {
        MemoryQuery {
            repository: self.workspace.repository().map(str::to_string),
            ..MemoryQuery::default()
        }
    }

    fn commit(&self, changes: Vec<MemoryChange>) -> MemoryCommit {
        MemoryCommit {
            actor: MemoryActor::User,
            session_id: Some(self.conversation_id.clone()),
            changes,
        }
    }

    /// Apply one call. A failure changes nothing and tells the model why.
    async fn save(&self, args: SaveMemoryArgs) -> ToolOutput {
        let applied = match args.action {
            SaveAction::Remember => self.remember(&args).await,
            SaveAction::Forget => self.forget(&args).await,
            SaveAction::Pin => self.pin(&args).await,
            SaveAction::Unpin => self.unpin(&args).await,
        };
        match applied {
            Ok(message) => ToolOutput::success(message),
            Err(error) => ToolOutput::error(format!("Nothing saved: {error}")),
        }
    }

    async fn remember(&self, args: &SaveMemoryArgs) -> MemoryResult<String> {
        let kind = MemoryType::parse(&args.kind)
            .ok_or_else(|| invalid(format!("unknown memory type {:?}", args.kind)))?;
        let key = if args.key.trim().is_empty() {
            slug_key(&args.content)
        } else {
            args.key.trim().to_string()
        };
        let repository = if args.scope == SaveScope::Repository || !args.path_globs.is_empty() {
            self.workspace
                .repository()
                .ok_or_else(|| {
                    invalid("repository scope or path globs require a repository workspace")
                })?
                .to_string()
        } else {
            String::new()
        };
        let situation = args.situation.trim();
        let record = create_record(
            MemoryDraft {
                r#type: kind,
                key,
                repository,
                path_globs: args.path_globs.clone(),
                situation: if situation.is_empty() {
                    "general".into()
                } else {
                    situation.into()
                },
                keywords: args.keywords.clone(),
                content: args.content.trim().to_string(),
                source: String::new(),
            },
            MemoryActor::User,
            now(),
        )?;
        let mut changes = vec![MemoryChange::Insert(record.clone())];
        let replaces = args.replaces.trim();
        if !replaces.is_empty() {
            let active = self.store.search(self.scope()).await?;
            changes.push(retire(
                find_active(&active, replaces, "")?,
                MemoryEventOp::Retire,
            ));
        }
        self.store.commit(self.commit(changes)).await?;
        let replaced = if replaces.is_empty() {
            String::new()
        } else {
            format!(" It replaces '{replaces}'.")
        };
        Ok(format!(
            "Saved {} '{}' — applies when: {}.{replaced}\nSaved content: {}",
            kind.as_str(),
            record.key,
            record.situation,
            record.content
        ))
    }

    async fn forget(&self, args: &SaveMemoryArgs) -> MemoryResult<String> {
        let active = self.store.search(self.scope()).await?;
        let record = find_active(&active, &args.key, &args.id)?;
        let key = record.key.clone();
        self.store
            .commit(self.commit(vec![retire(record, MemoryEventOp::Forget)]))
            .await?;
        Ok(format!(
            "Forgot '{key}'. It will not be selected again; already shown context remains until the next compaction or session."
        ))
    }

    async fn pin(&self, args: &SaveMemoryArgs) -> MemoryResult<String> {
        let text = self.bullet_text(args).await?;
        pinned_memories::pin(&self.pinned_memories, &text)?;
        Ok(format!(
            "Pinned; the system prompt carries it from the next compaction or session.\nPinned: {text}"
        ))
    }

    async fn unpin(&self, args: &SaveMemoryArgs) -> MemoryResult<String> {
        let text = self.bullet_text(args).await?;
        pinned_memories::unpin(&self.pinned_memories, &text)?;
        Ok(format!("Unpinned: {text}"))
    }

    /// The bullet a pin or unpin names: new text, or an existing memory's
    /// content when the call gives only its id or key.
    async fn bullet_text(&self, args: &SaveMemoryArgs) -> MemoryResult<String> {
        if !args.content.trim().is_empty() {
            return Ok(args.content.trim().to_string());
        }
        if args.id.trim().is_empty() && args.key.trim().is_empty() {
            return Err(invalid(
                "give the text to pin, or the id or key of a memory",
            ));
        }
        let active = self.store.search(self.scope()).await?;
        Ok(find_active(&active, &args.key, &args.id)?.content)
    }
}

impl Tool for SaveMemoryTool {
    type Args = SaveMemoryArgs;

    fn name(&self) -> &str {
        Self::NAME
    }

    fn description(&self) -> &str {
        "Save, forget, pin or unpin a durable preference, fact, procedure, pitfall or reference the user asked you to keep. \
Write content in natural wording; give scope, paths and situation when they matter. \
Pinned memories are always in the prompt, so pin only what applies everywhere. Do not save credentials or one-off task instructions."
    }

    fn parameters(&self) -> Map<String, Value> {
        object_schema(
            Map::from_iter([
                (
                    "action".to_string(),
                    serde_json::json!({"type": "string", "enum": ["remember", "forget", "pin", "unpin"], "description": "remember saves a memory; forget removes one; pin keeps one in every prompt; unpin stops that"}),
                ),
                (
                    "kind".to_string(),
                    serde_json::json!({"type": "string", "enum": ["preference", "fact", "procedure", "pitfall", "reference"], "description": "required for remember"}),
                ),
                (
                    "content".to_string(),
                    serde_json::json!({"type": "string", "maxLength": 4096, "description": "what to remember, or the bullet to pin or unpin"}),
                ),
                (
                    "key".to_string(),
                    serde_json::json!({"type": "string", "maxLength": 64, "description": "kebab-case handle; the target for forget, or the memory to pin or unpin"}),
                ),
                (
                    "id".to_string(),
                    serde_json::json!({"type": "string", "maxLength": 64, "description": "the id of an existing memory: the target for forget, or the memory to pin or unpin"}),
                ),
                (
                    "situation".to_string(),
                    serde_json::json!({"type": "string", "maxLength": 512, "description": "when this applies, e.g. 'reviewing pull requests'"}),
                ),
                (
                    "replaces".to_string(),
                    serde_json::json!({"type": "string", "maxLength": 64, "description": "key of the memory this one supersedes"}),
                ),
                (
                    "scope".to_string(),
                    serde_json::json!({"type": "string", "enum": ["global", "repository"], "default": "global", "description": "repository limits the memory to the current repository; path_globs imply it."}),
                ),
                (
                    "path_globs".into(),
                    serde_json::json!({"type": "array", "maxItems": 16, "items": {"type": "string"}, "description": "Repository-relative file patterns this memory attaches to."}),
                ),
                (
                    "keywords".into(),
                    serde_json::json!({"type": "array", "maxItems": 32, "items": {"type": "string"}, "description": "Extra words a later search for this memory might use"}),
                ),
            ]),
            &["action"],
        )
    }

    async fn run(
        &self,
        args: SaveMemoryArgs,
        _context: &ExecutionContext,
    ) -> Result<ToolOutput, ToolError> {
        Ok(self.save(args).await)
    }
}

/// The change that takes `record` out of selection under `operation`.
fn retire(mut record: MemoryRecord, operation: MemoryEventOp) -> MemoryChange {
    record.version += 1;
    record.status = MemoryStatus::Retired;
    MemoryChange::Update { record, operation }
}

/// Resolve a key or id to exactly one active record among `records`.
fn find_active(records: &[MemoryRecord], key: &str, id: &str) -> MemoryResult<MemoryRecord> {
    let (key, id) = (key.trim(), id.trim());
    if key.is_empty() && id.is_empty() {
        return Err(invalid("a key or id is required"));
    }
    let matches: Vec<&MemoryRecord> = records
        .iter()
        .filter(|record| record.status == MemoryStatus::Active)
        .filter(|record| {
            if id.is_empty() {
                record.key == key
            } else {
                record.id == id
            }
        })
        .collect();
    match matches.as_slice() {
        [] if id.is_empty() => Err(invalid(format!(
            "no active memory named {key:?} applies here; use search_memory to find the right key"
        ))),
        [] => Err(invalid(format!(
            "no active memory with id {id:?} applies here"
        ))),
        [record] => Ok((*record).clone()),
        several => Err(invalid(format!(
            "{} memories share the key {key:?}; name one by id: {}",
            several.len(),
            several
                .iter()
                .map(|record| format!(
                    "{} ({})",
                    record.id,
                    record.content.chars().take(60).collect::<String>()
                ))
                .collect::<Vec<_>>()
                .join("; ")
        ))),
    }
}

#[cfg(test)]
#[path = "save_memory_tool_test.rs"]
mod tests;
