//! What a runtime supplies to open memory for one session.

use std::path::PathBuf;
use std::sync::Arc;

use super::ConversationMemory;
use super::MemoryStore;
use super::SaveMemoryTool;
use super::SearchMemoryTool;
use crate::react::ContextHooks;
use crate::tools::AnyTool;
use crate::Workspace;

/// The capabilities memory runs on. The local worker builds them from its
/// configuration; a hosted session builds them from its session row, its
/// sandbox and its database.
pub struct MemoryEnvironment {
    /// Already bound to the owner.
    pub store: Box<dyn MemoryStore>,
    pub workspace: Box<dyn Workspace>,
    pub conversation_id: String,
    /// Memory opens only for a private session: its one participant is the
    /// owner, so nothing recalled can reach anyone else.
    pub private_session: bool,
    /// The owner's `PINNED_MEMORIES.md`; its bullets are in the prompt at
    /// every snapshot, and `save_memory` pins and unpins there.
    pub pinned_memories: PathBuf,
}

/// Memory as the session composes it: context hooks for the service and
/// the tools the model may call.
pub struct PersonalMemory {
    pub hooks: Box<dyn ContextHooks>,
    pub tools: Vec<Arc<dyn AnyTool>>,
    /// The pinned file, for the prompt builder to load.
    pub pinned_memories: PathBuf,
}

/// Open memory for a session, or `None` when the session must not have it.
pub fn open_memory(environment: MemoryEnvironment) -> Option<PersonalMemory> {
    if !environment.private_session {
        return None;
    }
    // Memory hands the store and workspace to everything it composes for the
    // session; they are shared from here, one handle each.
    let store: Arc<dyn MemoryStore> = Arc::from(environment.store);
    let workspace: Arc<dyn Workspace> = Arc::from(environment.workspace);
    let search = SearchMemoryTool::new(store.clone(), workspace.clone());
    let save = SaveMemoryTool::new(
        store.clone(),
        workspace.clone(),
        environment.conversation_id.clone(),
        environment.pinned_memories.clone(),
    );
    let memory = ConversationMemory::new(store, workspace, environment.conversation_id);
    Some(PersonalMemory {
        hooks: Box::new(memory),
        tools: vec![Arc::new(search), Arc::new(save)],
        pinned_memories: environment.pinned_memories,
    })
}
