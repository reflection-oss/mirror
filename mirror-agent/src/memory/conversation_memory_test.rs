use assert_matches::assert_matches;
use mirror_protocol::ChatRecord;
use mirror_protocol::MetadataRecord;
use mirror_protocol::ToolMessage;
use mirror_protocol::TouchedKind;
use mirror_protocol::TouchedPath;
use pretty_assertions::assert_eq;

use super::*;
use crate::memory::create_record;
use crate::memory::testing::seeded_store;
use crate::memory::testing::TestWorkspace;
use crate::memory::testing::REPO;
use crate::memory::MemoryActor;
use crate::memory::MemoryDraft;
use crate::memory::MemoryRecord;
use crate::memory::MemoryType;

fn record(key: &str, content: &str, globs: &[&str]) -> MemoryRecord {
    create_record(
        MemoryDraft {
            r#type: MemoryType::Fact,
            key: key.into(),
            repository: REPO.into(),
            path_globs: globs.iter().map(|glob| glob.to_string()).collect(),
            situation: "working in the harness".into(),
            keywords: Vec::new(),
            content: content.into(),
            source: String::new(),
        },
        MemoryActor::User,
        1_000,
    )
    .expect("record")
}

async fn memory_with(records: Vec<MemoryRecord>) -> (tempfile::TempDir, ConversationMemory) {
    let directory = tempfile::tempdir().expect("directory");
    let store = seeded_store(directory.path(), records).await;
    let memory = ConversationMemory::new(
        store,
        Arc::new(TestWorkspace::under("crates/agent")),
        "session",
    );
    (directory, memory)
}

fn user(content: &str) -> AgentMessage {
    AgentMessage {
        payload: Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::User(UserMessage {
                checkpoint: None,
                content: content.to_string(),
            })),
            ..ChatRecord::default()
        })),
        ..AgentMessage::default()
    }
}

fn tool_reply(touched: &[(&str, TouchedKind)]) -> AgentMessage {
    AgentMessage {
        payload: Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::Tool(ToolMessage::default())),
            success: true,
            ..ChatRecord::default()
        })),
        metadata: Some(MetadataRecord {
            touched: touched
                .iter()
                .map(|(path, kind)| TouchedPath {
                    path: path.to_string(),
                    kind: *kind as i32,
                })
                .collect(),
            ..MetadataRecord::default()
        }),
        ..AgentMessage::default()
    }
}

fn recall_text(entry: &AgentMessage) -> &str {
    match &entry.payload {
        Some(agent_message::Payload::Extension(extension)) => extension
            .model_message
            .as_ref()
            .map(|message| message.content.as_str())
            .expect("recall has a model message"),
        other => panic!("expected a recall, got {other:?}"),
    }
}

#[tokio::test]
async fn a_user_message_recalls_matching_memories_once_per_window() {
    let compaction = record(
        "compactor-rebuild",
        "The compactor is rebuilt for every prompt swap.",
        &[],
    );
    let mut fillers: Vec<MemoryRecord> = (0..8)
        .map(|index| {
            record(
                &format!("filler-{index}"),
                &format!("Unrelated note {index} about gazelle."),
                &[],
            )
        })
        .collect();
    fillers.push(compaction.clone());
    let (_directory, memory) = memory_with(fillers).await;
    let question = user("why does compaction rebuild the compactor");

    let first = memory
        .on_input(
            std::slice::from_ref(&question),
            std::slice::from_ref(&question),
        )
        .await;

    assert_eq!(first.len(), 1);
    assert!(recall_text(&first[0]).contains("The compactor is rebuilt"));
    assert!(recall_text(&first[0]).contains(&format!("id {}", compaction.id)));

    let history = vec![question.clone(), first[0].clone()];
    let again = memory
        .on_input(&history, std::slice::from_ref(&question))
        .await;
    assert!(
        again.is_empty(),
        "a record shown in this window is not shown again"
    );

    let trivial = user("ok");
    assert!(memory
        .on_input(&history, std::slice::from_ref(&trivial))
        .await
        .is_empty());
}

#[tokio::test]
async fn a_tool_batch_attaches_memories_for_the_files_it_touched() {
    let pitfall = record(
        "borrow-rules",
        "Mind the loop's borrow rules.",
        &["crates/agent/src/react/**"],
    );
    let unrelated = record("elsewhere", "About another directory.", &["docs/**"]);
    let (_directory, memory) = memory_with(vec![pitfall.clone(), unrelated]).await;
    let reply = tool_reply(&[("src/react/agent.rs", TouchedKind::Read)]);

    let attached = memory
        .on_tool_replies(std::slice::from_ref(&reply), std::slice::from_ref(&reply))
        .await;

    assert_eq!(attached.len(), 1);
    assert!(recall_text(&attached[0]).contains("attached: matched crates/agent/src/react/agent.rs"));
    assert!(recall_text(&attached[0]).contains("Mind the loop's borrow rules."));

    let no_files = tool_reply(&[]);
    assert!(memory
        .on_tool_replies(
            std::slice::from_ref(&no_files),
            std::slice::from_ref(&no_files)
        )
        .await
        .is_empty());
}

#[tokio::test]
async fn surfaced_reads_ids_and_size_from_the_windows_recalls() {
    let (_directory, memory) =
        memory_with(vec![record("one", "Alpha beta gamma delta.", &[])]).await;
    let question = user("alpha beta gamma delta");
    let recall = memory
        .on_input(
            std::slice::from_ref(&question),
            std::slice::from_ref(&question),
        )
        .await;
    assert_matches!(recall.as_slice(), [_]);

    let shown = surfaced([&question, &recall[0]]);

    assert_eq!(shown.ids.len(), 1);
    assert_eq!(shown.chars, recall_text(&recall[0]).len());
}
