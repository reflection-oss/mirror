use assert_matches::assert_matches;
use mirror_protocol::{
    agent_message, chat_record, AgentMessage, ChatRecord, InferenceConfig, UserMessage,
};
use pretty_assertions::assert_eq;

use crate::local::{
    conversation_directory, conversation_messages, list_conversations, prompt_history, SQLiteStore,
};
use crate::Store;
use mirror_protocol::daemon::{GetConversationMessagesRequest, ListConversationsRequest};

fn query(after_id: &str, limit: u32) -> ListConversationsRequest {
    ListConversationsRequest {
        after_id: after_id.into(),
        limit,
        ..Default::default()
    }
}

fn user(id: &str, text: &str, seconds: i64) -> AgentMessage {
    AgentMessage {
        id: id.into(),
        timestamp: Some(prost_types::Timestamp { seconds, nanos: 0 }),
        payload: Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::User(UserMessage {
                content: text.into(),
                ..Default::default()
            })),
            ..Default::default()
        })),
        inference: Some(InferenceConfig {
            model: "synthetic".into(),
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[tokio::test]
async fn history_reads_live_writers_and_fork_boundaries_without_projection() {
    let directory = tempfile::tempdir().unwrap();
    let database = directory.path().join("state.db");
    let mut parent = SQLiteStore::open(&database, "parent", directory.path(), true, false).unwrap();
    parent
        .append(
            "parent",
            0,
            &mut [
                user("one", "  first\n prompt  ", 10),
                user("two", "abandoned", 20),
            ],
        )
        .await
        .unwrap();
    SQLiteStore::fork(&database, "parent", "two", "child").unwrap();
    let mut child = SQLiteStore::open(&database, "child", directory.path(), false, false).unwrap();
    child
        .append("child", 1, &mut [user("three", "child prompt", 30)])
        .await
        .unwrap();
    let before = std::fs::read(&database).unwrap();

    assert_eq!(
        conversation_directory(&database, "child").unwrap(),
        Some(
            directory
                .path()
                .canonicalize()
                .unwrap()
                .to_string_lossy()
                .into_owned()
        )
    );
    assert_matches!(conversation_directory(&database, "missing").unwrap(), None);
    let page = list_conversations(&database, &query("", 1)).unwrap();
    assert_eq!(page.len(), 1);
    assert_eq!(page[0].id, "child");
    assert_eq!(page[0].title, "first prompt");
    assert_eq!(page[0].model, "synthetic");
    assert_eq!(
        page[0].directory,
        directory.path().canonicalize().unwrap().to_string_lossy()
    );
    assert_matches!(page[0].created_at, Some(timestamp) if timestamp.seconds == 10);
    assert_matches!(page[0].updated_at, Some(timestamp) if timestamp.seconds == 30);
    let next = list_conversations(&database, &query(&page[0].id, 1)).unwrap();
    assert_eq!(next[0].id, "parent");
    assert_matches!(next[0].updated_at, Some(timestamp) if timestamp.seconds == 20);
    assert!(list_conversations(&database, &query("parent", 1))
        .unwrap()
        .is_empty());
    assert_eq!(
        prompt_history(&database, "child", 10).unwrap(),
        ["child prompt", "  first\n prompt  ", "abandoned"]
    );
    assert_eq!(
        prompt_history(&database, "child", 1).unwrap(),
        ["child prompt"]
    );
    let latest = conversation_messages(
        &database,
        &GetConversationMessagesRequest {
            session_id: "child".into(),
            limit: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(latest.messages.len(), 1);
    assert_eq!(latest.messages[0].id, "three");
    assert_eq!(latest.messages[0].sequence, Some(2));
    assert_eq!(latest.next_before_sequence, Some(2));
    let older = conversation_messages(
        &database,
        &GetConversationMessagesRequest {
            session_id: "child".into(),
            before_sequence: latest.next_before_sequence,
            limit: 1,
        },
    )
    .unwrap();
    assert_eq!(older.messages[0].id, "one");
    assert_matches!(older.next_before_sequence, None);
    assert_eq!(std::fs::read(&database).unwrap(), before);

    drop(child);
    drop(parent);
    assert_eq!(
        prompt_history(&database, "missing", 10).unwrap(),
        ["child prompt", "abandoned", "  first\n prompt  "]
    );
}

#[tokio::test]
async fn history_excludes_display_only_and_blank_user_records() {
    let directory = tempfile::tempdir().unwrap();
    let database = directory.path().join("state.db");
    let mut store = SQLiteStore::open(&database, "session", directory.path(), true, false).unwrap();
    let hidden = AgentMessage {
        payload: Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::User(UserMessage {
                content: "hidden".into(),
                ..Default::default()
            })),
            display: Some("notice".into()),
            ..Default::default()
        })),
        ..Default::default()
    };
    store
        .append(
            "session",
            0,
            &mut [hidden, user("blank", "  ", 2), user("visible", "prompt", 3)],
        )
        .await
        .unwrap();
    assert_eq!(
        list_conversations(&database, &query("", 100)).unwrap()[0].title,
        "prompt"
    );
    assert_eq!(
        prompt_history(&database, "session", 100).unwrap(),
        ["prompt"]
    );
}

#[tokio::test]
async fn first_append_preserves_reserved_creation_time() {
    let directory = tempfile::tempdir().unwrap();
    let database = directory.path().join("state.db");
    let created = prost_types::Timestamp {
        seconds: 0,
        nanos: 0,
    };
    crate::local::create_conversation_metadata(&database, "reserved", directory.path(), created)
        .unwrap();
    let mut store =
        SQLiteStore::open(&database, "reserved", directory.path(), true, false).unwrap();
    store
        .append("reserved", 0, &mut [user("one", "first prompt", 10)])
        .await
        .unwrap();
    let summaries = list_conversations(&database, &query("", 1)).unwrap();
    assert_eq!(summaries[0].created_at, Some(created));
    assert_matches!(summaries[0].updated_at, Some(timestamp) if timestamp.seconds == 10);
}

#[tokio::test]
async fn indexed_summaries_do_not_read_transcript_payloads() {
    let directory = tempfile::tempdir().unwrap();
    let database = directory.path().join("state.db");
    let mut store = SQLiteStore::open(&database, "session", directory.path(), true, false).unwrap();
    store
        .append("session", 0, &mut [user("one", "indexed title", 10)])
        .await
        .unwrap();
    let connection = rusqlite::Connection::open(&database).unwrap();
    connection
        .execute("UPDATE react_entry SET payload = x'ff'", [])
        .unwrap();
    let summaries = list_conversations(&database, &query("", 100)).unwrap();
    assert_eq!(summaries[0].title, "indexed title");
    assert_matches!(summaries[0].updated_at, Some(timestamp) if timestamp.seconds == 10);
    assert_matches!(prompt_history(&database, "session", 1), Err(_));
}

#[tokio::test]
async fn message_pages_bound_bytes_and_do_not_decode_older_history() {
    let directory = tempfile::tempdir().unwrap();
    let database = directory.path().join("state.db");
    let mut store = SQLiteStore::open(&database, "session", directory.path(), true, false).unwrap();
    let content = "x".repeat(3 * 1024 * 1024);
    store
        .append(
            "session",
            0,
            &mut [
                user("old", "corrupt", 1),
                user("middle", &content, 2),
                user("newest", &content, 3),
            ],
        )
        .await
        .unwrap();
    let connection = rusqlite::Connection::open(&database).unwrap();
    connection
        .execute(
            "UPDATE react_entry SET payload = x'ff' WHERE sequence = 1",
            [],
        )
        .unwrap();
    let latest = conversation_messages(
        &database,
        &GetConversationMessagesRequest {
            session_id: "session".into(),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(latest.messages.len(), 1);
    assert_eq!(latest.messages[0].id, "newest");
    assert_eq!(latest.next_before_sequence, Some(3));
    let older = conversation_messages(
        &database,
        &GetConversationMessagesRequest {
            session_id: "session".into(),
            before_sequence: latest.next_before_sequence,
            limit: 1,
        },
    )
    .unwrap();
    assert_eq!(older.messages.len(), 1);
    assert_eq!(older.messages[0].id, "middle");
    assert_eq!(older.next_before_sequence, Some(2));
    assert_matches!(
        conversation_messages(
            &database,
            &GetConversationMessagesRequest {
                session_id: "session".into(),
                before_sequence: older.next_before_sequence,
                limit: 1,
            }
        ),
        Err(_)
    );
}

#[tokio::test]
async fn activity_pages_use_identity_ties_and_filter_before_limiting() {
    let directory = tempfile::tempdir().unwrap();
    let database = directory.path().join("state.db");
    let mut alpha = SQLiteStore::open(&database, "alpha", directory.path(), true, false).unwrap();
    let mut bravo = SQLiteStore::open(&database, "bravo", directory.path(), true, false).unwrap();
    let mut old = SQLiteStore::open(&database, "old", directory.path(), true, false).unwrap();
    alpha
        .append("alpha", 0, &mut [user("a", "alpha", 20)])
        .await
        .unwrap();
    bravo
        .append("bravo", 0, &mut [user("b", "bravo", 20)])
        .await
        .unwrap();
    old.append("old", 0, &mut [user("c", "older", 10)])
        .await
        .unwrap();
    let request = ListConversationsRequest {
        recent_first: true,
        limit: 2,
        ..Default::default()
    };
    let first = list_conversations(&database, &request).unwrap();
    assert_eq!(
        first
            .iter()
            .map(|summary| summary.id.as_str())
            .collect::<Vec<_>>(),
        ["bravo", "alpha"]
    );
    let next = list_conversations(
        &database,
        &ListConversationsRequest {
            after_id: first[1].id.clone(),
            before_updated_at: first[1].updated_at,
            ..request.clone()
        },
    )
    .unwrap();
    assert_eq!(
        next.iter()
            .map(|summary| summary.id.as_str())
            .collect::<Vec<_>>(),
        ["old"]
    );
    let filtered = list_conversations(
        &database,
        &ListConversationsRequest {
            session_ids: vec!["old".into()],
            limit: 1,
            ..request
        },
    )
    .unwrap();
    assert_eq!(filtered[0].title, "older");
}

#[test]
fn summary_upgrade_backfills_fork_boundaries_once() {
    use prost::Message;
    let directory = tempfile::tempdir().unwrap();
    let database = directory.path().join("state.db");
    let connection = rusqlite::Connection::open(&database).unwrap();
    connection
        .execute_batch(concat!(
            include_str!("../../migrations/0001_initial.sql"),
            "\n",
            include_str!("../../migrations/0002_session_lineage.sql"),
            "\n",
            include_str!("../../migrations/0003_system_prompts.sql")
        ))
        .unwrap();
    connection.execute("INSERT INTO react_session(id, workspace, lineage, fork_points) VALUES ('parent', '/fixture', '[\"parent\"]', '[9223372036854775807]'), ('child', '/fixture', '[\"parent\",\"child\"]', '[1,9223372036854775807]')", []).unwrap();
    connection.execute("INSERT INTO react_entry(session_id, sequence, entry_id, payload) VALUES ('parent', 1, 'first', ?1), ('parent', 2, 'later', ?2)", rusqlite::params![user("first", "inherited", 10).encode_to_vec(), user("later", "excluded", 20).encode_to_vec()]).unwrap();
    connection.execute("INSERT INTO react_session(id, workspace, lineage, fork_points) VALUES ('zzz-empty', '/fixture', '[\"zzz-empty\"]', '[9223372036854775807]')", []).unwrap();
    connection.pragma_update(None, "user_version", 3).unwrap();
    drop(connection);
    crate::local::prepare_history(&database).unwrap();
    let before = std::fs::read(&database).unwrap();
    let summaries = list_conversations(&database, &query("", 100)).unwrap();
    assert_eq!(summaries[0].id, "child");
    assert_eq!(summaries[0].title, "inherited");
    assert_matches!(summaries[0].updated_at, Some(timestamp) if timestamp.seconds == 10);
    assert_matches!(summaries[1].updated_at, Some(timestamp) if timestamp.seconds == 20);
    assert_eq!(summaries[2].id, "zzz-empty");
    assert_matches!(summaries[2].created_at, None);
    assert_matches!(summaries[2].updated_at, None);
    let last_page = list_conversations(
        &database,
        &ListConversationsRequest {
            recent_first: true,
            before_updated_at: summaries[0].updated_at,
            after_id: summaries[0].id.clone(),
            limit: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(last_page, vec![summaries[2].clone()]);
    assert_eq!(std::fs::read(&database).unwrap(), before);
}

#[test]
fn absent_history_does_not_create_a_database() {
    let directory = tempfile::tempdir().unwrap();
    let database = directory.path().join("absent/state.db");
    assert!(list_conversations(&database, &query("", 100))
        .unwrap()
        .is_empty());
    assert!(prompt_history(&database, "", 100).unwrap().is_empty());
    assert_matches!(conversation_directory(&database, "missing").unwrap(), None);
    assert!(!database.parent().unwrap().exists());
}

#[test]
fn empty_conversation_is_visible_and_reads_never_migrate() {
    let directory = tempfile::tempdir().unwrap();
    let database = directory.path().join("state.db");
    let store = SQLiteStore::open(&database, "empty", directory.path(), true, false).unwrap();
    let summaries = list_conversations(&database, &query("", 100)).unwrap();
    assert_eq!(summaries.len(), 1);
    assert_eq!(summaries[0].id, "empty");
    assert_eq!(summaries[0].title, "");
    assert_matches!(summaries[0].created_at, Some(timestamp) if timestamp.seconds > 0);
    assert_eq!(summaries[0].updated_at, summaries[0].created_at);
    drop(store);
    let connection = rusqlite::Connection::open(&database).unwrap();
    connection.pragma_update(None, "user_version", 999).unwrap();
    drop(connection);
    let before = std::fs::read(&database).unwrap();
    assert_matches!(list_conversations(&database, &query("", 100)), Err(crate::StoreError::Persistence(message)) if message.contains("no migration was attempted"));
    assert_eq!(std::fs::read(&database).unwrap(), before);
}
