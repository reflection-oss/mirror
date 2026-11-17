use assert_matches::assert_matches;
use mirror_protocol::daemon::SearchConversationsRequest;
use mirror_protocol::{
    agent_message, chat_record, AgentMessage, AssistantMessage, ChatRecord, ContextWindowRecord,
    UserMessage,
};
use pretty_assertions::assert_eq;
use prost::Message;

use crate::local::{index_conversation_history, search_conversations, SQLiteStore};
use crate::Store;

fn user(id: &str, text: &str) -> AgentMessage {
    AgentMessage {
        id: id.into(),
        payload: Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::User(UserMessage {
                content: text.into(),
                ..Default::default()
            })),
            ..Default::default()
        })),
        ..Default::default()
    }
}

fn query(text: &str) -> SearchConversationsRequest {
    SearchConversationsRequest {
        query: text.into(),
        ..Default::default()
    }
}

#[tokio::test]
async fn search_is_incremental_read_only_and_respects_fork_boundaries() {
    let directory = tempfile::tempdir().expect("fixture");
    let database = directory.path().join("state.db");
    let mut parent =
        SQLiteStore::open(&database, "parent", directory.path(), true, false).expect("parent");
    parent
        .append(
            "parent",
            0,
            &mut [
                user("one", "albatross prior"),
                user("two", "abandoned héron"),
            ],
        )
        .await
        .expect("append");
    SQLiteStore::fork(&database, "parent", "two", "child").expect("fork before second input");
    let before = std::fs::read(&database).expect("database");
    assert!(!index_conversation_history(&database).expect("index committed messages"));
    let inherited = search_conversations(&database, &query("albatross")).expect("search");
    assert_eq!(
        inherited
            .matches
            .iter()
            .map(|entry| entry.session_id.as_str())
            .collect::<Vec<_>>(),
        ["child", "parent"]
    );
    assert_eq!(inherited.matches[0].message_id, "one");
    assert_eq!(inherited.matches[0].sequence, 1);
    assert_eq!(inherited.matches[0].snippet, "albatross prior");
    let abandoned = search_conversations(&database, &query("heron")).expect("accent insensitive");
    assert_eq!(abandoned.matches.len(), 1);
    assert_eq!(abandoned.matches[0].session_id, "parent");
    assert_eq!(std::fs::read(&database).expect("database"), before);
    assert_matches!(
        parent
            .append(
                "parent",
                2,
                &mut [user("three", "rollback token"), user("one", "duplicate")]
            )
            .await,
        Err(_)
    );
    assert_eq!(
        search_conversations(&database, &query("rollback"))
            .expect("rolled back")
            .matches
            .len(),
        0
    );
}

#[tokio::test]
async fn search_indexes_only_visible_chat_text_and_treats_queries_as_literals() {
    let directory = tempfile::tempdir().expect("fixture");
    let database = directory.path().join("state.db");
    let mut store =
        SQLiteStore::open(&database, "session", directory.path(), true, false).expect("store");
    let assistant = AgentMessage {
        payload: Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::Assistant(AssistantMessage {
                content: Some("Use the migration index for 東京 records".into()),
                reasoning_content: Some("private-reasoning-canary".into()),
                ..Default::default()
            })),
            ..Default::default()
        })),
        ..Default::default()
    };
    let system = AgentMessage {
        payload: Some(agent_message::Payload::ContextWindow(ContextWindowRecord {
            system_prompt: "system-canary".into(),
            ..Default::default()
        })),
        ..Default::default()
    };
    let hidden = AgentMessage {
        payload: Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::User(UserMessage {
                content: "hidden-canary".into(),
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
            &mut [
                user("prompt", "Please investigate"),
                assistant,
                system,
                hidden,
            ],
        )
        .await
        .expect("append");
    assert!(!index_conversation_history(&database).expect("index committed messages"));
    assert_eq!(
        search_conversations(&database, &query("migr index 東京"))
            .expect("word prefixes")
            .matches
            .len(),
        1
    );
    assert_eq!(
        search_conversations(&database, &query("canary"))
            .expect("no private text")
            .matches
            .len(),
        0
    );
    assert_eq!(
        search_conversations(&database, &query("\" OR * - :"))
            .expect("no query syntax")
            .matches
            .len(),
        0
    );
    assert_matches!(
        search_conversations(&database, &query(&"x".repeat(1025))),
        Err(_)
    );
}

#[tokio::test]
async fn search_filters_before_limit_and_returns_one_message_per_conversation() {
    let directory = tempfile::tempdir().expect("fixture");
    let database = directory.path().join("state.db");
    let mut first =
        SQLiteStore::open(&database, "one", directory.path(), true, false).expect("one");
    let mut second =
        SQLiteStore::open(&database, "two", directory.path(), true, false).expect("two");
    first
        .append(
            "one",
            0,
            &mut [user("one-a", "needle"), user("one-b", "needle")],
        )
        .await
        .expect("first");
    second
        .append("two", 0, &mut [user("two-a", "a longer needle answer")])
        .await
        .expect("second");
    assert!(!index_conversation_history(&database).expect("index committed messages"));
    let result = search_conversations(
        &database,
        &SearchConversationsRequest {
            query: "needle".into(),
            limit: 1,
            session_ids: vec!["two".into()],
        },
    )
    .expect("scoped");
    assert_eq!(result.matches[0].session_id, "two");
    assert_eq!(
        search_conversations(&database, &query("needle"))
            .expect("deduplicated")
            .matches
            .len(),
        2
    );
    second.delete_session("two").expect("delete");
    assert!(index_conversation_history(&database).expect("prune deleted projection"));
    assert!(!index_conversation_history(&database).expect("pruning complete"));
    let cache = rusqlite::Connection::open(directory.path().join("state.db.search-v1.sqlite"))
        .expect("cache");
    assert_eq!(
        cache
            .query_row(
                "SELECT COUNT(*) FROM entries WHERE session_id = 'two'",
                [],
                |row| row.get::<_, i64>(0)
            )
            .expect("deleted text removed"),
        0
    );
    assert_eq!(
        search_conversations(&database, &query("needle"))
            .expect("deleted")
            .matches
            .len(),
        1
    );
}

#[tokio::test]
async fn sidecar_backfills_without_migrating_history_or_changing_older_writers() {
    let directory = tempfile::tempdir().expect("fixture");
    let database = directory.path().join("state #?%.db");
    let mut store =
        SQLiteStore::open(&database, "session", directory.path(), true, false).expect("store");
    store
        .append("session", 0, &mut [user("old", "backfill albatross")])
        .await
        .expect("append");
    drop(store);
    let connection = rusqlite::Connection::open(&database).expect("legacy fixture");
    let before = std::fs::read(&database).expect("history before search");
    let pending = search_conversations(&database, &query("albatross")).expect("pending");
    assert!(pending.indexing);
    assert_eq!(pending.matches.len(), 0);
    assert!(!index_conversation_history(&database).expect("backfill"));
    assert_eq!(
        std::fs::read(&database).expect("unmodified history"),
        before
    );
    assert_eq!(
        connection
            .query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
            .expect("schema"),
        4
    );
    assert_eq!(connection.query_row("SELECT COUNT(*) FROM sqlite_master WHERE name LIKE '%search%' OR name LIKE '%fts%'", [], |row| row.get::<_, i64>(0)).expect("no search tables or triggers"), 0);
    assert_eq!(
        search_conversations(&database, &query("albatross"))
            .expect("indexed")
            .matches
            .len(),
        1
    );
    connection.execute("INSERT INTO react_entry(session_id, sequence, entry_id, payload) VALUES ('session', 2, 'legacy', ?)", [user("legacy", "legacy flamingo").encode_to_vec()]).expect("older writer");
    assert!(!index_conversation_history(&database).expect("incremental indexing"));
    assert_eq!(
        search_conversations(&database, &query("flamingo"))
            .expect("indexed")
            .matches[0]
            .message_id,
        "legacy"
    );
    assert!(!index_conversation_history(&database).expect("idle"));
    let mut reopened = SQLiteStore::open(&database, "session", directory.path(), false, false)
        .expect("v4 writer reopens");
    assert_eq!(
        reopened.load("session", true).await.expect("history").len(),
        2
    );
}

#[tokio::test]
async fn deleted_row_ids_never_reuse_stale_message_text() {
    let directory = tempfile::tempdir().expect("fixture");
    let database = directory.path().join("state.db");
    let mut first =
        SQLiteStore::open(&database, "first", directory.path(), true, false).expect("first");
    let mut second =
        SQLiteStore::open(&database, "second", directory.path(), true, false).expect("second");
    first
        .append("first", 0, &mut [user("one", "retained kestrel")])
        .await
        .expect("first append");
    second
        .append("second", 0, &mut [user("two", "deleted albatross")])
        .await
        .expect("second append");
    assert!(!index_conversation_history(&database).expect("index"));
    second.delete_session("second").expect("delete last row");
    first
        .append("first", 1, &mut [user("three", "replacement flamingo")])
        .await
        .expect("reuse row id");
    let stale = search_conversations(&database, &query("albatross")).expect("no stale hit");
    assert!(stale.indexing);
    assert!(stale.matches.is_empty());
    assert!(!index_conversation_history(&database).expect("index replacement"));
    assert_eq!(
        search_conversations(&database, &query("flamingo"))
            .expect("new hit")
            .matches[0]
            .message_id,
        "three"
    );
}

#[tokio::test]
async fn replaced_history_rebuilds_even_when_message_identities_are_the_same() {
    let directory = tempfile::tempdir().expect("fixture");
    let database = directory.path().join("state.db");
    let mut store =
        SQLiteStore::open(&database, "session", directory.path(), true, false).expect("store");
    store
        .append("session", 0, &mut [user("one", "original albatross")])
        .await
        .expect("append");
    assert!(!index_conversation_history(&database).expect("index"));
    drop(store);
    std::fs::rename(&database, directory.path().join("old.db")).expect("replace history");
    let mut replacement = SQLiteStore::open(&database, "session", directory.path(), true, false)
        .expect("replacement");
    replacement
        .append("session", 0, &mut [user("one", "replacement flamingo")])
        .await
        .expect("append");
    let stale = search_conversations(&database, &query("albatross")).expect("old cache hidden");
    assert!(stale.indexing);
    assert!(stale.matches.is_empty());
    assert!(!index_conversation_history(&database).expect("rebuild"));
    assert_eq!(
        search_conversations(&database, &query("flamingo"))
            .expect("replacement hit")
            .matches[0]
            .snippet,
        "replacement flamingo"
    );
}

#[tokio::test]
async fn removed_or_incompatible_cache_rebuilds_without_touching_history() {
    use std::os::unix::fs::PermissionsExt;

    let directory = tempfile::tempdir().expect("fixture");
    let database = directory.path().join("state.db");
    let cache = directory.path().join("state.db.search-v1.sqlite");
    let mut store =
        SQLiteStore::open(&database, "session", directory.path(), true, false).expect("store");
    store
        .append("session", 0, &mut [user("one", "albatross")])
        .await
        .expect("append");
    let before = std::fs::read(&database).expect("history");
    assert!(!index_conversation_history(&database).expect("index"));
    assert_eq!(
        std::fs::metadata(&cache)
            .expect("private cache")
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
    std::fs::remove_file(&cache).expect("remove cache");
    let pending = search_conversations(&database, &query("albatross")).expect("cache absent");
    assert!(pending.indexing);
    assert!(pending.matches.is_empty());
    assert!(!index_conversation_history(&database).expect("rebuild removed cache"));
    rusqlite::Connection::open(&cache)
        .expect("cache")
        .execute_batch("PRAGMA user_version = 99; DROP TABLE message_fts;")
        .expect("incompatible cache");
    let pending =
        search_conversations(&database, &query("albatross")).expect("incompatible cache hidden");
    assert!(pending.indexing);
    assert!(pending.matches.is_empty());
    assert!(!index_conversation_history(&database).expect("rebuild incompatible cache"));
    assert_eq!(
        search_conversations(&database, &query("albatross"))
            .expect("rebuilt hit")
            .matches
            .len(),
        1
    );
    assert_eq!(std::fs::read(&database).expect("untouched history"), before);
}

#[tokio::test]
async fn backfill_is_bounded_and_a_broken_index_does_not_block_history_commits() {
    let directory = tempfile::tempdir().expect("fixture");
    let database = directory.path().join("state.db");
    let mut store =
        SQLiteStore::open(&database, "session", directory.path(), true, false).expect("store");
    store
        .append("session", 0, &mut vec![user("", "message text"); 33])
        .await
        .expect("append batch");
    assert!(index_conversation_history(&database).expect("first batch"));
    assert!(!index_conversation_history(&database).expect("second batch"));
    let cache = directory.path().join("state.db.search-v1.sqlite");
    std::fs::write(&cache, b"corrupt disposable cache").expect("broken cache");
    store
        .append("session", 33, &mut [user("after", "safe history")])
        .await
        .expect("index never owns commits");
    assert!(index_conversation_history(&database).expect("rebuild first batch"));
    assert!(!index_conversation_history(&database).expect("rebuild second batch"));
    assert_eq!(
        search_conversations(&database, &query("safe history"))
            .expect("rebuilt")
            .matches[0]
            .message_id,
        "after"
    );
    assert_eq!(
        store
            .load("session", true)
            .await
            .expect("canonical history")
            .last()
            .map(|entry| entry.id.as_str()),
        Some("after")
    );
}
