use std::path::Path;

use crate::local::sqlite_migrations::v3_context_windows::{
    LegacyCompaction, LegacyFields, LegacyMetadata,
};
use assert_matches::assert_matches;
use mirror_protocol::agent_message;
use mirror_protocol::chat_record;
use mirror_protocol::AgentMessage;
use pretty_assertions::assert_eq;
use prost::Message;
use rusqlite::params;
use rusqlite::Connection;

use super::*;

fn user_entry(content: &str) -> AgentMessage {
    AgentMessage {
        payload: Some(agent_message::Payload::Chat(mirror_protocol::ChatRecord {
            message: Some(chat_record::Message::User(mirror_protocol::UserMessage {
                checkpoint: None,
                content: content.to_string(),
            })),
            success: true,
            ..mirror_protocol::ChatRecord::default()
        })),
        ..AgentMessage::default()
    }
}

fn system_entry(content: &str) -> AgentMessage {
    AgentMessage {
        payload: Some(agent_message::Payload::ContextWindow(
            mirror_protocol::ContextWindowRecord {
                system_prompt: content.to_string(),
                ..Default::default()
            },
        )),
        ..AgentMessage::default()
    }
}

#[tokio::test]
async fn system_prompts_are_deduplicated_transactionally_and_survive_forks() {
    let directory = tempfile::tempdir().expect("directory");
    let database = directory.path().join("react.db");
    let mut store = open(&database, "parent", directory.path(), true);
    let mut first = vec![system_entry("prompt"), identified("fork", "user")];
    store.append("parent", 0, &mut first).await.expect("append");
    assert_matches!(first[1].chat_message(), Some(chat_record::Message::User(user)) if user.checkpoint.is_none());
    let mut next = vec![
        system_entry("prompt"),
        system_entry("changed"),
        system_entry(""),
    ];
    store
        .append("parent", 2, &mut next)
        .await
        .expect("compact again");
    let mut expected = first.clone();
    expected.extend(next);
    drop(store);

    let mut store = open(&database, "parent", directory.path(), false);
    assert_eq!(store.load("parent", true).await.expect("reload"), expected);
    assert_eq!(
        store.load("parent", false).await.expect("latest window"),
        expected[4..]
    );
    let count: i64 = store
        .connection
        .query_row("SELECT COUNT(*) FROM react_system_prompt", [], |row| {
            row.get(0)
        })
        .expect("count prompts");
    assert_eq!(count, 3);
    let stored: Vec<u8> = store
        .connection
        .query_row(
            "SELECT payload FROM react_entry WHERE sequence = 1",
            [],
            |row| row.get(0),
        )
        .expect("stored payload");
    assert_matches!(AgentMessage::decode(stored.as_slice()).expect("decode").payload,
        Some(agent_message::Payload::ContextWindow(window)) if window.system_prompt.is_empty());
    let references: i64 = store
        .connection
        .query_row(
            "SELECT COUNT(DISTINCT system_prompt_hash) FROM react_entry WHERE sequence IN (1, 3)",
            [],
            |row| row.get(0),
        )
        .expect("references");
    assert_eq!(references, 1);

    assert_matches!(
        store
            .append("parent", 0, &mut [system_entry("stale")])
            .await,
        Err(StoreError::Stale)
    );
    assert_matches!(
        store
            .append(
                "parent",
                5,
                &mut [system_entry("rolled back"), identified("fork", "duplicate")]
            )
            .await,
        Err(StoreError::MessageIdConflict { .. })
    );
    let count: i64 = store
        .connection
        .query_row("SELECT COUNT(*) FROM react_system_prompt", [], |row| {
            row.get(0)
        })
        .expect("count after failed append");
    assert_eq!(count, 3);
    assert_eq!(
        store.load("parent", true).await.expect("unchanged"),
        expected
    );

    SQLiteStore::fork(&database, "parent", "fork", "child").expect("fork");
    assert_eq!(
        store.load("child", false).await.expect("inherited prompt"),
        first[..1]
    );
    assert_eq!(
        store
            .messages_page("child", 0, 1)
            .await
            .expect("inherited page"),
        (first[..1].to_vec(), 1)
    );
    assert_eq!(
        store
            .messages_page("parent", 2, 2)
            .await
            .expect("middle page"),
        (expected[2..4].to_vec(), 5)
    );
    assert_eq!(
        store
            .messages_page("parent", u64::MAX, 1)
            .await
            .expect("past end"),
        (Vec::new(), 5)
    );
    assert_matches!(
        store.delete_session("parent"),
        Err(StoreError::SessionHasForks { .. })
    );
    store
        .connection
        .execute(
            "UPDATE react_entry SET payload = X'FF' WHERE sequence = 1",
            [],
        )
        .expect("unreadable archived payload");
    assert_eq!(
        store
            .messages_page("parent", 1, 2)
            .await
            .expect("off-page payload is not decoded"),
        (expected[1..3].to_vec(), 5)
    );
    assert_eq!(
        store
            .load("parent", false)
            .await
            .expect("archived payload is not loaded"),
        expected[4..]
    );
    let mut other = open(&database, "other", directory.path(), true);
    let mut shared = vec![system_entry("prompt")];
    other
        .append("other", 0, &mut shared)
        .await
        .expect("share prompt across sessions");
    store.delete_session("child").expect("delete child");
    store.delete_session("parent").expect("delete parent");
    assert_eq!(
        other
            .load("other", false)
            .await
            .expect("shared prompt survives deletion"),
        shared
    );
    let count: i64 = store
        .connection
        .query_row("SELECT COUNT(*) FROM react_system_prompt", [], |row| {
            row.get(0)
        })
        .expect("globally deduplicated prompts");
    assert_eq!(count, 3);
}

fn identified(id: &str, content: &str) -> AgentMessage {
    AgentMessage {
        id: id.to_string(),
        ..user_entry(content)
    }
}

fn contents(entries: &[AgentMessage]) -> Vec<(u64, &str)> {
    entries
        .iter()
        .map(|entry| {
            let content = match entry.chat_message() {
                Some(chat_record::Message::User(user)) => user.content.as_str(),
                _ => "",
            };
            (entry.sequence.expect("stored sequence"), content)
        })
        .collect()
}

fn open(database: &Path, session_id: &str, workspace: &Path, create: bool) -> SQLiteStore {
    let mut store =
        SQLiteStore::open(database, session_id, workspace, create, false).expect("open store");
    store.checkpoints_enabled = true;
    store
}

#[tokio::test]
async fn appends_are_a_compare_and_swap_across_restarts() {
    let directory = tempfile::tempdir().expect("directory");
    let workspace = tempfile::tempdir().expect("workspace");
    let database = directory.path().join("react.db");
    assert!(std::process::Command::new("git")
        .args(["init", "--quiet"])
        .current_dir(workspace.path())
        .status()
        .expect("Git init")
        .success());
    let mut store = open(&database, "session", workspace.path(), true);
    assert!(store.load("session", true).await.expect("load").is_empty());
    assert_matches!(
        store
            .append("session", 3, &mut [user_entry("future")])
            .await,
        Err(StoreError::Stale)
    );
    assert_matches!(store.append("session", 0, &mut [identified("duplicate", "one"), identified("duplicate", "two")]).await, Err(StoreError::MessageIdConflict { entry_id, .. }) if entry_id == "duplicate");

    let mut committed = vec![user_entry("hello")];
    store
        .append("session", 0, &mut committed)
        .await
        .expect("append");

    assert_eq!(committed[0].sequence, Some(1));
    assert_matches!(committed[0].chat_message(), Some(chat_record::Message::User(user))
        if user.checkpoint.as_ref().is_some_and(|checkpoint| !checkpoint.snapshot_commit.is_empty()));
    assert_matches!(
        store.append("session", 0, &mut [user_entry("stale")]).await,
        Err(StoreError::Stale)
    );
    assert_matches!(
        store
            .append("session", 1, &mut [identified(&committed[0].id, "again")])
            .await,
        Err(StoreError::MessageIdConflict { .. })
    );
    drop(store);

    let mut store = open(&database, "session", workspace.path(), false);
    assert_eq!(
        store.load("session", true).await.expect("reload"),
        committed
    );
    workspace.close().expect("remove workspace");
    let mut without_checkpoint = [user_entry("capture fails")];
    store
        .append("session", 1, &mut without_checkpoint)
        .await
        .expect("append despite capture failure");
    assert_matches!(without_checkpoint[0].chat_message(), Some(chat_record::Message::User(user)) if user.checkpoint.is_none());
    committed.extend(without_checkpoint);
    assert_eq!(store.load("session", true).await.expect("load"), committed);
}

#[tokio::test]
async fn disabled_checkpoints_preserve_history_and_skip_capture() {
    let directory = tempfile::tempdir().expect("database directory");
    let workspace = tempfile::tempdir().expect("workspace");
    let database = directory.path().join("react.db");
    assert!(std::process::Command::new("git")
        .args(["init", "--quiet"])
        .current_dir(workspace.path())
        .status()
        .expect("Git init")
        .success());
    let mut store = open(&database, "session", workspace.path(), true);
    let mut first = [user_entry("enabled")];
    store
        .append("session", 0, &mut first)
        .await
        .expect("append");
    assert_matches!(first[0].chat_message(), Some(chat_record::Message::User(user)) if user.checkpoint.is_some());
    drop(store);

    let mut store = open(&database, "session", workspace.path(), false);
    store.checkpoints_enabled = false;
    // Capture would fail after removing the workspace, but persistence must still work.
    workspace.close().expect("remove workspace");
    let mut second = [user_entry("disabled")];
    store
        .append("session", 1, &mut second)
        .await
        .expect("append without capture");
    assert_matches!(second[0].chat_message(), Some(chat_record::Message::User(user)) if user.checkpoint.is_none());
    assert_eq!(
        store.load("session", true).await.expect("load"),
        [first[0].clone(), second[0].clone()]
    );
}

#[tokio::test]
async fn user_messages_in_one_append_share_one_capture() {
    let directory = tempfile::tempdir().expect("database directory");
    let workspace = tempfile::tempdir().expect("workspace");
    assert!(std::process::Command::new("git")
        .args(["init", "--quiet"])
        .current_dir(workspace.path())
        .status()
        .expect("Git init")
        .success());
    std::fs::write(workspace.path().join("file"), "contents").expect("workspace file");
    std::fs::write(
        workspace.path().join(".gitattributes"),
        "file filter=count\n",
    )
    .expect("filter attributes");
    // Equal commit IDs alone cannot distinguish repeated captures within one second.
    assert!(std::process::Command::new("git")
        .args([
            "config",
            "filter.count.clean",
            "printf 'capture\\n' >> .git/captures; cat; cat .git/captures",
        ])
        .current_dir(workspace.path())
        .status()
        .expect("configure counting filter")
        .success());
    let mut store = open(
        &directory.path().join("react.db"),
        "session",
        workspace.path(),
        true,
    );
    let mut entries = [
        identified("input", "question"),
        system_entry("context"),
        identified("skill", "skill context"),
    ];

    store
        .append("session", 0, &mut entries)
        .await
        .expect("append batch");

    assert_matches!(
        (entries[0].chat_message(), entries[2].chat_message()),
        (Some(chat_record::Message::User(first)), Some(chat_record::Message::User(second)))
            if first.checkpoint.is_some() && first.checkpoint == second.checkpoint
    );
    assert_eq!(store.load("session", true).await.expect("reload"), entries);
    let captures =
        std::fs::read_to_string(workspace.path().join(".git/captures")).expect("capture count");
    assert!(!captures.is_empty());

    store
        .append("session", 3, &mut [system_entry("next context")])
        .await
        .expect("append without user messages");
    store
        .append("session", 4, &mut [])
        .await
        .expect("empty append");
    assert_eq!(
        std::fs::read_to_string(workspace.path().join(".git/captures")).expect("capture count"),
        captures
    );
}

#[tokio::test]
async fn view_of_a_deep_lineage_is_dense_and_excludes_abandoned_rows() {
    let directory = tempfile::tempdir().expect("directory");
    let database = directory.path().join("react.db");
    let mut store = open(&database, "root", directory.path(), true);
    store
        .append(
            "root",
            0,
            &mut [
                identified("r1", "root one"),
                identified("r2", "root two"),
                identified("r3", "root three"),
            ],
        )
        .await
        .expect("seed root");
    let original = store.load("root", true).await.expect("root before fork");
    SQLiteStore::fork(&database, "root", "r3", "middle").expect("fork middle");
    assert_matches!(
        store
            .append("middle", 0, &mut [user_entry("inside inherited prefix")])
            .await,
        Err(StoreError::Stale)
    );
    assert_matches!(
        store
            .append("middle", 4, &mut [user_entry("beyond current head")])
            .await,
        Err(StoreError::Stale)
    );
    assert_matches!(store.append("middle", 2, &mut [identified("r1", "duplicate ancestor")]).await, Err(StoreError::MessageIdConflict { entry_id, .. }) if entry_id == "r1");
    store
        .append(
            "middle",
            2,
            &mut [
                identified("m3", "middle three"),
                identified("m4", "middle four"),
            ],
        )
        .await
        .expect("extend middle");
    SQLiteStore::fork(&database, "middle", "m4", "leaf").expect("fork leaf");
    store
        .append("leaf", 3, &mut [identified("m4", "leaf four")])
        .await
        .expect("extend leaf");

    assert_eq!(
        contents(&store.load("leaf", true).await.expect("leaf")),
        [
            (1, "root one"),
            (2, "root two"),
            (3, "middle three"),
            (4, "leaf four")
        ]
    );

    assert_eq!(
        store.load("root", true).await.expect("root after fork"),
        original
    );
    SQLiteStore::fork(&database, "leaf", "m4", "shallow").expect("fork before the leaf's own rows");
    assert_eq!(
        contents(&store.load("shallow", true).await.expect("shallow")),
        [(1, "root one"), (2, "root two"), (3, "middle three")]
    );
    SQLiteStore::fork(&database, "leaf", "r2", "early").expect("fork inside an ancestor's range");
    assert_eq!(
        contents(&store.load("early", true).await.expect("early")),
        [(1, "root one")]
    );
}

#[tokio::test]
async fn deleting_a_parent_with_a_child_is_refused() {
    let directory = tempfile::tempdir().expect("directory");
    let database = directory.path().join("react.db");
    let mut store = open(&database, "parent", directory.path(), true);
    store
        .append("parent", 0, &mut [identified("user-1", "one")])
        .await
        .expect("seed parent");
    SQLiteStore::fork(&database, "parent", "user-1", "child").expect("fork");

    assert_matches!(
        store.delete_session("parent"),
        Err(StoreError::SessionHasForks { .. })
    );
    store.delete_session("child").expect("delete the child");
    store.delete_session("parent").expect("delete the parent");
    assert_matches!(
        store.load("parent", true).await,
        Err(StoreError::SessionNotFound { .. })
    );
}

#[tokio::test]
async fn upgrades_a_database_written_before_lineage() {
    for version in [0, 1] {
        let directory = tempfile::tempdir().expect("directory");
        let database = directory.path().join("react.db");
        let workspace = std::fs::canonicalize(directory.path()).expect("workspace");
        let connection = Connection::open(&database).expect("open raw database");
        connection
            .execute_batch(
                "CREATE TABLE react_sessions (
                    id TEXT PRIMARY KEY,
                    state_id TEXT NOT NULL,
                    workspace TEXT NOT NULL
                );
                CREATE TABLE react_entries (
                    session_id TEXT NOT NULL REFERENCES react_sessions(id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL,
                    entry_id TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    PRIMARY KEY (session_id, sequence),
                    UNIQUE (session_id, entry_id)
                );",
            )
            .expect("create legacy schema");
        connection
            .execute(
                "INSERT INTO react_sessions(id, state_id, workspace) VALUES (?, ?, ?)",
                params!["legacy", "token", workspace.to_string_lossy()],
            )
            .expect("insert legacy session");
        connection
            .execute(
                "INSERT INTO react_entries(session_id, sequence, entry_id, payload) VALUES (?, ?, ?, ?)",
                params!["legacy", 1, "user-1", identified("user-1", "one").encode_to_vec()],
            )
            .expect("insert legacy entry");
        connection
            .pragma_update(None, "user_version", version)
            .expect("set legacy schema version");
        drop(connection);

        let mut store = open(&database, "legacy", directory.path(), false);
        let entries = store
            .load("legacy", true)
            .await
            .expect("load legacy session");
        assert_eq!(entries[0].id, "user-1");
        assert_eq!(contents(&entries), [(1, "one")]);
        store
            .append("legacy", 1, &mut [identified("user-2", "two")])
            .await
            .expect("append to the upgraded session");
        SQLiteStore::fork(&database, "legacy", "user-2", "child")
            .expect("fork the upgraded session");
        assert_eq!(
            contents(&store.load("child", true).await.expect("child")),
            [(1, "one")]
        );
        drop(store);
        let mut reopened = open(&database, "child", directory.path(), false);
        assert_eq!(
            contents(&reopened.load("child", true).await.expect("reopen child")),
            [(1, "one")]
        );
    }
}

#[tokio::test]
async fn legacy_records_upgrade_without_changing_sequences_or_fork_boundaries() {
    let directory = tempfile::tempdir().expect("directory");
    let database = directory.path().join("react.db");
    let connection = Connection::open(&database).expect("database");
    connection
        .execute_batch(include_str!("../../migrations/0001_initial.sql"))
        .expect("initial schema");
    connection
        .execute_batch(include_str!("../../migrations/0002_session_lineage.sql"))
        .expect("lineage schema");
    connection
        .pragma_update(None, "user_version", 2)
        .expect("old version");
    connection.execute(
        "INSERT INTO react_session(id, workspace, lineage, fork_points) VALUES ('legacy', ?, '[\"legacy\"]', '[9223372036854775807]')",
        [std::fs::canonicalize(directory.path()).expect("workspace").to_string_lossy().as_ref()],
    ).expect("session");
    let legacy = |id: &str, fields: LegacyFields| {
        let mut bytes = AgentMessage {
            id: id.to_string(),
            ..Default::default()
        }
        .encode_to_vec();
        bytes.extend(fields.encode_to_vec());
        bytes
    };
    let metadata = legacy(
        "metadata",
        LegacyFields {
            metadata: Some(LegacyMetadata {
                model: "legacy-model".to_string(),
                compaction_model: Some("legacy-summarizer".to_string()),
                inference: Some(mirror_protocol::InferenceConfig {
                    model: "legacy-model".to_string(),
                    ..Default::default()
                }),
            }),
            compaction: None,
        },
    );
    let system = AgentMessage {
        id: "system".to_string(),
        payload: Some(agent_message::Payload::Chat(mirror_protocol::ChatRecord {
            message: Some(chat_record::Message::System(
                mirror_protocol::SystemMessage {
                    content: "stored prompt".to_string(),
                },
            )),
            ..Default::default()
        })),
        ..Default::default()
    };
    let compacted = legacy(
        "compacted",
        LegacyFields {
            metadata: None,
            compaction: Some(LegacyCompaction {
                summary: "old summary".to_string(),
                recent: vec![identified("before", "retained question")],
                usage: Some(mirror_protocol::Usage {
                    prompt_tokens: 100,
                    ..Default::default()
                }),
            }),
        },
    );
    assert!(AgentMessage::decode(metadata.as_slice())
        .expect("reserved metadata tag")
        .payload
        .is_none());
    assert!(AgentMessage::decode(compacted.as_slice())
        .expect("reserved compaction tag")
        .payload
        .is_none());
    assert_eq!(
        AgentMessage {
            payload: Some(agent_message::Payload::ContextWindow(Default::default())),
            ..Default::default()
        }
        .encode_to_vec(),
        vec![0x5a, 0]
    );
    let rows = [
        ("metadata", metadata.clone()),
        ("system", system.encode_to_vec()),
        (
            "before",
            identified("before", "retained question").encode_to_vec(),
        ),
        ("compacted", compacted),
        (
            "after",
            identified("after", "next question").encode_to_vec(),
        ),
    ];
    for (index, (id, payload)) in rows.iter().enumerate() {
        connection.execute(
            "INSERT INTO react_entry(session_id, sequence, entry_id, payload) VALUES ('legacy', ?, ?, ?)",
            params![index as i64 + 1, id, payload],
        ).expect("legacy entry");
    }
    for (id, head) in [("early", 2), ("late", 4)] {
        connection.execute("INSERT INTO react_session(id, workspace, lineage, fork_points) SELECT ?, workspace, json_array('legacy', ?), json_array(?, 9223372036854775807) FROM react_session WHERE id = 'legacy'", params![id, id, head]).expect("legacy fork");
    }
    let fork_compaction = legacy(
        "fork-compacted",
        LegacyFields::decode(rows[3].1.as_slice()).expect("old compaction"),
    );
    connection.execute("INSERT INTO react_entry(session_id, sequence, entry_id, payload) VALUES ('late', 5, 'fork-compacted', ?)", [fork_compaction]).expect("fork compaction");
    connection
        .execute(
            "UPDATE react_entry SET payload = X'FF' WHERE session_id = 'legacy' AND sequence = 5",
            [],
        )
        .expect("corrupt old entry");
    assert!(SQLiteStore::open(&database, "legacy", directory.path(), false, false).is_err());
    assert_eq!(
        connection
            .query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
            .expect("rolled back version"),
        2
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT payload FROM react_entry WHERE session_id = 'legacy' AND sequence = 1",
                [],
                |row| row.get::<_, Vec<u8>>(0)
            )
            .expect("rolled back metadata"),
        metadata
    );
    connection
        .execute(
            "UPDATE react_entry SET payload = ? WHERE session_id = 'legacy' AND sequence = 5",
            [&rows[4].1],
        )
        .expect("repair old entry");
    drop(connection);
    let mut store = open(&database, "legacy", directory.path(), false);
    let entries = store
        .load("legacy", true)
        .await
        .expect("project legacy history");
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry.sequence)
            .collect::<Vec<_>>(),
        vec![Some(2), Some(3), Some(4), Some(5)]
    );
    assert_matches!(&entries[0].payload, Some(agent_message::Payload::ContextWindow(window)) if window.metadata.as_ref().and_then(|metadata| metadata.fields.get("model")).and_then(|value| value.kind.as_ref()) == Some(&prost_types::value::Kind::StringValue("legacy-model".to_string())) && window.system_prompt == "stored prompt");
    let active = store
        .load("legacy", false)
        .await
        .expect("legacy active window");
    assert_eq!(active, entries[2..]);
    assert_eq!(
        active[0]
            .inference
            .as_ref()
            .expect("migrated inference")
            .model,
        "legacy-model"
    );
    assert_matches!(&active[0].payload, Some(agent_message::Payload::ContextWindow(window)) if window.compaction.as_ref().is_some_and(|compaction| compaction.summary.as_deref() == Some("old summary") && compaction.model == "legacy-summarizer" && compaction.recent[0].id == "before" && compaction.usage.as_ref().is_some_and(|usage| usage.prompt_tokens == 100)));

    assert_eq!(
        store
            .load("early", false)
            .await
            .expect("inherited initial window"),
        entries[..1]
    );

    let mut fork_window = entries[2].clone();
    fork_window.id = "fork-compacted".to_string();
    fork_window.sequence = Some(5);
    assert_eq!(
        store
            .load("late", false)
            .await
            .expect("migrated fork compaction"),
        vec![fork_window]
    );
    let mut new = vec![system_entry("new prompt")];
    store
        .append("legacy", 5, &mut new)
        .await
        .expect("append new format at unchanged head");
    assert_eq!(new[0].sequence, Some(6));
    assert_eq!(
        store
            .load("legacy", false)
            .await
            .expect("indexed new window"),
        new
    );
    let count: i64 = store
        .connection
        .query_row(
            "SELECT COUNT(*) FROM react_entry WHERE system_prompt_hash IS NOT NULL",
            [],
            |row| row.get(0),
        )
        .expect("indexed windows");
    assert_eq!(count, 4);
    assert_eq!(
        store
            .connection
            .query_row("SELECT COUNT(*) FROM react_system_prompt", [], |row| row
                .get::<_, i64>(0))
            .expect("deduplicated prompts"),
        2
    );
    store
        .connection
        .execute(
            "UPDATE react_entry SET payload = X'FF' WHERE session_id = 'legacy' AND sequence = 3",
            [],
        )
        .expect("make archived entry unreadable");
    drop(store);
    let mut store = open(&database, "legacy", directory.path(), false);
    assert_eq!(
        store
            .load("legacy", false)
            .await
            .expect("reopen migrated database"),
        new
    );
}

fn legacy_system_entry(content: &str) -> AgentMessage {
    AgentMessage {
        id: "system".to_string(),
        payload: Some(agent_message::Payload::Chat(mirror_protocol::ChatRecord {
            message: Some(chat_record::Message::System(
                mirror_protocol::SystemMessage {
                    content: content.to_string(),
                },
            )),
            ..Default::default()
        })),
        ..Default::default()
    }
}

async fn assert_upgrade_without_metadata(
    prefix: &[AgentMessage],
    recent: Vec<AgentMessage>,
    expected_prompts: &[&str],
) {
    let directory = tempfile::tempdir().expect("directory");
    let database = directory.path().join("react.db");
    let connection = Connection::open(&database).expect("database");
    connection
        .execute_batch(include_str!("../../migrations/0001_initial.sql"))
        .expect("initial schema");
    connection
        .execute_batch(include_str!("../../migrations/0002_session_lineage.sql"))
        .expect("lineage schema");
    connection
        .pragma_update(None, "user_version", 2)
        .expect("old version");
    connection.execute(
        "INSERT INTO react_session(id, workspace, lineage, fork_points) VALUES ('legacy', ?, '[\"legacy\"]', '[9223372036854775807]')",
        [std::fs::canonicalize(directory.path()).expect("workspace").to_string_lossy().as_ref()],
    ).expect("session");
    for (index, entry) in prefix.iter().enumerate() {
        connection.execute(
            "INSERT INTO react_entry(session_id, sequence, entry_id, payload) VALUES ('legacy', ?, ?, ?)",
            params![index as i64 + 1, entry.id, entry.encode_to_vec()],
        ).expect("prefix entry");
    }
    let compaction_sequence = prefix.len() as u64 + 1;
    let mut compacted = AgentMessage {
        id: "compacted".to_string(),
        ..Default::default()
    }
    .encode_to_vec();
    compacted.extend(
        LegacyFields {
            metadata: None,
            compaction: Some(LegacyCompaction {
                summary: "saved summary".to_string(),
                recent,
                usage: Some(mirror_protocol::Usage {
                    prompt_tokens: 100,
                    ..Default::default()
                }),
            }),
        }
        .encode_to_vec(),
    );
    connection.execute(
        "INSERT INTO react_entry(session_id, sequence, entry_id, payload) VALUES ('legacy', ?, 'compacted', ?)",
        params![compaction_sequence as i64, compacted],
    ).expect("compaction entry");
    let mut after = identified("after", "next question");
    after.sequence = Some(compaction_sequence + 1);
    connection.execute(
        "INSERT INTO react_entry(session_id, sequence, entry_id, payload) VALUES ('legacy', ?, 'after', ?)",
        params![compaction_sequence as i64 + 1, after.encode_to_vec()],
    ).expect("following entry");
    drop(connection);

    // Opening another session must not be blocked by an older saved history.
    let mut store = open(&database, "unrelated", directory.path(), true);
    assert_eq!(
        store
            .connection
            .query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
            .expect("migrated version"),
        3
    );
    let entries = store.load("legacy", true).await.expect("full history");
    let prompts: Vec<_> = entries
        .iter()
        .filter_map(|entry| match &entry.payload {
            Some(agent_message::Payload::ContextWindow(window)) => {
                Some(window.system_prompt.as_str())
            }
            _ => None,
        })
        .collect();
    assert_eq!(prompts, expected_prompts);
    assert_eq!(entries.len(), prefix.len() + 2);
    let active = store.load("legacy", false).await.expect("active window");
    assert_eq!(active.len(), 2);
    assert_eq!(active[0].id, "compacted");
    assert_eq!(active[0].sequence, Some(compaction_sequence));
    assert_eq!(active[1], after);
    assert_matches!(&active[0].payload, Some(agent_message::Payload::ContextWindow(window)) if window.metadata.is_none() && window.compaction.as_ref().is_some_and(|compaction| compaction.summary.as_deref() == Some("saved summary") && compaction.usage.as_ref().is_some_and(|usage| usage.prompt_tokens == 100) && compaction.recent == vec![identified("retained", "retained question")]));
    drop(store);
    let mut reopened = open(&database, "legacy", directory.path(), false);
    assert_eq!(
        reopened
            .load("legacy", true)
            .await
            .expect("reopened history"),
        entries
    );
}

#[tokio::test]
async fn legacy_compaction_without_metadata_or_system_prompt_upgrades() {
    assert_upgrade_without_metadata(
        &[identified("before", "earlier question")],
        vec![identified("retained", "retained question")],
        &[""],
    )
    .await;
}

#[tokio::test]
async fn legacy_system_prompt_without_metadata_upgrades() {
    assert_upgrade_without_metadata(
        &[legacy_system_entry("stored prompt")],
        vec![identified("retained", "retained question")],
        &["stored prompt", "stored prompt"],
    )
    .await;
}

#[tokio::test]
async fn legacy_compaction_recovers_embedded_system_prompt_without_metadata() {
    assert_upgrade_without_metadata(
        &[identified("before", "earlier question")],
        vec![
            legacy_system_entry("embedded prompt"),
            identified("retained", "retained question"),
        ],
        &["embedded prompt"],
    )
    .await;
}

#[test]
fn rejects_workspace_mismatch() {
    let directory = tempfile::tempdir().expect("directory");
    let other_workspace = tempfile::tempdir().expect("other workspace");
    let database = directory.path().join("react.db");
    drop(open(&database, "session", directory.path(), true));

    assert_matches!(
        SQLiteStore::open(
            &database,
            "session",
            other_workspace.path(),
            false,
            false,
        ),
        Err(StoreError::Persistence(error)) if error.contains("belongs to workspace")
    );
    SQLiteStore::open(&database, "session", other_workspace.path(), false, true)
        .expect("allow mismatch");
}

#[tokio::test]
async fn fork_keeps_existing_parent_writers_valid() {
    let directory = tempfile::tempdir().expect("directory");
    let database = directory.path().join("react.db");
    let mut store = open(&database, "parent", directory.path(), true);
    store
        .append(
            "parent",
            0,
            &mut [identified("one", "one"), identified("two", "two")],
        )
        .await
        .expect("seed parent");
    SQLiteStore::fork(&database, "parent", "two", "child").expect("fork");
    store
        .append("parent", 2, &mut [identified("three", "three")])
        .await
        .expect("existing parent writer continues");
    assert_eq!(
        contents(&store.load("child", true).await.expect("fixed prefix")),
        [(1, "one")]
    );
    store.delete_session("child").expect("delete child");
    assert_eq!(
        contents(&store.load("parent", true).await.expect("parent survives")),
        [(1, "one"), (2, "two"), (3, "three")]
    );
}

#[test]
fn rejects_newer_agent_schema_version() {
    let directory = tempfile::tempdir().expect("directory");
    let database = directory.path().join("react.db");
    drop(SQLiteStore::open(&database, "session", directory.path(), true, false).expect("create"));
    let connection = Connection::open(&database).expect("open database");
    connection
        .pragma_update(None, "user_version", 4)
        .expect("set future version");
    drop(connection);

    assert_matches!(
        SQLiteStore::open(&database, "session", directory.path(), false, false),
        Err(StoreError::Persistence(error)) if error.contains("newer than this binary")
    );
}

#[tokio::test]
async fn malformed_lineage_does_not_create_a_child() {
    let directory = tempfile::tempdir().expect("directory");
    let database = directory.path().join("react.db");
    let mut store = open(&database, "parent", directory.path(), true);
    store
        .append("parent", 0, &mut [identified("user", "one")])
        .await
        .expect("seed");
    store
        .connection
        .execute(
            "UPDATE react_session SET fork_points = '[]' WHERE id = 'parent'",
            [],
        )
        .expect("simulate malformed persisted lineage");
    assert_matches!(
        SQLiteStore::fork(&database, "parent", "user", "child"),
        Err(StoreError::Persistence(_))
    );
    assert_matches!(
        store.load("child", true).await,
        Err(StoreError::SessionNotFound { .. })
    );
}

#[test]
fn a_second_open_of_a_conversation_names_the_holder() {
    let directory = tempfile::tempdir().expect("directory");
    let database = directory.path().join("react.db");
    let _held = open(&database, "session", directory.path(), true);

    let error = SQLiteStore::open(&database, "session", directory.path(), false, false)
        .expect_err("second open");

    assert_matches!(
        &error,
        StoreError::OpenElsewhere { session_id, holder }
            if session_id == "session" && *holder == std::process::id().to_string()
    );
    assert_eq!(
        error.to_string(),
        format!(
            "conversation \"session\" is open in another Mirror process (pid {})",
            std::process::id()
        )
    );
}

#[test]
fn dropping_a_store_releases_its_conversation() {
    let directory = tempfile::tempdir().expect("directory");
    let database = directory.path().join("react.db");
    drop(open(&database, "session", directory.path(), true));

    open(&database, "session", directory.path(), false);
}

#[test]
fn dropping_a_store_releases_its_conversation_despite_a_shared_descriptor() {
    let directory = tempfile::tempdir().expect("directory");
    let database = directory.path().join("react.db");
    let store = open(&database, "session", directory.path(), true);
    // A subprocess being spawned holds a copy of every descriptor until it execs.
    let spawning_copy = store
        ._lock
        .as_ref()
        .expect("conversation lock")
        .0
        .try_clone()
        .expect("shared lock descriptor");
    drop(store);

    open(&database, "session", directory.path(), false);
    drop(spawning_copy);
}

#[test]
fn conversations_open_independently() {
    let directory = tempfile::tempdir().expect("directory");
    let database = directory.path().join("react.db");
    let _first = open(&database, "first", directory.path(), true);

    open(&database, "second", directory.path(), true);
}

#[test]
fn an_alias_of_the_database_shares_its_conversation_locks() {
    let directory = tempfile::tempdir().expect("directory");
    let database = directory.path().join("react.db");
    let _held = open(&database, "session", directory.path(), true);
    let alias = directory.path().join("alias.db");
    std::os::unix::fs::symlink(&database, &alias).expect("alias database");

    assert_matches!(
        SQLiteStore::open(&alias, "session", directory.path(), false, false),
        Err(StoreError::OpenElsewhere { .. })
    );
}

#[test]
fn an_in_memory_database_takes_no_lock() {
    let directory = tempfile::tempdir().expect("directory");
    let _first = open(Path::new(":memory:"), "session", directory.path(), true);

    open(Path::new(":memory:"), "session", directory.path(), true);
}
