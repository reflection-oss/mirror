use assert_matches::assert_matches;
use mirror_protocol::AgentMessage;
use pretty_assertions::assert_eq;

use super::*;

fn entry(id: &str) -> AgentMessage {
    AgentMessage {
        id: id.to_string(),
        ..AgentMessage::default()
    }
}

#[tokio::test]
async fn forks_share_lineage_and_isolate_suffixes() {
    let mut store = InMemoryStore::default();
    let mut history = vec![
        AgentMessage {
            payload: Some(mirror_protocol::agent_message::Payload::ContextWindow(
                mirror_protocol::ContextWindowRecord {
                    system_prompt: "frozen".to_string(),
                    ..Default::default()
                },
            )),
            ..entry("window")
        },
        AgentMessage {
            payload: Some(mirror_protocol::agent_message::Payload::Chat(
                mirror_protocol::ChatRecord {
                    message: Some(mirror_protocol::chat_record::Message::User(
                        mirror_protocol::UserMessage {
                            checkpoint: None,
                            content: "hello".to_string(),
                        },
                    )),
                    ..Default::default()
                },
            )),
            ..entry("user")
        },
        entry("answer"),
    ];
    store
        .append("parent", 0, &mut history)
        .await
        .expect("parent");
    store.fork("parent", None, "head").expect("fork at head");
    assert!(store.sessions["head"].is_empty());
    assert_eq!(store.load("head", true).await.expect("inherited"), history);
    assert_matches!(
        store.append("head", 0, &mut [entry("stale")]).await,
        Err(StoreError::Stale)
    );
    assert_matches!(
        store.append("head", 3, &mut [entry("user")]).await,
        Err(StoreError::MessageIdConflict { .. })
    );
    let mut child_suffix = vec![entry("child-only")];
    store
        .append("head", 3, &mut child_suffix)
        .await
        .expect("child append");
    store
        .append("parent", 3, &mut [entry("parent-only")])
        .await
        .expect("parent append");
    assert_eq!(
        store.load("head", true).await.expect("child"),
        [history.clone(), child_suffix.clone()].concat()
    );
    assert!(!store
        .load("parent", true)
        .await
        .expect("parent")
        .iter()
        .any(|entry| entry.id == "child-only"));
    store
        .fork("head", Some("user"), "rewound")
        .expect("nested rewind into ancestor");
    assert_eq!(
        store.load("rewound", true).await.expect("rewound"),
        history[..1]
    );
    let mut replacement = vec![history[1].clone()];
    store
        .append("rewound", 1, &mut replacement)
        .await
        .expect("reuse abandoned suffix id");
    assert_eq!(replacement[0], history[1]);
    store
        .fork("head", None, "grandchild")
        .expect("nested head fork");
    store
        .append("head", 4, &mut [entry("later")])
        .await
        .expect("advance child");
    assert_eq!(
        store.load("grandchild", false).await.expect("grandchild"),
        [history, child_suffix].concat()
    );
}

#[tokio::test]
async fn forks_validate_boundaries_and_support_empty_or_bounded_snapshots() {
    let mut store = InMemoryStore::default();
    assert_matches!(
        store.fork("missing", None, "child"),
        Err(StoreError::SessionNotFound { .. })
    );
    store.load("empty", true).await.expect("empty");
    store
        .fork("empty", None, "empty-child")
        .expect("empty fork");
    assert!(store
        .load("empty-child", true)
        .await
        .expect("empty child")
        .is_empty());
    let snapshot = vec![AgentMessage {
        sequence: Some(42),
        ..entry("tail")
    }];
    let mut store = InMemoryStore::from_entries("parent".to_string(), snapshot.clone());
    assert_matches!(
        store.fork("parent", Some("missing"), "bad"),
        Err(StoreError::EntryNotFound { .. })
    );
    assert_matches!(
        store.fork("parent", Some("tail"), "bad"),
        Err(StoreError::InvalidForkTarget { .. })
    );
    store.fork("parent", None, "child").expect("snapshot fork");
    assert!(store.fork("parent", None, "child").is_err());
    assert!(store.fork("child", None, "parent").is_err());
    assert_eq!(
        store.load("child", true).await.expect("preserved snapshot"),
        snapshot
    );
    let mut appended = vec![entry("new")];
    store
        .append("child", 42, &mut appended)
        .await
        .expect("append after original head");
    assert_eq!(appended[0].sequence, Some(43));
    assert!(ForkPoint::after(i64::MAX as u64).is_none());
}

#[tokio::test]
async fn appends_are_a_compare_and_swap_on_the_head() {
    let mut store = InMemoryStore::default();
    assert!(store.load("session", true).await.expect("load").is_empty());
    assert_matches!(
        store
            .append("session", 0, &mut [entry("duplicate"), entry("duplicate")])
            .await,
        Err(StoreError::MessageIdConflict { .. })
    );

    let mut committed = vec![AgentMessage::default()];
    store
        .append("session", 0, &mut committed)
        .await
        .expect("append");

    assert_eq!(committed[0].sequence, Some(1));
    assert_matches!(
        store
            .append("session", 0, &mut [AgentMessage::default()])
            .await,
        Err(StoreError::Stale)
    );
    assert_matches!(
        store
            .append("session", 1, &mut [entry(&committed[0].id)])
            .await,
        Err(StoreError::MessageIdConflict { .. })
    );
    assert_eq!(
        store.load("session", true).await.expect("reload"),
        committed
    );
}
