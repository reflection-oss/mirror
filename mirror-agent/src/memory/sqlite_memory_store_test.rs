use assert_matches::assert_matches;
use pretty_assertions::assert_eq;

use super::*;
use crate::memory::create_record;
use crate::memory::MemoryActor;
use crate::memory::MemoryChange;
use crate::memory::MemoryDraft;
use crate::memory::MemoryEventOp;
use crate::memory::MemoryStatus;
use crate::memory::MemoryType;

#[tokio::test]
async fn commits_are_atomic_and_visible_to_every_connection() {
    let directory = tempfile::tempdir().expect("directory");
    let database = SqliteMemoryDatabase::open(directory.path()).expect("database");
    let store = SqliteMemoryStore::new(database).expect("store");
    // Another process owning its own connection to the same store.
    let second =
        SqliteMemoryStore::new(SqliteMemoryDatabase::open(directory.path()).expect("database"))
            .expect("store");
    let query = MemoryQuery {
        text: Some("bazelisk".into()),
        limit: 5,
        ..MemoryQuery::default()
    };
    let record = create_record(
        MemoryDraft {
            r#type: MemoryType::Procedure,
            key: "bazelisk".into(),
            content: "Use bazelisk to build.".into(),
            situation: "building".into(),
            ..Default::default()
        },
        MemoryActor::User,
        1,
    )
    .expect("record");
    store
        .commit(MemoryCommit {
            actor: MemoryActor::User,
            session_id: None,
            changes: vec![MemoryChange::Insert(record.clone())],
        })
        .await
        .expect("commit");
    let found = second
        .search(query.clone())
        .await
        .expect("search from other connection");
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].id, record.id);
    assert_eq!(
        second.get(&record.id).await.expect("get"),
        Some(found[0].clone())
    );

    let mut retired = record.clone();
    retired.version += 1;
    retired.status = MemoryStatus::Retired;
    let retirement = MemoryChange::Update {
        record: retired,
        operation: MemoryEventOp::Retire,
    };
    // A valid retirement followed by a duplicate insert must roll back both.
    assert_matches!(
        store
            .commit(MemoryCommit {
                actor: MemoryActor::User,
                session_id: None,
                changes: vec![retirement.clone(), MemoryChange::Insert(record.clone())],
            })
            .await,
        Err(MemoryError::Invalid(_))
    );
    assert_eq!(
        second
            .get(&record.id)
            .await
            .expect("get")
            .map(|stored| stored.version),
        Some(1)
    );
    store
        .commit(MemoryCommit {
            actor: MemoryActor::User,
            session_id: None,
            changes: vec![retirement.clone()],
        })
        .await
        .expect("retire");
    assert_matches!(
        second
            .commit(MemoryCommit {
                actor: MemoryActor::User,
                session_id: None,
                changes: vec![retirement],
            })
            .await,
        Err(MemoryError::Stale),
        "the same change against the same version is stale the second time"
    );
    assert!(second.search(query).await.expect("search").is_empty());
}
