use std::os::unix::fs::PermissionsExt;

use assert_matches::assert_matches;
use pretty_assertions::assert_eq;

use super::*;
use crate::memory::create_record;
use crate::memory::MemoryDraft;

const REPO: &str = "github.com/example/project";

fn draft(key: &str, content: &str) -> MemoryDraft {
    MemoryDraft {
        r#type: MemoryType::Pitfall,
        key: key.into(),
        repository: REPO.into(),
        path_globs: vec!["harness-rs/src/**/*.rs".into()],
        situation: "editing the ReactAgent loop".into(),
        keywords: vec!["ReactAgentConfig".into(), "compaction".into()],
        content: content.into(),
        source: String::new(),
    }
}

fn search(text: &str, exclude: &[String], limit: usize) -> MemoryQuery {
    MemoryQuery {
        repository: Some(REPO.into()),
        text: Some(text.into()),
        exclude: exclude.to_vec(),
        limit,
        ..MemoryQuery::default()
    }
}

fn commit(
    actor: MemoryActor,
    session_id: Option<&str>,
    changes: Vec<MemoryChange>,
) -> MemoryCommit {
    MemoryCommit {
        actor,
        session_id: session_id.map(str::to_string),
        changes,
    }
}

fn insert(database: &mut SqliteMemoryDatabase, record: &MemoryRecord) {
    database
        .commit(&commit(
            record.actor,
            None,
            vec![MemoryChange::Insert(record.clone())],
        ))
        .expect("insert");
}

fn ids(records: &[MemoryRecord]) -> Vec<&str> {
    records.iter().map(|record| record.id.as_str()).collect()
}

fn open() -> (tempfile::TempDir, SqliteMemoryDatabase) {
    let directory = tempfile::tempdir().expect("directory");
    let root = directory.path().join("profiles").join("personal");
    let database = SqliteMemoryDatabase::open(&root).expect("database");
    (directory, database)
}

#[test]
fn creates_a_private_store() {
    let (directory, _repository) = open();
    let root = directory.path().join("profiles").join("personal");
    let directory_mode = std::fs::metadata(&root).expect("dir").permissions().mode() & 0o777;
    let file_mode = std::fs::metadata(root.join(MEMORY_DATABASE))
        .expect("db")
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(directory_mode, 0o700);
    assert_eq!(file_mode, 0o600);
}

#[test]
fn insert_indexes_the_record_and_logs_an_accept_event() {
    let (_directory, mut database) = open();
    let record = create_record(
        draft(
            "compaction-rebuild",
            "The compactor is rebuilt for every prompt swap.",
        ),
        MemoryActor::Extractor,
        1_000,
    )
    .expect("record");
    database
        .commit(&commit(
            MemoryActor::User,
            Some("session-1"),
            vec![MemoryChange::Insert(record.clone())],
        ))
        .expect("insert");

    let stored = database.get(&record.id).expect("get").expect("stored");
    assert_eq!(
        stored.actor,
        MemoryActor::User,
        "the commit's actor wrote it"
    );
    assert!(stored.updated_at >= record.updated_at);
    assert_eq!(
        MemoryRecord {
            actor: record.actor,
            updated_at: record.updated_at,
            ..stored.clone()
        },
        record
    );
    // camelCase in keywords is split at index time, so a plain query term finds it.
    let found = database
        .search(&search("compactor agent", &[], 5))
        .expect("search");
    assert_eq!(ids(&found), vec![record.id.as_str()]);
    let events = database.events(&record.id).expect("events");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].op, "accept");
    assert_eq!(events[0].actor, "user");
    assert_eq!(events[0].session_id.as_deref(), Some("session-1"));
    assert_eq!(events[0].version, 1);
}

#[test]
fn a_stale_version_changes_nothing_and_a_retired_record_leaves_the_index() {
    let (_directory, mut database) = open();
    let record = create_record(
        draft("stale-check", "The lookup table is a hint; Rust re-checks."),
        MemoryActor::User,
        1_000,
    )
    .expect("record");
    insert(&mut database, &record);

    let mut skipped = record.clone();
    skipped.version = 3;
    assert_matches!(
        database.commit(&commit(
            MemoryActor::User,
            None,
            vec![MemoryChange::Update {
                record: skipped,
                operation: MemoryEventOp::Correct,
            }]
        )),
        Err(MemoryError::Stale)
    );
    assert_eq!(
        database
            .get(&record.id)
            .expect("get")
            .map(|stored| stored.version),
        Some(1)
    );

    let mut retired = record.clone();
    retired.version = 2;
    retired.status = MemoryStatus::Retired;
    database
        .commit(&commit(
            MemoryActor::Consolidator,
            None,
            vec![MemoryChange::Update {
                record: retired,
                operation: MemoryEventOp::Retire,
            }],
        ))
        .expect("retire");
    let head = database.get(&record.id).expect("get").expect("head");
    assert_eq!(head.status, MemoryStatus::Retired);
    assert_eq!(head.actor, MemoryActor::Consolidator);
    assert!(database
        .search(&search("lookup", &[], 5))
        .expect("search")
        .is_empty());
    let events = database.events(&record.id).expect("events");
    assert_eq!(
        events
            .iter()
            .map(|event| (event.op.as_str(), event.version))
            .collect::<Vec<_>>(),
        vec![("accept", 1), ("retire", 2)]
    );
}

#[test]
fn an_update_must_describe_the_status_change() {
    let (_directory, mut database) = open();
    let record = create_record(
        draft("describe", "An observation."),
        MemoryActor::User,
        1_000,
    )
    .expect("record");
    insert(&mut database, &record);
    let mut retired = record.clone();
    retired.version = 2;
    retired.status = MemoryStatus::Retired;
    assert_matches!(
        database.commit(&commit(
            MemoryActor::User,
            None,
            vec![MemoryChange::Update {
                record: retired,
                operation: MemoryEventOp::Correct,
            }]
        )),
        Err(MemoryError::Invalid(_))
    );
}

#[test]
fn reopening_keeps_records() {
    let (directory, mut database) = open();
    let root = directory.path().join("profiles").join("personal");
    let record = create_record(
        draft("survives-reopen", "Records outlive the process."),
        MemoryActor::User,
        1_000,
    )
    .expect("record");
    insert(&mut database, &record);
    drop(database);

    let reopened = SqliteMemoryDatabase::open(&root).expect("reopen");
    assert_eq!(
        ids(&reopened.search(&MemoryQuery::default()).expect("scan")),
        Vec::<&str>::new(),
        "outside any repository only repository-agnostic records qualify"
    );
    let scoped = MemoryQuery {
        repository: Some(REPO.into()),
        ..MemoryQuery::default()
    };
    assert_eq!(
        ids(&reopened.search(&scoped).expect("scan")),
        vec![record.id.as_str()]
    );
}

#[test]
fn rejects_invalid_records_before_touching_storage() {
    let (_directory, mut database) = open();
    let mut record = create_record(
        draft("valid-key", "A valid observation."),
        MemoryActor::User,
        1_000,
    )
    .expect("record");
    record.content.clear();
    assert_matches!(
        database.commit(&commit(
            MemoryActor::User,
            None,
            vec![MemoryChange::Insert(record)]
        )),
        Err(MemoryError::Invalid(_))
    );
    let scoped = MemoryQuery {
        repository: Some(REPO.into()),
        ..MemoryQuery::default()
    };
    assert!(database.search(&scoped).expect("scan").is_empty());
}

#[test]
fn a_commit_applies_all_changes_or_none() {
    let (_directory, mut database) = open();
    let first = create_record(draft("first", "One observation."), MemoryActor::User, 1_000)
        .expect("record");
    let mut missing = first.clone();
    missing.id = uuid::Uuid::new_v4().to_string();
    missing.version = 2;
    assert_matches!(
        database.commit(&commit(
            MemoryActor::User,
            None,
            vec![
                MemoryChange::Insert(first.clone()),
                MemoryChange::Update {
                    record: missing,
                    operation: MemoryEventOp::Correct,
                },
            ]
        )),
        Err(MemoryError::Invalid(_))
    );
    assert_matches!(database.get(&first.id), Ok(None));
}

/// Unrelated records that make a searched term distinctive enough to clear
/// the relevance floor, which is relative to the store's size.
fn insert_fillers(database: &mut SqliteMemoryDatabase, count: usize) {
    for index in 0..count {
        let mut filler = draft(
            &format!("filler-{index}"),
            &format!("Unrelated note {index} about bazel targets and gazelle."),
        );
        filler.keywords = vec!["bazel".into()];
        let record = create_record(filler, MemoryActor::User, 1_000).expect("record");
        insert(database, &record);
    }
}

#[test]
fn search_applies_scope_and_exclusions_before_the_limit() {
    let (_directory, mut database) = open();
    insert_fillers(&mut database, 12);
    let mut ids = Vec::new();
    for index in 0..4 {
        let mut draft = draft(&format!("lookup-{index}"), "The lookup table is a hint.");
        draft.keywords = vec!["lookup".into()];
        if index != 3 {
            draft.repository = "github.com/example/other".into();
        }
        let record = create_record(draft, MemoryActor::User, 1_000).expect("record");
        ids.push(record.id.clone());
        insert(&mut database, &record);
    }
    let found = database.search(&search("lookup", &[], 1)).expect("search");
    assert_eq!(found.len(), 1);
    assert_eq!(
        found[0].id, ids[3],
        "the other repository never reaches the limit"
    );
    let excluded = database
        .search(&search("lookup", &[ids[3].clone()], 5))
        .expect("search");
    assert!(excluded.is_empty());
}

#[test]
fn match_query_quotes_distinct_lowercased_terms() {
    assert_eq!(
        fts_match_query("Fix ReactAgentConfig's compaction, compaction!"),
        Some("\"fix\" OR \"react\" OR \"agent\" OR \"config\" OR \"reactagentconfig\" OR \"compaction\"".to_string())
    );
    assert_matches!(fts_match_query("k"), None);
}

#[test]
fn search_stems_query_terms_and_folds_compatibility_characters() {
    let (_directory, mut database) = open();
    let widen = create_record(
        draft(
            "bazel-visibility",
            "Widen the target visibility instead of threading a primitive.",
        ),
        MemoryActor::User,
        1_000,
    )
    .expect("record");
    let ligature = create_record(
        draft("ligature-lookup", "Visit the oﬃce before editing."),
        MemoryActor::User,
        1_000,
    )
    .expect("record");
    insert(&mut database, &widen);
    insert(&mut database, &ligature);

    // Porter stemming: the query term "widened" reaches the stored "widen".
    let found = database.search(&search("widened", &[], 1)).expect("search");
    assert_eq!(ids(&found), vec![widen.id.as_str()]);
    for query in ["office", "oﬃce"] {
        let found = database.search(&search(query, &[], 5)).expect("search");
        assert_eq!(ids(&found), vec![ligature.id.as_str()]);
    }
}

#[test]
fn distinct_content_search_looks_past_duplicate_top_hits() {
    let (_directory, mut database) = open();
    insert_fillers(&mut database, 20);
    let needle = |key: &str, content: &str| {
        let mut draft = draft(key, content);
        draft.situation = "finding the needle".into();
        draft.keywords = vec!["needle".into()];
        create_record(draft, MemoryActor::User, 1_000).expect("record")
    };
    for index in 0..5 {
        insert(
            &mut database,
            &needle(&format!("duplicate-{index}"), "needle"),
        );
    }
    let unique = needle("unique", "needle with distinct guidance and more text");
    insert(&mut database, &unique);

    let query = MemoryQuery {
        distinct_content: true,
        ..search("needle", &[], 2)
    };
    let found = database.search(&query).expect("search");
    assert!(
        found.len() <= 4,
        "at most two variants per distinct content"
    );
    assert!(found.iter().any(|record| record.id != unique.id));
    assert!(found.iter().any(|record| record.id == unique.id));
}

#[test]
fn a_scan_returns_scoped_records_newest_first_and_can_keep_only_glob_candidates() {
    let (_directory, mut database) = open();
    let older = create_record(draft("older", "Older."), MemoryActor::User, 1_000).expect("record");
    let mut plain = draft("newer-plain", "Newer, no globs.");
    plain.path_globs.clear();
    let newer = create_record(plain, MemoryActor::User, 2_000).expect("record");
    insert(&mut database, &older);
    insert(&mut database, &newer);

    let scoped = MemoryQuery {
        repository: Some(REPO.into()),
        ..MemoryQuery::default()
    };
    assert_eq!(
        ids(&database.search(&scoped).expect("scan")),
        vec![newer.id.as_str(), older.id.as_str()]
    );
    let globbed = MemoryQuery {
        with_globs_only: true,
        ..scoped
    };
    assert_eq!(
        ids(&database.search(&globbed).expect("scan")),
        vec![older.id.as_str()]
    );
}

#[test]
fn marking_a_pass_records_when_it_ran_and_the_sequence_it_read() {
    let (_directory, mut database) = open();
    let record = create_record(draft("k", "content"), MemoryActor::User, 1_000).expect("record");
    insert(&mut database, &record);
    database.mark_consolidated(5_000).expect("mark");
    let state = database.maintenance().expect("state");
    assert_eq!(state.last_consolidated_at, 5_000);
    assert_eq!(state.last_event_seq, 1);
}
