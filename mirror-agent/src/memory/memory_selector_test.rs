use std::collections::HashSet;

use assert_matches::assert_matches;
use pretty_assertions::assert_eq;

use super::*;
use crate::memory::create_record;
use crate::memory::MemoryActor;
use crate::memory::MemoryChange;
use crate::memory::MemoryCommit;
use crate::memory::MemoryDraft;
use crate::memory::MemoryType;
use crate::memory::SqliteMemoryDatabase;
use crate::memory::SqliteMemoryStore;

const REPO: &str = "github.com/example/project";

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

fn scope() -> MemoryQuery {
    MemoryQuery {
        repository: Some(REPO.into()),
        ..MemoryQuery::default()
    }
}

fn request<'a>(
    message: Option<&'a str>,
    new_paths: &'a [String],
    surfaced: &'a HashSet<String>,
) -> SelectionRequest<'a> {
    SelectionRequest {
        message,
        new_paths,
        surfaced,
        surfaced_chars: 0,
        now: 1_000,
    }
}

fn keys(selected: &[Selected]) -> Vec<&str> {
    selected
        .iter()
        .map(|selected| selected.record.key.as_str())
        .collect()
}

#[test]
fn queries_follow_the_gate_and_the_paths_in_context() {
    let none = HashSet::new();
    let paths = vec!["src/react/agent.rs".to_string()];
    assert_matches!(
        lexical_query(&scope(), &request(Some("ok"), &[], &none)),
        None
    );
    assert_matches!(lexical_query(&scope(), &request(None, &paths, &none)), None);
    let lexical = lexical_query(
        &scope(),
        &request(Some("why does compaction rebuild"), &[], &none),
    )
    .expect("a real message is searched");
    assert_eq!(lexical.text.as_deref(), Some("why does compaction rebuild"));
    assert!(lexical.distinct_content);
    assert_matches!(glob_query(&scope(), &request(None, &[], &none)), None);
    assert!(
        glob_query(&scope(), &request(None, &paths, &none))
            .expect("paths entered context")
            .with_globs_only
    );
    let spent = SelectionRequest {
        surfaced_chars: 61_440,
        ..request(Some("why does compaction rebuild"), &[], &none)
    };
    assert_matches!(lexical_query(&scope(), &spent), None);
}

#[test]
fn glob_channel_attaches_the_most_specific_match_once_and_skips_surfaced_records() {
    let broad = record("broad", "Anything under src.", &["src/**"]);
    let specific = record("specific", "The react loop.", &["src/react/*.rs"]);
    let shown = record("shown", "Already shown.", &["src/react/**"]);
    let surfaced = HashSet::from([shown.id.clone()]);
    let paths = vec!["src/react/agent.rs".to_string()];

    let selected = select(
        &[],
        &[broad.clone(), specific.clone(), shown],
        &request(None, &paths, &surfaced),
    );

    assert_eq!(keys(&selected), vec!["specific", "broad"]);
    assert_eq!(selected[0].channel, Channel::Glob);
    assert_eq!(
        selected[0].matched_path.as_deref(),
        Some("src/react/agent.rs")
    );
    assert!(selected[0]
        .rendered
        .contains("attached: matched src/react/agent.rs"));
}

#[test]
fn lexical_hits_keep_the_store_order_capped_at_five_distinct_contents() {
    let hits: Vec<MemoryRecord> = (0..8)
        .map(|index| {
            let content = if index == 1 {
                "First observation.".to_string()
            } else {
                format!("Observation {index}.")
            };
            record(&format!("hit-{index}"), &content, &[])
        })
        .collect();
    let none = HashSet::new();

    let selected = select(&hits, &[], &request(Some("observation please"), &[], &none));

    // hit-0 and hit-1 share content once normalized? They do not, so five distinct records are taken in order.
    assert_eq!(
        keys(&selected),
        vec!["hit-0", "hit-1", "hit-2", "hit-3", "hit-4"]
    );
    assert!(selected
        .iter()
        .all(|selected| selected.channel == Channel::Lexical));
}

#[test]
fn a_record_admitted_by_both_channels_is_reported_once() {
    let both = record("both", "Attached and retrieved.", &["src/**"]);
    let only_lexical = record("lexical", "Retrieved only.", &[]);
    let paths = vec!["src/lib.rs".to_string()];
    let none = HashSet::new();

    let selected = select(
        &[only_lexical.clone(), both.clone()],
        std::slice::from_ref(&both),
        &request(Some("retrieve everything"), &paths, &none),
    );

    assert_eq!(keys(&selected), vec!["both", "lexical"]);
    assert_eq!(selected[0].channel, Channel::GlobLexical);
    assert_eq!(selected[1].channel, Channel::Lexical);
}

#[test]
fn duplicate_content_is_shown_once() {
    let first = record("first", "Same words.", &[]);
    let duplicate = record("duplicate", "same words.", &[]);
    let none = HashSet::new();

    let selected = select(
        &[first, duplicate],
        &[],
        &request(Some("same words please"), &[], &none),
    );

    assert_eq!(keys(&selected), vec!["first"]);
}

#[test]
fn records_that_do_not_fit_the_recall_budget_are_skipped_not_truncated() {
    // Each renders to just over 4 KiB (the body cap); the 4,096-token recall
    // holds three of them.
    let hits: Vec<MemoryRecord> = (0..5)
        .map(|index| {
            record(
                &format!("large-{index}"),
                &format!("{index}").repeat(4_000),
                &[],
            )
        })
        .collect();
    let none = HashSet::new();

    let selected = select(
        &hits,
        &[],
        &request(Some("large records please"), &[], &none),
    );

    assert_eq!(keys(&selected), vec!["large-0", "large-1", "large-2"]);
}

#[tokio::test]
async fn select_from_reads_each_enabled_channel_from_the_store() {
    let directory = tempfile::tempdir().expect("directory");
    let store =
        SqliteMemoryStore::new(SqliteMemoryDatabase::open(directory.path()).expect("database"))
            .expect("store");
    let attached = record(
        "attached",
        "Mind the agent loop's borrow rules.",
        &["src/react/**"],
    );
    let retrieved = record(
        "retrieved",
        "Compaction rebuilds the compactor for every prompt swap.",
        &[],
    );
    let fillers: Vec<MemoryRecord> = (0..8)
        .map(|index| {
            record(
                &format!("filler-{index}"),
                &format!("Unrelated note {index} about gazelle."),
                &[],
            )
        })
        .collect();
    let mut changes: Vec<MemoryChange> = fillers.into_iter().map(MemoryChange::Insert).collect();
    changes.push(MemoryChange::Insert(attached.clone()));
    changes.push(MemoryChange::Insert(retrieved.clone()));
    store
        .commit(MemoryCommit {
            actor: MemoryActor::User,
            session_id: None,
            changes,
        })
        .await
        .expect("seed");
    let paths = vec!["src/react/agent.rs".to_string()];
    let none = HashSet::new();

    let selected = select_from(
        &store,
        &scope(),
        &request(
            Some("why does compaction rebuild the compactor"),
            &paths,
            &none,
        ),
    )
    .await
    .expect("select");

    assert_eq!(keys(&selected), vec!["attached", "retrieved"]);
    assert_eq!(selected[0].channel, Channel::Glob);
    assert_eq!(selected[1].channel, Channel::Lexical);

    let quiet = select_from(&store, &scope(), &request(Some("ok"), &[], &none))
        .await
        .expect("select");
    assert!(
        quiet.is_empty(),
        "a trivial message with no new paths reads nothing"
    );
}
