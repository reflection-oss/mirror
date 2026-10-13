use std::path::Path;

use pretty_assertions::assert_eq;

use super::*;
use crate::memory::MemoryEventOp;

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().expect("parent")).expect("dir");
    std::fs::write(path, text).expect("write");
}

fn claude_fixture(root: &Path) -> PathBuf {
    let memory = root.join("memory");
    write(
        &memory.join("MEMORY.md"),
        "# Memory Index\n- [Ask](ask-before-defaults.md) — rule\n",
    );
    write(
        &memory.join("ask-before-defaults.md"),
        "---\nname: ask-before-defaults\ndescription: \"Never set defaults alone: list options and ask\"\nmetadata:\n  type: feedback\n---\n\nNever set defaults by yourself, always ask. Related: [[plan-before-acting]].\n",
    );
    write(
        &memory.join("codebase-map").join("engine.md"),
        "---\nname: engine\ndescription: where the engine lives\nmetadata:\n  type: reference\n---\n\nThe engine is under services/engine.\n",
    );
    write(
        &memory.join("old").join("superseded.md"),
        "---\nname: superseded\n---\n\nDo not load me.\n",
    );
    write(&memory.join("plain.md"), "A note without frontmatter.\n");
    memory
}

fn codex_fixture(root: &Path) -> PathBuf {
    let memories = root.join("memories");
    write(
        &memories.join("memory_summary.md"),
        "v1\n\n## User Profile\n\nWorks on Mirror.\n\n## User preferences\n\n- Never send email without authorization.\n\n## General Tips\n\n- Use progressive disclosure.\n",
    );
    write(
        &memories.join("MEMORY.md"),
        "# Task Group: Mirror\n\n## Task 1: Review Mirror memory\n\n## Reusable knowledge\n\n- Mirror's baseline is preferences.md,\n  plus SQLite state. [Task 1]\n\n## Failures and how to do differently\n\n- Inspect the code before describing gaps. [Task 1]\n",
    );
    memories
}

fn active(database: &mut SqliteMemoryDatabase) -> Vec<MemoryRecord> {
    database.search(&MemoryQuery::default()).expect("scan")
}

#[test]
fn claude_code_notes_map_types_keywords_and_skip_the_index_and_old() {
    let temp = tempfile::tempdir().expect("temp");
    let notes = claude_code_notes::claude_code_notes(&claude_fixture(temp.path())).expect("notes");
    let keys: Vec<&str> = notes.iter().map(|note| note.key.as_str()).collect();
    assert_eq!(keys, vec!["ask-before-defaults", "engine", "plain"]);
    assert_eq!(notes[0].kind, MemoryType::Preference);
    assert_eq!(
        notes[0].situation,
        "Never set defaults alone: list options and ask"
    );
    assert_eq!(notes[0].keywords, vec!["plan-before-acting".to_string()]);
    assert!(
        notes[0].observed_at.is_some(),
        "the file's modification time"
    );
    assert_eq!(notes[1].kind, MemoryType::Reference);
    assert_eq!(notes[1].file, "codebase-map/engine.md");
    assert_eq!(notes[2].content, "A note without frontmatter.");
}

#[test]
fn codex_notes_come_from_the_summary_and_the_task_memory() {
    let temp = tempfile::tempdir().expect("temp");
    let notes = codex_notes::codex_notes(&codex_fixture(temp.path())).expect("notes");
    let kinds: Vec<(MemoryType, &str)> = notes
        .iter()
        .map(|note| (note.kind, note.situation.as_str()))
        .collect();
    assert_eq!(
        kinds,
        vec![
            (MemoryType::Fact, "who the user is, as Codex summarized it"),
            (
                MemoryType::Preference,
                "how the user wants work done, as Codex summarized it"
            ),
            (
                MemoryType::Procedure,
                "working practice Codex learned across tasks"
            ),
            (MemoryType::Fact, "Codex task: Review Mirror memory"),
            (MemoryType::Pitfall, "Codex task: Review Mirror memory"),
        ]
    );
    assert_eq!(
        notes[3].content,
        "Mirror's baseline is preferences.md, plus SQLite state."
    );
}

#[test]
fn imports_are_idempotent_follow_the_source_and_never_undo_a_forget() {
    let temp = tempfile::tempdir().expect("temp");
    let memory = claude_fixture(temp.path());
    let mut database = SqliteMemoryDatabase::open(&temp.path().join("store")).expect("store");
    let sources = vec![ImportSource::ClaudeCode {
        directory: memory.clone(),
        repository: None,
    }];

    let first = MemoryImport::new(&mut database)
        .run(&sources)
        .expect("first run");
    assert_eq!((first.saved, first.retired), (3, 0));
    let records = active(&mut database);
    assert_eq!(records.len(), 3);
    assert!(records
        .iter()
        .all(|record| record.actor == MemoryActor::Import));
    assert!(records
        .iter()
        .all(|record| record.source.starts_with("Claude Code [")));

    let again = MemoryImport::new(&mut database)
        .run(&sources)
        .expect("second run");
    assert_eq!((again.saved, again.unchanged, again.retired), (0, 3, 0));

    // The user forgets one; the next import leaves it forgotten.
    let plain = records
        .iter()
        .find(|record| record.key.starts_with("plain-"))
        .expect("plain note")
        .clone();
    database
        .commit(&MemoryCommit {
            actor: MemoryActor::User,
            session_id: None,
            changes: vec![MemoryChange::retire(plain.clone(), MemoryEventOp::Forget)],
        })
        .expect("forget");
    let after_forget = MemoryImport::new(&mut database)
        .run(&sources)
        .expect("third run");
    assert_eq!((after_forget.saved, after_forget.retired), (0, 0));
    assert_eq!(active(&mut database).len(), 2);

    // A deleted file retires its record.
    std::fs::remove_file(memory.join("codebase-map").join("engine.md")).expect("delete");
    let after_delete = MemoryImport::new(&mut database)
        .run(&sources)
        .expect("fourth run");
    assert_eq!(after_delete.retired, 1);
    let keys: Vec<String> = active(&mut database)
        .into_iter()
        .map(|record| record.key)
        .collect();
    assert_eq!(keys.len(), 1);
    assert!(keys[0].starts_with("ask-before-defaults-"));
}

#[test]
fn a_key_another_writer_holds_is_refused() {
    let temp = tempfile::tempdir().expect("temp");
    let memories = codex_fixture(temp.path());
    let mut database = SqliteMemoryDatabase::open(&temp.path().join("store")).expect("store");
    let sources = vec![ImportSource::Codex {
        directory: memories,
        repository: None,
    }];
    let notes = sources[0].read().expect("notes");
    let taken = note_key(&sources[0], &notes[0]);
    database
        .commit(&MemoryCommit {
            actor: MemoryActor::User,
            session_id: None,
            changes: vec![MemoryChange::Insert(
                create_record(
                    MemoryDraft {
                        r#type: MemoryType::Fact,
                        key: taken.clone(),
                        situation: "who I am".into(),
                        content: "The user wrote this one.".into(),
                        ..MemoryDraft::default()
                    },
                    MemoryActor::User,
                    1_000,
                )
                .expect("user record"),
            )],
        })
        .expect("seed");

    let report = MemoryImport::new(&mut database).run(&sources).expect("run");

    assert_eq!(report.saved, 4);
    assert_eq!(
        report.rejected,
        vec![(taken, "another memory already has this key".to_string())]
    );
    assert_eq!(
        ImportSource::parse("codex=~/x", None).expect("parse"),
        ImportSource::Codex {
            directory: PathBuf::from(expand_home("~/x")),
            repository: None,
        }
    );
}
