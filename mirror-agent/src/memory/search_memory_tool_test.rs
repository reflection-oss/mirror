use assert_matches::assert_matches;
use pretty_assertions::assert_eq;

use super::*;
use crate::memory::create_record;
use crate::memory::testing::seeded_store;
use crate::memory::testing::TestWorkspace;
use crate::memory::testing::REPO;
use crate::memory::MemoryActor;
use crate::memory::MemoryDraft;
use crate::tools::AnyTool;

fn record(key: &str, repository: &str, content: &str, globs: &[&str]) -> MemoryRecord {
    create_record(
        MemoryDraft {
            r#type: MemoryType::Procedure,
            key: key.into(),
            repository: repository.into(),
            path_globs: globs.iter().map(|glob| glob.to_string()).collect(),
            situation: "building the harness".into(),
            keywords: vec!["bazelisk".into()],
            content: content.into(),
            source: String::new(),
        },
        MemoryActor::User,
        1_000,
    )
    .expect("record")
}

async fn tool_with(records: Vec<MemoryRecord>) -> (tempfile::TempDir, SearchMemoryTool) {
    let directory = tempfile::tempdir().expect("directory");
    let store = seeded_store(directory.path(), records).await;
    (
        directory,
        SearchMemoryTool::new(store, Arc::new(TestWorkspace::under("crates/agent"))),
    )
}

async fn call(tool: &SearchMemoryTool, arguments: &str) -> ToolOutput {
    AnyTool::execute(tool, arguments, &ExecutionContext::default()).await
}

#[tokio::test]
async fn a_query_returns_matching_records_and_names_their_ids() {
    let fillers: Vec<MemoryRecord> = (0..8)
        .map(|index| {
            let mut filler = record(
                &format!("filler-{index}"),
                REPO,
                &format!("Unrelated note {index}."),
                &[],
            );
            filler.keywords = vec!["gazelle".into()];
            filler
        })
        .collect();
    let bazelisk = record(
        "bazelisk-build",
        REPO,
        "Use bazelisk to build; bazel alone is unpinned.",
        &[],
    );
    let other = record(
        "other-repo",
        "github.com/example/other",
        "Bazelisk note for another repository.",
        &[],
    );
    let mut records = fillers;
    records.push(bazelisk.clone());
    records.push(other);
    let (_directory, tool) = tool_with(records).await;

    let result = call(&tool, r#"{"query":"how do I build with bazelisk"}"#).await;

    assert!(result.success, "{}", result.content);
    assert!(result.content.starts_with("1 matching memory:"));
    assert!(result.content.contains("Use bazelisk to build"));
    assert!(result
        .content
        .ends_with(&format!("{RESULT_IDS_PREFIX}{}", bazelisk.id)));
    assert_eq!(
        search_result_ids(&result.content).collect::<Vec<_>>(),
        vec![bazelisk.id]
    );
}

#[tokio::test]
async fn an_id_returns_the_complete_record_when_it_applies_here() {
    let here = record("here", REPO, "Applies here.", &[]);
    let elsewhere = record(
        "elsewhere",
        "github.com/example/other",
        "Applies elsewhere.",
        &[],
    );
    let (_directory, tool) = tool_with(vec![here.clone(), elsewhere.clone()]).await;

    let found = call(&tool, &format!(r#"{{"id":"{}"}}"#, here.id)).await;
    assert!(found.success);
    assert!(found.content.contains("Applies here."));
    assert!(found
        .content
        .ends_with(&format!("{RESULT_IDS_PREFIX}{}", here.id)));

    let refused = call(&tool, &format!(r#"{{"id":"{}"}}"#, elsewhere.id)).await;
    assert_matches!(refused, ToolOutput { success: false, .. });
}

#[tokio::test]
async fn paths_return_memories_tied_to_them() {
    let tied = record(
        "react-loop",
        REPO,
        "Mind the loop.",
        &["crates/agent/src/react/**"],
    );
    let (_directory, tool) =
        tool_with(vec![tied.clone(), record("free", REPO, "Unrelated.", &[])]).await;

    let result = call(&tool, r#"{"paths":["src/react/agent.rs"]}"#).await;

    assert!(result.success);
    assert!(result
        .content
        .contains("attached: matched crates/agent/src/react/agent.rs"));
    assert!(result
        .content
        .ends_with(&format!("{RESULT_IDS_PREFIX}{}", tied.id)));
}

#[tokio::test]
async fn bad_arguments_are_tool_errors_and_no_match_is_a_plain_answer() {
    let (_directory, tool) = tool_with(Vec::new()).await;

    assert_matches!(
        call(&tool, r#"{}"#).await,
        ToolOutput { success: false, .. }
    );
    assert_matches!(
        call(&tool, r#"{"query":"x","types":["rumour"]}"#).await,
        ToolOutput { success: false, .. }
    );
    let none = call(&tool, r#"{"query":"nothing saved about this"}"#).await;
    assert!(none.success);
    assert_eq!(
        none.content,
        "No memories match. Nothing has been saved about this yet."
    );
}
