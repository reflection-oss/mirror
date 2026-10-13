use assert_matches::assert_matches;
use pretty_assertions::assert_eq;

use super::*;
use crate::memory::pinned_bullets;
use crate::memory::testing::open_store;
use crate::memory::testing::TestWorkspace;
use crate::memory::testing::REPO;
use crate::tools::AnyTool;

fn tool(directory: &tempfile::TempDir) -> (SaveMemoryTool, Arc<dyn MemoryStore>) {
    let store = open_store(&directory.path().join("store"));
    let tool = SaveMemoryTool::new(
        store.clone(),
        Arc::new(TestWorkspace::ROOT),
        "session",
        directory.path().join("PINNED_MEMORIES.md"),
    );
    (tool, store)
}

async fn call(tool: &SaveMemoryTool, arguments: &str) -> ToolOutput {
    AnyTool::execute(tool, arguments, &ExecutionContext::default()).await
}

async fn active(store: &dyn MemoryStore) -> Vec<MemoryRecord> {
    store
        .search(MemoryQuery {
            repository: Some(REPO.into()),
            ..MemoryQuery::default()
        })
        .await
        .expect("scan")
}

#[tokio::test]
async fn remember_saves_a_user_record_and_replaces_retires_the_old_one() {
    let directory = tempfile::tempdir().expect("directory");
    let (tool, store) = tool(&directory);

    let saved = call(
        &tool,
        r#"{"action":"remember","kind":"preference","content":"Reply tersely.","situation":"answering questions"}"#,
    )
    .await;
    assert!(saved.success, "{}", saved.content);
    assert!(
        saved
            .content
            .starts_with("Saved preference 'reply-tersely'"),
        "{}",
        saved.content
    );
    let records = active(store.as_ref()).await;
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].actor, MemoryActor::User);
    assert_eq!(records[0].repository, "", "global by default");

    let replaced = call(
        &tool,
        r#"{"action":"remember","kind":"preference","content":"Reply in two sentences.","situation":"answering questions","replaces":"reply-tersely","scope":"repository"}"#,
    )
    .await;
    assert!(replaced.success, "{}", replaced.content);
    let records = active(store.as_ref()).await;
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].content, "Reply in two sentences.");
    assert_eq!(records[0].repository, REPO);

    let refused = call(
        &tool,
        r#"{"action":"remember","kind":"rumour","content":"x"}"#,
    )
    .await;
    assert_matches!(refused, ToolOutput { success: false, .. });
}

#[tokio::test]
async fn forget_retires_by_key_or_id() {
    let directory = tempfile::tempdir().expect("directory");
    let (tool, store) = tool(&directory);
    let saved = call(
        &tool,
        r#"{"action":"remember","kind":"fact","content":"The build uses bazelisk.","key":"bazelisk"}"#,
    )
    .await;
    assert!(saved.success, "{}", saved.content);
    let id = active(store.as_ref()).await[0].id.clone();

    let missing = call(&tool, r#"{"action":"forget","key":"nothing"}"#).await;
    assert_matches!(missing, ToolOutput { success: false, .. });
    let forgotten = call(&tool, &format!(r#"{{"action":"forget","id":"{id}"}}"#)).await;
    assert!(forgotten.success, "{}", forgotten.content);
    assert!(active(store.as_ref()).await.is_empty());
    assert_eq!(
        store
            .get(&id)
            .await
            .expect("get")
            .map(|record| record.status),
        Some(MemoryStatus::Retired)
    );
}

#[tokio::test]
async fn pin_writes_a_bullet_from_text_or_from_an_existing_record() {
    let directory = tempfile::tempdir().expect("directory");
    let pinned = directory.path().join("PINNED_MEMORIES.md");
    let (tool, store) = tool(&directory);
    let saved = call(
        &tool,
        r#"{"action":"remember","kind":"preference","content":"Ask before force-pushing.","key":"ask-before-force-push"}"#,
    )
    .await;
    assert!(saved.success, "{}", saved.content);
    let id = active(store.as_ref()).await[0].id.clone();

    let from_text = call(&tool, r#"{"action":"pin","content":"Reply tersely."}"#).await;
    assert!(from_text.success, "{}", from_text.content);
    let from_record = call(&tool, &format!(r#"{{"action":"pin","id":"{id}"}}"#)).await;
    assert!(from_record.success, "{}", from_record.content);
    assert_eq!(
        pinned_bullets(&pinned).expect("bullets"),
        vec!["Reply tersely.", "Ask before force-pushing."]
    );
    assert_eq!(
        active(store.as_ref()).await.len(),
        1,
        "the record stays retrievable"
    );

    let unpinned = call(&tool, r#"{"action":"unpin","key":"ask-before-force-push"}"#).await;
    assert!(unpinned.success, "{}", unpinned.content);
    assert_eq!(
        pinned_bullets(&pinned).expect("bullets"),
        vec!["Reply tersely."]
    );
    let nothing = call(&tool, r#"{"action":"pin"}"#).await;
    assert_matches!(nothing, ToolOutput { success: false, .. });
}
