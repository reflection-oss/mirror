use crate::Store;
use assert_matches::assert_matches;
use futures_util::future::BoxFuture;
use mirror_protocol::AgentMessage;
use mirror_protocol::ChatRecord;
use mirror_protocol::ToolMessage;
use mirror_protocol::UserMessage;
use pretty_assertions::assert_eq;

use super::*;
use crate::react::InMemoryStore;
use crate::task_manager::TaskManager;
use crate::AgentApprovalMode;
use crate::ExecutionContext;

fn execution_context() -> ExecutionContext {
    let (_manager, tasks, _task_events, _task_runner) = TaskManager::new();
    ExecutionContext::new(AgentApprovalMode::Auto, None, tasks, None, None)
}

/// Follows input and tool replies with a note saying what it saw.
struct Recorder;

impl ContextHooks for Recorder {
    fn on_input<'a>(
        &'a self,
        history: &'a [AgentMessage],
        input: &'a [AgentMessage],
    ) -> BoxFuture<'a, Vec<AgentMessage>> {
        Box::pin(async move {
            vec![note(&format!(
                "after {} input with {} in history",
                input.len(),
                history.len()
            ))]
        })
    }

    fn on_tool_replies<'a>(
        &'a self,
        _history: &'a [AgentMessage],
        replies: &'a [AgentMessage],
    ) -> BoxFuture<'a, Vec<AgentMessage>> {
        Box::pin(async move { vec![note(&format!("after {} replies", replies.len()))] })
    }
}

fn user(content: &str) -> AgentMessage {
    AgentMessage {
        payload: Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::User(UserMessage {
                checkpoint: None,
                content: content.to_string(),
            })),
            ..ChatRecord::default()
        })),
        ..AgentMessage::default()
    }
}

/// Injected context: a user-role record with a `display`, not the user's own words.
fn note(content: &str) -> AgentMessage {
    AgentMessage {
        payload: Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::User(UserMessage {
                checkpoint: None,
                content: content.to_string(),
            })),
            display: Some("note".to_string()),
            ..ChatRecord::default()
        })),
        ..AgentMessage::default()
    }
}

fn user_contents(entries: &[AgentMessage]) -> Vec<&str> {
    entries
        .iter()
        .filter_map(|entry| match entry.chat_message() {
            Some(chat_record::Message::User(user)) => Some(user.content.as_str()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn hooks_follow_committed_input_and_tool_replies_in_the_same_push() {
    let (context, mut events) = ReactContext::new(
        "session",
        Vec::new(),
        Box::new(InMemoryStore::default()),
        execution_context(),
    );
    let mut context = context.with_hooks(vec![Box::new(Recorder)]);

    context
        .push(AgentEvent::Entries(vec![user("hello"), note("injected")]))
        .await
        .expect("append input");

    assert_eq!(
        user_contents(context.entries()),
        ["hello", "injected", "after 1 input with 2 in history"]
    );
    assert_matches!(events.recv().await, Some(AgentEvent::Entries(committed)) if committed.len() == 2);
    assert_matches!(
        events.recv().await,
        Some(AgentEvent::Entries(followed)) if followed.len() == 1 && followed[0].sequence == Some(3)
    );

    context
        .push(AgentEvent::Entries(vec![AgentMessage {
            payload: Some(agent_message::Payload::Chat(ChatRecord {
                message: Some(chat_record::Message::Tool(ToolMessage::default())),
                ..ChatRecord::default()
            })),
            ..AgentMessage::default()
        }]))
        .await
        .expect("append tool reply");
    context
        .push(AgentEvent::StreamFinal {
            entry: Box::default(),
            tokens_per_second: None,
        })
        .await
        .expect("append assistant reply");

    assert_eq!(
        user_contents(context.entries()).last(),
        Some(&"after 1 replies")
    );
    assert_eq!(context.entries().len(), 6);
}

#[tokio::test]
async fn in_memory_fork_preserves_the_window_and_commits_only_to_the_child() {
    let mut store = InMemoryStore::default();
    let mut history = vec![AgentMessage::default()];
    store
        .append("parent", 0, &mut history)
        .await
        .expect("old history");
    let mut entries = vec![
        AgentMessage {
            payload: Some(mirror_protocol::agent_message::Payload::ContextWindow(
                mirror_protocol::ContextWindowRecord {
                    system_prompt: "frozen".to_string(),
                    ..Default::default()
                },
            )),
            ..Default::default()
        },
        AgentMessage::default(),
    ];
    store
        .append("parent", 1, &mut entries)
        .await
        .expect("active window");
    history.extend_from_slice(&entries);
    let (mut parent, mut events) = ReactContext::new(
        "parent",
        entries.clone(),
        Box::new(store),
        execution_context(),
    );
    let mut child = parent
        .in_memory_fork(execution_context())
        .await
        .expect("fork after final entry");
    assert_eq!(child.entries(), entries);
    child
        .push(AgentEvent::Entries(vec![AgentMessage::default()]))
        .await
        .expect("child append");
    assert_eq!(
        child.entries().last().expect("child entry").sequence,
        Some(4)
    );
    assert_eq!(
        parent.history(0, 100).await.expect("parent history").0,
        history
    );
    assert!(events.try_recv().is_err());
    parent
        .push(AgentEvent::Entries(vec![AgentMessage::default()]))
        .await
        .expect("parent append");
    assert_ne!(
        child.entries().last().expect("child").id,
        parent.entries().last().expect("parent").id
    );
    assert_eq!(
        child.history(0, 100).await.expect("child history").0.len(),
        3
    );
}

#[tokio::test]
async fn publishes_committed_entries_and_keeps_ephemeral_events_out_of_history() {
    let mut store = InMemoryStore::default();
    let mut initial = vec![AgentMessage::default()];
    store
        .append("session", 0, &mut initial)
        .await
        .expect("seed history");
    let (mut context, mut events) =
        ReactContext::new("session", initial, Box::new(store), execution_context());

    context
        .push(AgentEvent::Entries(vec![AgentMessage::default()]))
        .await
        .expect("append entry");
    let Some(AgentEvent::Entries(committed)) = events.recv().await else {
        panic!("expected committed entries");
    };
    assert_eq!(committed, context.entries()[1..]);
    assert_eq!(committed[0].sequence, Some(2));
    assert!(!committed[0].id.is_empty());

    context
        .push(AgentEvent::CompactionStarted)
        .await
        .expect("publish ephemeral event");
    assert_matches!(events.recv().await, Some(AgentEvent::CompactionStarted));
    context
        .push(AgentEvent::StreamFinal {
            entry: Box::default(),
            tokens_per_second: Some(42.0),
        })
        .await
        .expect("append stream final");
    let Some(AgentEvent::StreamFinal {
        entry: committed,
        tokens_per_second,
    }) = events.recv().await
    else {
        panic!("expected committed stream final");
    };
    assert_eq!(context.entries().len(), 3);
    assert_eq!(*committed, context.entries()[2]);
    assert_eq!(committed.sequence, Some(3));
    assert_eq!(tokens_per_second, Some(42.0));
}

#[tokio::test]
async fn input_retries_before_the_window_fail_at_persistence() {
    let mut store = InMemoryStore::default();
    let mut history = vec![AgentMessage {
        id: "old".to_string(),
        ..Default::default()
    }];
    store
        .append("session", 0, &mut history)
        .await
        .expect("old input");
    let mut window = vec![AgentMessage {
        payload: Some(mirror_protocol::agent_message::Payload::ContextWindow(
            mirror_protocol::ContextWindowRecord {
                system_prompt: "system".to_string(),
                ..Default::default()
            },
        )),
        ..Default::default()
    }];
    store
        .append("session", 1, &mut window)
        .await
        .expect("checkpoint");
    let (mut context, _) =
        ReactContext::new("session", window, Box::new(store), execution_context());
    context.input().push(history[0].clone()).expect("retry");
    context
        .input()
        .push(AgentMessage::default())
        .expect("attached context");
    let pending = context.flush_pending();
    assert_eq!(pending.len(), 2);
    assert!(!context.has_pending());
    assert_matches!(
        context.push(AgentEvent::Entries(pending)).await,
        Err(ReactContextError::Store(
            StoreError::MessageIdConflict { .. }
        ))
    );
    assert_eq!(context.entries().len(), 1);
    assert_eq!(
        context
            .history(0, 100)
            .await
            .expect("unchanged history")
            .0
            .len(),
        2
    );
}

#[tokio::test]
async fn push_persists_and_publishes_input_when_checkpoint_capture_fails() {
    let directory = tempfile::tempdir().expect("database directory");
    let workspace = tempfile::tempdir().expect("workspace");
    assert!(std::process::Command::new("git")
        .args(["init", "--quiet"])
        .current_dir(workspace.path())
        .status()
        .expect("Git init")
        .success());
    let nested = workspace.path().join("nested");
    std::fs::create_dir(&nested).expect("nested repository");
    assert!(std::process::Command::new("git")
        .args(["init", "--quiet"])
        .current_dir(&nested)
        .status()
        .expect("nested Git init")
        .success());
    let mut store = crate::local::SQLiteStore::open(
        &directory.path().join("state.db"),
        "session",
        workspace.path(),
        true,
        false,
    )
    .expect("open store");
    store.checkpoints_enabled = true;
    let (mut context, mut events) =
        ReactContext::new("session", Vec::new(), Box::new(store), execution_context());

    context
        .push(AgentEvent::Entries(vec![user("first")]))
        .await
        .expect("capture failure must not reject input");
    assert_matches!(context.entries()[0].chat_message(), Some(chat_record::Message::User(user)) if user.checkpoint.is_none());
    assert_matches!(events.try_recv(), Ok(AgentEvent::Entries(entries)) if entries == context.entries());

    std::fs::remove_dir_all(nested).expect("remove unborn nested repository");
    context
        .push(AgentEvent::Entries(vec![user("second")]))
        .await
        .expect("later input must still succeed");
    assert_matches!(context.entries()[1].chat_message(), Some(chat_record::Message::User(user)) if user.checkpoint.is_some());
    assert_matches!(events.try_recv(), Ok(AgentEvent::Entries(entries)) if entries == context.entries()[1..]);
    assert_eq!(user_contents(context.entries()), ["first", "second"]);
    assert_eq!(
        context.history(0, 100).await.expect("persisted history").0,
        context.entries()
    );
}

#[tokio::test]
async fn stale_push_does_not_mutate_or_publish_history() {
    let mut store = InMemoryStore::default();
    let mut initial = vec![AgentMessage::default()];
    store
        .append("session", 0, &mut initial)
        .await
        .expect("seed history");
    store
        .append("session", 1, &mut [AgentMessage::default()])
        .await
        .expect("advance beyond the context's snapshot");
    let (mut context, mut events) = ReactContext::new(
        "session",
        initial.clone(),
        Box::new(store),
        execution_context(),
    );

    assert_matches!(
        context
            .push(AgentEvent::Entries(vec![AgentMessage::default()]))
            .await,
        Err(ReactContextError::Store(StoreError::Stale))
    );
    assert_eq!(context.entries(), initial);
    assert!(events.try_recv().is_err());
    assert_matches!(
        context.push(AgentEvent::Entries(vec![user("retry")])).await,
        Err(ReactContextError::Store(StoreError::Persistence(message)))
            if message.contains("reopen the session")
    );
}

#[tokio::test]
async fn stripping_before_the_active_window_drops_older_entries_and_keeps_the_head() {
    let mut store = InMemoryStore::default();
    let mut history = vec![
        window("first"),
        AgentMessage {
            id: "old".to_string(),
            ..Default::default()
        },
        window("second"),
        AgentMessage {
            id: "recent".to_string(),
            ..Default::default()
        },
    ];
    store
        .append("session", 0, &mut history)
        .await
        .expect("seed history");
    let (mut context, _) = ReactContext::new(
        "session",
        history.clone(),
        Box::new(store),
        execution_context(),
    );

    context
        .strip_before_active_window()
        .expect("history has a window");

    assert_eq!(context.entries(), &history[2..]);
    assert_eq!(context.system_prompt().expect("system prompt"), "second");
    context
        .push(AgentEvent::Entries(vec![AgentMessage::default()]))
        .await
        .expect("append after retaining");
    assert_eq!(context.entries().len(), 3);
    assert_eq!(context.entries()[2].sequence, Some(5));
    assert_eq!(
        context
            .history(0, 100)
            .await
            .expect("stored history")
            .0
            .len(),
        5
    );
}

fn window(system_prompt: &str) -> AgentMessage {
    AgentMessage {
        payload: Some(mirror_protocol::agent_message::Payload::ContextWindow(
            mirror_protocol::ContextWindowRecord {
                system_prompt: system_prompt.to_string(),
                ..Default::default()
            },
        )),
        ..Default::default()
    }
}

#[tokio::test]
async fn stripping_without_a_window_is_an_invalid_history() {
    let (mut context, _) = ReactContext::new(
        "session",
        vec![AgentMessage::default()],
        Box::new(InMemoryStore::default()),
        execution_context(),
    );

    assert_matches!(
        context.strip_before_active_window(),
        Err(ReactContextError::InvalidHistory(_))
    );
    assert_eq!(context.entries().len(), 1);
}
