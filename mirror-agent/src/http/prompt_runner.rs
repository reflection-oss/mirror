//! Single-turn admission and deadlines for headless frontends.

use std::time::Duration;

use mirror_agent::{
    agent_message, chat_record, AgentMessage, ChatRecord, ReactAgentServiceHandle,
    ReactServiceError, ReactServiceEvent, UserMessage,
};
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

/// Own headless submission to one service; overlapping prompts are rejected.
#[derive(Clone)]
pub struct PromptRunner {
    service: ReactAgentServiceHandle,
    prompts: mpsc::Sender<Prompt>,
}

struct Prompt {
    content: String,
    reply: oneshot::Sender<Result<(), PromptError>>,
}

/// Failure to admit or complete a headless turn.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PromptError {
    #[error("a turn is already running")]
    Busy,
    #[error("prompt must not be empty")]
    Empty,
    #[error("turn interrupted")]
    Interrupted,
    #[error("turn deadline exceeded")]
    TimedOut,
    #[error("agent failed: {0}")]
    Failed(String),
    #[error(transparent)]
    Service(#[from] ReactServiceError),
}

impl PromptRunner {
    /// Take exclusive frontend ownership; dropping all handles closes the service.
    pub fn new(service: ReactAgentServiceHandle, timeout: Duration) -> Self {
        let (prompts, receiver) = mpsc::channel(8);
        tokio::spawn(run(service.clone(), timeout, receiver));
        Self { service, prompts }
    }

    /// Wait for a turn to complete or settle after cancellation.
    pub async fn prompt(&self, content: String) -> Result<(), PromptError> {
        if content.trim().is_empty() {
            return Err(PromptError::Empty);
        }
        let (reply, response) = oneshot::channel();
        self.prompts
            .try_send(Prompt { content, reply })
            .map_err(|error| match error {
                mpsc::error::TrySendError::Full(_) => PromptError::Busy,
                mpsc::error::TrySendError::Closed(_) => ReactServiceError::Closed.into(),
            })?;
        response.await.map_err(|_| ReactServiceError::Closed)?
    }

    /// Close the service and cancel all of its tools.
    pub async fn close(&self) -> Result<(), ReactServiceError> {
        self.service.close().await
    }
}

async fn run(
    service: ReactAgentServiceHandle,
    timeout: Duration,
    mut prompts: mpsc::Receiver<Prompt>,
) {
    while let Some(prompt) = prompts.recv().await {
        let turn = execute(&service, prompt.content, timeout);
        tokio::pin!(turn);
        loop {
            tokio::select! {
                result = &mut turn => {
                    let _ = prompt.reply.send(result);
                    break;
                }
                next = prompts.recv() => match next {
                    Some(next) => { let _ = next.reply.send(Err(PromptError::Busy)); }
                    None => {
                        let _ = service.close().await;
                        return;
                    }
                }
            }
        }
    }
    let _ = service.close().await;
}

async fn execute(
    service: &ReactAgentServiceHandle,
    content: String,
    timeout: Duration,
) -> Result<(), PromptError> {
    let mut events = service.events();
    service.execute(AgentMessage {
        id: Uuid::new_v4().to_string(),
        payload: Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::User(UserMessage {
                content,
                checkpoint: None,
            })),
            ..ChatRecord::default()
        })),
        ..AgentMessage::default()
    })?;
    let result = tokio::time::timeout(timeout, async {
        loop {
            match events.recv().await {
                Ok(ReactServiceEvent::TurnCompleted) => return Ok(()),
                Ok(ReactServiceEvent::TurnInterrupted) => return Err(PromptError::Interrupted),
                Ok(ReactServiceEvent::TurnError(error)) => return Err(PromptError::Failed(error)),
                Ok(ReactServiceEvent::Closed) | Err(_) => {
                    return Err(ReactServiceError::Closed.into())
                }
                _ => {}
            }
        }
    })
    .await
    .unwrap_or(Err(PromptError::TimedOut));
    let running = match result.is_err().then(|| service.interrupt()) {
        Some(Ok(running)) => running.await.unwrap_or(false),
        _ => false,
    };
    if running {
        // Wait for cancellation and tool repair before admitting another prompt.
        while let Ok(event) = events.recv().await {
            if matches!(
                event,
                ReactServiceEvent::TurnCompleted | ReactServiceEvent::Closed
            ) {
                break;
            }
        }
    }
    result
}
