use futures_util::future::BoxFuture;
use mirror_protocol::{agent_message, chat_record, AgentMessage, ChatRecord, UserMessage};

use crate::StoreError;

pub trait Inbox: Send + Sync {
    fn admit<'a>(
        &'a self,
        session_id: &'a str,
        input: &'a AgentMessage,
    ) -> BoxFuture<'a, Result<InboxAdmission, StoreError>>;

    fn take<'a>(
        &'a self,
        session_id: &'a str,
    ) -> BoxFuture<'a, Result<Vec<AgentMessage>, StoreError>>;

    fn has_unfinished<'a>(
        &'a self,
        _session_id: &'a str,
    ) -> BoxFuture<'a, Result<bool, StoreError>> {
        Box::pin(async { Ok(false) })
    }

    fn complete<'a>(&'a self, _session_id: &'a str) -> BoxFuture<'a, Result<(), StoreError>> {
        Box::pin(async { Ok(()) })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum InboxAdmission {
    Admitted,
    Pending(AgentMessage),
    Appended(AgentMessage),
}

pub fn input_payload(entry: &AgentMessage) -> Option<agent_message::Payload> {
    match entry.payload.as_ref()? {
        agent_message::Payload::Chat(chat) => {
            let Some(chat_record::Message::User(user)) = &chat.message else {
                return None;
            };
            Some(agent_message::Payload::Chat(ChatRecord {
                message: Some(chat_record::Message::User(UserMessage {
                    checkpoint: None,
                    ..user.clone()
                })),
                ..chat.clone()
            }))
        }
        agent_message::Payload::Extension(extension) => {
            Some(agent_message::Payload::Extension(extension.clone()))
        }
        agent_message::Payload::TaskResult(result) => {
            Some(agent_message::Payload::TaskResult(result.clone()))
        }
        _ => None,
    }
}
