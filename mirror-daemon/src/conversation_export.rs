use std::path::Path;

use mirror_protocol::agent_message::Payload;
use mirror_protocol::chat_record::Message;
use mirror_protocol::daemon::GetConversationMessagesRequest;
use mirror_protocol::{ChatRecord, PromotedChat};
use prost::Message as _;
use tonic::Status;

pub(crate) fn history(database: &Path, session_id: String) -> Result<Vec<PromotedChat>, Status> {
    let mut request = GetConversationMessagesRequest {
        session_id,
        limit: 128,
        ..Default::default()
    };
    let mut history = Vec::new();
    let mut bytes = 0;
    loop {
        let page = mirror_agent::local::conversation_messages(database, &request)
            .map_err(|e| Status::failed_precondition(e.to_string()))?;
        for entry in page.messages.into_iter().rev() {
            let chat = match entry.payload {
                Some(Payload::Chat(chat)) => chat,
                Some(Payload::Extension(_) | Payload::Fork(_) | Payload::TaskResult(_)) => {
                    let Some(message) = entry.to_llm_messages().pop() else {
                        continue;
                    };
                    ChatRecord {
                        message: Some(message),
                        success: true,
                        ..Default::default()
                    }
                }
                _ => continue,
            };
            let Some(chat) = chat.for_promotion() else {
                continue;
            };
            if history.is_empty()
                && !matches!(&chat.message, Some(Message::Assistant(a)) if !chat.truncated && a.tool_calls.is_empty() && a.content.as_ref().is_some_and(|s| !s.is_empty()))
            {
                return Err(Status::failed_precondition(
                    "wait for a completed assistant response before promoting",
                ));
            }
            let promoted = PromotedChat {
                id: entry.id,
                sequence: entry.sequence.unwrap_or_default(),
                timestamp: entry.timestamp,
                chat: Some(chat),
            };
            bytes += promoted.encoded_len() + 6;
            if bytes > 9 * 1024 * 1024 || history.len() == 4096 {
                return Err(Status::resource_exhausted(
                    "conversation exceeds the promotion limit",
                ));
            }
            history.push(promoted);
        }
        request.before_sequence = page.next_before_sequence;
        if request.before_sequence.is_none() {
            break;
        }
    }
    if history.is_empty() {
        return Err(Status::failed_precondition(
            "conversation has no completed response",
        ));
    }
    history.reverse();
    Ok(history)
}
