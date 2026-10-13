use mirror_protocol::agent_message;
use mirror_protocol::chat_record;
use mirror_protocol::AgentMessage;

/// Select the latest self-contained window and its subsequent records.
pub fn active_window<'a>(
    entries: impl IntoIterator<Item = &'a AgentMessage>,
) -> Vec<&'a AgentMessage> {
    let entries: Vec<_> = entries.into_iter().collect();
    let start = entries
        .iter()
        .rposition(|entry| {
            matches!(
                entry.payload,
                Some(agent_message::Payload::ContextWindow(_))
            )
        })
        .unwrap_or(0);
    entries.into_iter().skip(start).collect()
}

/// Replay the active window as model messages, without trimming.
pub fn active_messages<'a>(
    entries: impl IntoIterator<Item = &'a AgentMessage>,
) -> Vec<chat_record::Message> {
    active_window(entries)
        .into_iter()
        .flat_map(AgentMessage::to_llm_messages)
        .collect()
}
