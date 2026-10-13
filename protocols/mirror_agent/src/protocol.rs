//! Generated Mirror agent protocol bindings.

tonic::include_proto!("mirror.agent.v1");

/// Protocol descriptors for native JSON trace artifacts.
pub const FILE_DESCRIPTOR_SET: &[u8] = tonic::include_file_descriptor_set!("descriptor");

impl Compaction {
    pub fn to_llm_messages(&self) -> Vec<chat_record::Message> {
        let mut messages = Vec::new();
        if let Some(summary) = self
            .summary
            .as_deref()
            .filter(|summary| !summary.is_empty())
        {
            messages.push(chat_record::Message::User(UserMessage {
                checkpoint: None,
                content: format!(
                    "The conversation so far was compacted into this summary. Continue from where it leaves off instead of starting over:\n<summary>\n{}\n</summary>",
                    summary
                ),
            }));
        }
        messages.extend(self.recent.iter().flat_map(AgentMessage::to_llm_messages));
        messages
    }
}

impl AgentMessage {
    /// Describe an action's approval without retaining its arguments or provider fields.
    pub fn tool_approval(
        call: &ToolCall,
        outcome: ToolApprovalOutcome,
        source: ApprovalSource,
    ) -> Self {
        Self {
            payload: Some(agent_message::Payload::ToolApproval(ToolApprovalRecord {
                call_id: call.id.clone(),
                tool_name: call
                    .function
                    .as_ref()
                    .map(|function| function.name.clone())
                    .unwrap_or_default(),
                outcome: outcome.into(),
                source: source.into(),
            })),
            ..Self::default()
        }
    }

    /// The chat payload carried by this trace record, when present.
    pub fn chat_message(&self) -> Option<&chat_record::Message> {
        match &self.payload {
            Some(agent_message::Payload::Chat(ChatRecord {
                message: Some(message),
                ..
            })) => Some(message),
            _ => None,
        }
    }

    /// Project this trace record into messages sent to the model.
    pub fn to_llm_messages(&self) -> Vec<chat_record::Message> {
        let Some(payload) = &self.payload else {
            return Vec::new();
        };
        match payload {
            agent_message::Payload::Chat(chat) => chat.message.clone().into_iter().collect(),
            agent_message::Payload::ContextWindow(window) => {
                let mut messages = Vec::new();
                if !window.system_prompt.trim().is_empty() {
                    messages.push(chat_record::Message::System(SystemMessage {
                        content: window.system_prompt.clone(),
                    }));
                }
                if let Some(compaction) = &window.compaction {
                    messages.extend(compaction.to_llm_messages());
                }
                messages
            }
            agent_message::Payload::TaskResult(result) => {
                let status = if result.success {
                    "succeeded"
                } else {
                    "failed"
                };
                vec![chat_record::Message::User(UserMessage {
                checkpoint: None,
                    content: format!(
                        "<task_result id=\"{}\" tool=\"{}\" status=\"{status}\">\n{}\n</task_result>",
                        result.task_id, result.tool_name, result.content
                    ),
                })]
            }
            agent_message::Payload::Extension(extension) => extension
                .model_message
                .clone()
                .map(chat_record::Message::User)
                .into_iter()
                .collect(),
            agent_message::Payload::Log(_) | agent_message::Payload::ToolApproval(_) => Vec::new(),
        }
    }
}
