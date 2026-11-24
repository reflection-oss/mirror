//! Generated Mirror agent protocol bindings.

tonic::include_proto!("mirror.agent.v1");

/// Protocol descriptors for native JSON trace artifacts.
pub const FILE_DESCRIPTOR_SET: &[u8] = tonic::include_file_descriptor_set!("descriptor");

impl ChatRecord {
    /// Keep portable conversation content, never local checkpoints or provider-specific state.
    pub fn for_promotion(mut self) -> Option<Self> {
        self.sampling = None;
        match self.message.as_mut()? {
            chat_record::Message::User(user) => user.checkpoint = None,
            chat_record::Message::Assistant(assistant) => {
                assistant.provider_fields = None;
                assistant.reasoning_details.clear();
                for call in &mut assistant.tool_calls {
                    call.provider_fields = None;
                }
            }
            chat_record::Message::Tool(_) => {}
            chat_record::Message::System(_) => return None,
        }
        Some(self)
    }
}

impl Compaction {
    pub fn to_llm_messages(&self) -> Vec<chat_record::Message> {
        let mut messages = Vec::new();
        if let Some(summary) = self
            .summary
            .as_deref()
            .filter(|summary| !summary.is_empty())
        {
            messages.push(chat_record::Message::User(UserMessage {
                parts: Vec::new(),
                documents: Vec::new(),
                checkpoint: None,
                images: Vec::new(),
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
                parts: Vec::new(),
                documents: Vec::new(),
                checkpoint: None,
                images: Vec::new(),
                    content: format!(
                        "<task_result id=\"{}\" tool=\"{}\" status=\"{status}\">\n{}\n</task_result>",
                        result.task_id, result.tool_name, result.content
                    ),
                })]
            }
            agent_message::Payload::Fork(fork) => {
                vec![chat_record::Message::User(UserMessage {
                    parts: Vec::new(),
                    documents: Vec::new(),
                    checkpoint: None,
                    images: Vec::new(),
                    content: format!(
                        "This conversation was forked from session {}. Earlier messages describe \
                         work in {:?}. The current workspace is {:?}. Continue in this checkout; \
                         verify inherited absolute paths before using them.",
                        fork.parent_session_id, fork.source_workspace, fork.destination_workspace
                    ),
                })]
            }
            agent_message::Payload::Extension(extension) => extension
                .model_message
                .clone()
                .map(chat_record::Message::User)
                .into_iter()
                .collect(),
            agent_message::Payload::Log(_)
            | agent_message::Payload::ToolApproval(_)
            | agent_message::Payload::TurnEnd(_) => Vec::new(),
        }
    }
}

impl ImageContent {
    /// The IANA media type of `data`, read from its leading bytes; `None` for
    /// anything other than a PNG, JPEG, GIF, or WebP.
    pub fn media_type(&self) -> Option<&'static str> {
        match self.data.as_slice() {
            [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n', ..] => Some("image/png"),
            [0xff, 0xd8, 0xff, ..] => Some("image/jpeg"),
            [b'G', b'I', b'F', b'8', b'7' | b'9', b'a', ..] => Some("image/gif"),
            [b'R', b'I', b'F', b'F', _, _, _, _, b'W', b'E', b'B', b'P', ..] => Some("image/webp"),
            _ => None,
        }
    }
}

impl UserMessage {
    /// Project documents into model text without changing the stored prompt.
    pub fn text_content(&self) -> String {
        let mut content = self.content.clone();
        for document in &self.documents {
            content.push_str(&format!(
                "\n\n<document name={:?}>\n{}\n</document>",
                document.name, document.content
            ));
        }
        content
    }
}
