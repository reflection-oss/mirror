//! Skills the user names with `$skill` in a message, injected as context.

use futures_util::future::BoxFuture;
use mirror_protocol::agent_message;
use mirror_protocol::chat_record;
use mirror_protocol::AgentMessage;
use mirror_protocol::ChatRecord;
use mirror_protocol::UserMessage;

use crate::react::ContextHooks;
use crate::system_prompt::mentioned_skills;
use crate::system_prompt::Skill;

/// Follows each user message with the full body of every skill it mentions.
pub struct MentionedSkills {
    skills: Vec<Skill>,
}

impl MentionedSkills {
    pub fn new(skills: Vec<Skill>) -> Self {
        Self { skills }
    }
}

impl ContextHooks for MentionedSkills {
    fn on_input<'a>(
        &'a self,
        _history: &'a [AgentMessage],
        input: &'a [AgentMessage],
    ) -> BoxFuture<'a, Vec<AgentMessage>> {
        let mentioned = input
            .iter()
            .filter_map(|entry| match entry.chat_message() {
                Some(chat_record::Message::User(user)) => Some(user.content.as_str()),
                _ => None,
            })
            .flat_map(|content| mentioned_skills(content, &self.skills))
            .collect::<Vec<_>>();
        Box::pin(async move { skill_context(mentioned) })
    }
}

/// Persisted user context invoking each distinct skill, in mention order.
pub fn skill_context<'a>(skills: impl IntoIterator<Item = &'a Skill>) -> Vec<AgentMessage> {
    let mut seen = std::collections::HashSet::new();
    skills
        .into_iter()
        .filter(|skill| seen.insert(skill.name.as_str()))
        .map(|skill| AgentMessage {
            payload: Some(agent_message::Payload::Chat(ChatRecord {
                message: Some(chat_record::Message::User(UserMessage {
                    checkpoint: None,
                    content: skill.invocation_content(),
                })),
                success: true,
                display: Some(format!("${}", skill.name)),
                ..ChatRecord::default()
            })),
            ..AgentMessage::default()
        })
        .collect()
}
