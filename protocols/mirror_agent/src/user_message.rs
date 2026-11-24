//! How a composed user message reaches the model.
//!
//! Clients send what the user composed as `UserMessage.parts`; the agent
//! renders `content` from them here, so quoted selections read the same to the
//! model whichever client the user typed in.

use crate::user_message_part::Part;
use crate::UserMessage;
use crate::UserMessagePart;

const OPEN: &str = "<quoted_selection>";
const CLOSE: &str = "</quoted_selection>";
const TAG_NAME: &str = "quoted_selection>";

impl UserMessage {
    /// Render `content` from `parts`, leaving plain-text messages unchanged.
    pub fn render_parts(&mut self) {
        if !self.parts.is_empty() {
            self.content = render(&self.parts);
        }
    }
}

impl UserMessagePart {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            part: Some(Part::Text(text.into())),
        }
    }

    pub fn quoted_selection(text: impl Into<String>) -> Self {
        Self {
            part: Some(Part::QuotedSelection(text.into())),
        }
    }
}

/// Each quote becomes a tagged block on its own lines, absorbing the spacing
/// that surrounded it.
fn render(parts: &[UserMessagePart]) -> String {
    let mut content = String::new();
    let mut after_block = false;
    for part in parts {
        match &part.part {
            Some(Part::Text(text)) => {
                let text = if after_block {
                    let text = text.trim_start_matches([' ', '\t']);
                    text.strip_prefix('\n').unwrap_or(text)
                } else {
                    text
                };
                content.push_str(text);
                after_block = false;
            }
            Some(Part::QuotedSelection(quoted)) => {
                let kept = content.trim_end_matches([' ', '\t']);
                let kept = kept.strip_suffix('\n').unwrap_or(kept).len();
                content.truncate(kept);
                content.push_str(&format!("\n{OPEN}\n{}\n{CLOSE}\n", escape(quoted)));
                after_block = true;
            }
            None => {}
        }
    }
    content.trim_matches('\n').to_owned()
}

/// Add a backslash to tags in quoted text, so a block always ends at its own
/// closing tag.
fn escape(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(index) = rest.find(TAG_NAME) {
        let before = &rest[..index];
        let head = before.trim_end_matches('\\');
        let opening = head.strip_suffix('/').unwrap_or(head);
        escaped.push_str(head);
        if opening.ends_with('<') {
            escaped.push('\\');
        }
        escaped.push_str(&before[head.len()..]);
        escaped.push_str(TAG_NAME);
        rest = &rest[index + TAG_NAME.len()..];
    }
    escaped.push_str(rest);
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rendered(parts: Vec<UserMessagePart>) -> String {
        let mut message = UserMessage {
            images: Vec::new(),
            documents: Vec::new(),
            content: String::new(),
            checkpoint: None,
            parts,
        };
        message.render_parts();
        message.content
    }

    #[test]
    fn quotes_render_as_blocks_on_their_own_lines() {
        let content = rendered(vec![
            UserMessagePart::text("Why "),
            UserMessagePart::quoted_selection("line one\nline two"),
            UserMessagePart::text(" fails?"),
        ]);

        assert_eq!(
            content,
            "Why\n<quoted_selection>\nline one\nline two\n</quoted_selection>\nfails?"
        );
    }

    #[test]
    fn quoted_tags_are_escaped() {
        let content = rendered(vec![UserMessagePart::quoted_selection(
            "<quoted_selection>a</quoted_selection> </\\quoted_selection> quoted_selection>",
        )]);

        assert_eq!(
            content,
            "<quoted_selection>\n<\\quoted_selection>a</\\quoted_selection> </\\\\quoted_selection> quoted_selection>\n</quoted_selection>"
        );
    }

    #[test]
    fn plain_messages_keep_their_content() {
        let mut message = UserMessage {
            images: Vec::new(),
            documents: Vec::new(),
            content: "hello".into(),
            checkpoint: None,
            parts: Vec::new(),
        };

        message.render_parts();

        assert_eq!(message.content, "hello");
    }
}
