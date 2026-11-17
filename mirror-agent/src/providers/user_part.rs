//! The parts of a user message that carries images.
//!
//! Each provider adapter maps these parts to its own wire shape. A model that
//! reads only text gets a note in place of each image.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use mirror_protocol::chat_record;
use mirror_protocol::ImageContent;
use mirror_protocol::UserMessage;

/// One part of a user message that carries images.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum UserPart {
    Text(String),
    /// Base64 image data and its IANA media type.
    Image {
        media_type: &'static str,
        data: String,
    },
}

const CLOSE_TAG: &str = "</image>";
const OMITTED: &str = "image content omitted because you do not support image input";
const UNREADABLE: &str = "image content omitted because it is not a PNG, JPEG, GIF, or WebP";

/// Each image between the tags that name it, then the message text.
///
/// A message without images has no parts to split: adapters send its text in
/// the plain shape they always used.
pub(crate) fn user_parts(message: &UserMessage) -> Vec<UserPart> {
    let mut parts = Vec::new();
    for image in &message.images {
        let Some(media_type) = image.media_type() else {
            parts.push(UserPart::Text(note(image, UNREADABLE)));
            continue;
        };
        let named = !image.name.is_empty();
        if named {
            parts.push(UserPart::Text(open_tag(image)));
        }
        parts.push(UserPart::Image {
            media_type,
            data: STANDARD.encode(&image.data),
        });
        if named {
            parts.push(UserPart::Text(CLOSE_TAG.to_string()));
        }
    }
    if !message.content.is_empty() {
        parts.push(UserPart::Text(message.content.clone()));
    }
    parts
}

/// Replace every image with a note, for a model that reads only text.
pub(crate) fn omit_images(messages: &mut [chat_record::Message]) {
    for message in messages {
        let chat_record::Message::User(message) = message else {
            continue;
        };
        if message.images.is_empty() {
            continue;
        }
        let mut lines = message
            .images
            .iter()
            .map(|image| note(image, OMITTED))
            .collect::<Vec<_>>();
        if !message.content.is_empty() {
            lines.push(std::mem::take(&mut message.content));
        }
        message.content = lines.join("\n");
        message.images.clear();
    }
}

fn open_tag(image: &ImageContent) -> String {
    if image.name.is_empty() {
        "<image>".to_string()
    } else {
        format!("<image name={}>", image.name)
    }
}

fn note(image: &ImageContent, text: &str) -> String {
    format!("{}{text}{CLOSE_TAG}", open_tag(image))
}
