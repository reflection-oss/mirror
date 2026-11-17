//! Internal display annotations, separate from the immutable wire content schema.

use std::sync::Arc;

use crate::protocol;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Span {
    pub text: String,
    pub sgr: String,
    pub action: Option<Arc<Action>>,
    pub source: Option<Source>,
}

/// A byte position in a logical block, independent of its width-specific layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Source {
    pub block: u32,
    pub offset: u32,
}

/// Annotate unwrapped leaf lines before layout adds padding, gutters, or borders.
pub fn annotate_lines(lines: &mut [crate::Line], mut source: Source) {
    for line in lines {
        if line.is_empty() {
            line.push(Span::default());
        }
        for span in line {
            span.source = Some(source);
            source = source.shifted(span.text.len());
        }
        source = source.shifted(1);
    }
}

impl Source {
    pub fn shifted(self, bytes: usize) -> Self {
        Self {
            offset: self
                .offset
                .saturating_add(u32::try_from(bytes).unwrap_or(u32::MAX)),
            ..self
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    Copy(String),
    Link(String),
}

impl Span {
    pub fn bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.text.capacity()
            + self.sgr.capacity()
            + self.action.as_ref().map_or(0, |action| action.bytes())
    }
}

impl Action {
    pub fn bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + match self {
                Self::Copy(text) | Self::Link(text) => text.capacity(),
            }
    }
}

pub fn link_action(url: &str) -> Option<Action> {
    ((url.starts_with("https://") || url.starts_with("http://") || url.starts_with("mailto:"))
        && !url.chars().any(char::is_control))
    .then(|| Action::Link(url.to_owned()))
}

impl From<protocol::Span> for Span {
    fn from(value: protocol::Span) -> Self {
        Self {
            text: value.text,
            sgr: value.sgr,
            action: None,
            source: None,
        }
    }
}
