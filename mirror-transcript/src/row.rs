//! One encoded terminal row, independent of application state.

use std::sync::Arc;

/// Immutable encoded bytes shared by the source cache and committed frame.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Row {
    pub b: Arc<[u8]>,
    pub w: u16,
    pub s: u32,
    pub source: Option<crate::Source>,
    pub text: Arc<str>,
    pub actions: Arc<[Hit]>,
    pub selection: Option<(usize, String)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub start: usize,
    pub end: usize,
    pub action: Arc<crate::Action>,
}

impl Row {
    /// Conservative cache accounting includes retained hit targets and plain text.
    pub fn bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.b.len()
            + self.text.len()
            + self
                .actions
                .iter()
                .map(|hit| std::mem::size_of::<Hit>() + hit.action.bytes())
                .sum::<usize>()
            + self.selection.as_ref().map_or(0, |(_, text)| text.len())
    }

    /// Fill unstyled cells while preserving explicit content backgrounds.
    pub fn on_canvas(&self, columns: u16, canvas: &str) -> Self {
        if canvas.is_empty() {
            return self.clone();
        }
        let reset = format!("\x1b[0m\x1b[{canvas}m");
        let content = String::from_utf8_lossy(&self.b).replace("\x1b[0m", &reset);
        let padding = " ".repeat(columns.saturating_sub(self.w) as usize);
        Self {
            b: format!("{reset}{content}{reset}{padding}\x1b[0m")
                .into_bytes()
                .into(),
            w: columns,
            s: self.s,
            source: self.source,
            text: self.text.clone(),
            actions: self.actions.clone(),
            selection: self.selection.clone(),
        }
    }
}
