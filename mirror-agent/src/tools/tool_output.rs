//! Normalized completed tool output.

use mirror_protocol::MetadataRecord;
use mirror_protocol::TouchedPath;

/// A completed tool call returned to the model and terminal.
///
/// `touched` and `exit_code` are facts the tool reports about what it did;
/// they are stored in history beside the content and never shown to the model.
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use]
pub struct ToolOutput {
    pub content: String,
    pub display: Option<String>,
    pub success: bool,
    pub touched: Vec<TouchedPath>,
    pub exit_code: Option<i32>,
}

impl ToolOutput {
    /// Construct a successful result.
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            display: None,
            success: true,
            touched: Vec::new(),
            exit_code: None,
        }
    }

    /// Construct a failed result with the canonical prefix.
    pub fn error(message: impl AsRef<str>) -> Self {
        Self {
            content: format!("Tool error: {}", message.as_ref()),
            display: None,
            success: false,
            touched: Vec::new(),
            exit_code: None,
        }
    }

    /// Construct a failed result whose content is shown to the model verbatim.
    pub fn failure(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            display: None,
            success: false,
            touched: Vec::new(),
            exit_code: None,
        }
    }

    /// Attach terminal-only output while preserving model content.
    pub fn with_display(mut self, display: impl Into<String>) -> Self {
        self.display = Some(display.into());
        self
    }

    /// Record a path this tool touched.
    pub fn touching(mut self, path: TouchedPath) -> Self {
        self.touched.push(path);
        self
    }

    /// The facts recorded beside the content in history; `None` when the tool
    /// reported none, so a record without facts carries no metadata at all.
    pub fn metadata(&self) -> Option<MetadataRecord> {
        (!self.touched.is_empty() || self.exit_code.is_some()).then(|| MetadataRecord {
            touched: self.touched.clone(),
            exit_code: self.exit_code,
        })
    }
}
