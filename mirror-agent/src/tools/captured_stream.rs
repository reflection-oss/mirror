//! A finished capture of one command output stream.

use super::capture_storage::CaptureStorage;
use super::partial_text::PartialText;

/// Characters of each end of the stream kept for a timed-out result.
pub(super) const RAW_END_CHARS: usize = 2_048;

/// A whole stream's capture: its ends, its counts and its kept text, as
/// `StreamCapture::finish` records them.
#[derive(Debug)]
pub(crate) struct CapturedStream {
    /// Whether the stream wrote nothing.
    pub(super) is_empty: bool,
    /// The stream's length in characters, as `from_utf8_lossy` decodes it.
    pub(super) raw_chars: usize,
    /// The stream's first `2 * RAW_END_CHARS` characters.
    pub(super) raw_head: String,
    /// The stream's last `RAW_END_CHARS` characters.
    pub(super) raw_tail: String,
    /// Which of the capture's needles the stream contains.
    pub(super) found: Vec<bool>,
    /// Whether the stream has a non-blank character.
    pub(super) has_text: bool,
    /// Characters from the start of the kept text to its stripped end.
    pub(super) text_chars: usize,
    /// Line breaks from the start of the kept text to its stripped end.
    pub(super) text_newlines: usize,
    /// Characters from the start of the kept text to the stream's end.
    pub(super) rest_chars: usize,
    /// Line breaks from the start of the kept text to the stream's end.
    pub(super) rest_newlines: usize,
    /// The most characters of text kept.
    pub(super) max_kept_chars: usize,
    /// The kept text.
    pub(super) storage: CaptureStorage,
}

impl CapturedStream {
    /// Capture `bytes` as one stream that wrote them in a single read.
    pub(super) fn from_bytes(
        bytes: &[u8],
        max_kept_chars: usize,
        needles: Vec<&'static str>,
    ) -> Self {
        let mut capture = super::stream_capture::StreamCapture::new(max_kept_chars, needles);
        capture.push(bytes);
        capture.finish()
    }

    /// The whole stream when it is at most `2 * RAW_END_CHARS` characters,
    /// otherwise its first `RAW_END_CHARS`.
    pub(super) fn raw_head(&self) -> &str {
        if self.raw_chars <= 2 * RAW_END_CHARS {
            return &self.raw_head;
        }
        let end = self
            .raw_head
            .char_indices()
            .nth(RAW_END_CHARS)
            .map_or(self.raw_head.len(), |(index, _)| index);
        &self.raw_head[..end]
    }

    /// The index of the first needle the stream contains.
    pub(super) fn first_needle(&self) -> Option<usize> {
        self.found.iter().position(|found| *found)
    }

    /// The stream as `strip_empty_lines` leaves it.
    ///
    /// # Errors
    ///
    /// Returns an error when the kept text cannot be read or was lost.
    pub(super) fn stripped(&mut self) -> std::io::Result<PartialText> {
        if !self.has_text {
            return Ok(PartialText::whole(String::new()));
        }
        Ok(PartialText {
            kept: self.kept_prefix(self.text_chars)?,
            chars: self.text_chars,
            newlines: self.text_newlines,
        })
    }

    /// The stream with `suffix` appended, as `strip_empty_lines` leaves it;
    /// `suffix` must end in a non-blank character, so only leading blank lines
    /// are dropped.
    ///
    /// # Errors
    ///
    /// Returns an error when the kept text cannot be read or was lost.
    pub(super) fn stripped_with_suffix(&mut self, suffix: &str) -> std::io::Result<PartialText> {
        if !self.has_text {
            return Ok(PartialText::whole(strip_leading_lines(suffix)));
        }
        let rest = PartialText {
            kept: self.kept_prefix(self.rest_chars)?,
            chars: self.rest_chars,
            newlines: self.rest_newlines,
        };
        Ok(rest.concat(PartialText::whole(suffix.to_string())))
    }

    fn kept_prefix(&mut self, chars: usize) -> std::io::Result<String> {
        let text = String::from_utf8(self.storage.read_all()?)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        let chars = chars.min(self.max_kept_chars);
        let end = text
            .char_indices()
            .nth(chars)
            .map_or(text.len(), |(index, _)| index);
        Ok(text[..end].to_string())
    }
}

/// Drop the lines before the first non-blank character.
fn strip_leading_lines(text: &str) -> String {
    let Some(first) = text.find(|character: char| !character.is_whitespace()) else {
        return String::new();
    };
    let start = text[..first].rfind('\n').map_or(0, |newline| newline + 1);
    text[start..].to_string()
}

#[cfg(test)]
#[path = "captured_stream_test.rs"]
mod tests;

impl Default for CapturedStream {
    fn default() -> Self {
        Self::from_bytes(&[], 0, Vec::new())
    }
}
