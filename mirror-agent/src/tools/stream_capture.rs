//! A command output stream captured to disk as it arrives.
//!
//! A command can write without end, so a stream is never held whole. Its text
//! from the line of its first non-blank character is kept in `CaptureStorage`
//! up to the character cap the formatted result is cut to anyway; past that
//! cap only counts are kept. With the stream's first and last few thousand
//! characters, kept in memory for a timed-out result, that is everything the
//! bash result needs to read exactly as it would with the whole stream in
//! memory.

use std::collections::VecDeque;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use super::capture_storage::CaptureStorage;
use super::captured_stream::CapturedStream;
use super::captured_stream::RAW_END_CHARS;

/// How much text is buffered before it is written to storage.
const WRITE_CHUNK_BYTES: usize = 64 * 1024;
/// Reads queued for a capture before the reader waits for it to catch up.
const QUEUED_READS: usize = 16;

/// A stream being captured, fed its bytes in the order the command wrote them.
pub(super) struct StreamCapture {
    max_kept_chars: usize,
    needles: Vec<&'static str>,
    found: Vec<bool>,
    window: String,
    undecoded: Vec<u8>,
    raw_bytes: u64,
    raw_chars: usize,
    raw_head: String,
    raw_tail: VecDeque<char>,
    storage: CaptureStorage,
    pending: Vec<u8>,
    seen_text: bool,
    kept_chars: usize,
    kept_newlines: usize,
    text_newlines: usize,
    text_end: Option<usize>,
}

impl StreamCapture {
    /// A capture that keeps at most `max_kept_chars` of stripped text and
    /// records which of `needles` the stream contains.
    pub(super) fn new(max_kept_chars: usize, needles: Vec<&'static str>) -> Self {
        Self {
            max_kept_chars,
            found: vec![false; needles.len()],
            needles,
            window: String::new(),
            undecoded: Vec::new(),
            raw_bytes: 0,
            raw_chars: 0,
            raw_head: String::new(),
            raw_tail: VecDeque::with_capacity(RAW_END_CHARS),
            storage: CaptureStorage::new(),
            pending: Vec::new(),
            seen_text: false,
            kept_chars: 0,
            kept_newlines: 0,
            text_newlines: 0,
            text_end: None,
        }
    }

    /// Run a new capture on the blocking pool, off the runtime its reader is
    /// on, since it decodes every character and writes to disk.
    ///
    /// The capture takes the reads sent to the returned sender, in order; the
    /// handle yields it finished once every sender is dropped.
    pub(super) fn spawn(
        max_kept_chars: usize,
        needles: Vec<&'static str>,
    ) -> (mpsc::Sender<Vec<u8>>, JoinHandle<CapturedStream>) {
        let (sender, mut receiver) = mpsc::channel::<Vec<u8>>(QUEUED_READS);
        let handle = tokio::task::spawn_blocking(move || {
            let mut capture = Self::new(max_kept_chars, needles);
            while let Some(read) = receiver.blocking_recv() {
                capture.push(&read);
            }
            capture.finish()
        });
        (sender, handle)
    }

    /// Record the next bytes the stream produced.
    pub(super) fn push(&mut self, bytes: &[u8]) {
        self.raw_bytes += bytes.len() as u64;
        let mut input = std::mem::take(&mut self.undecoded);
        input.extend_from_slice(bytes);
        let mut chunks = input.utf8_chunks().peekable();
        while let Some(chunk) = chunks.next() {
            self.push_text(chunk.valid());
            if chunk.invalid().is_empty() {
                continue;
            }
            if chunks.peek().is_none() {
                // Invalid bytes that end the read may be a character cut in
                // two, so they are decoded with the bytes that follow; decoded
                // either way, they read as `from_utf8_lossy` reads the whole.
                self.undecoded = chunk.invalid().to_vec();
            } else {
                self.push_text("\u{FFFD}");
            }
        }
    }

    /// The finished capture, once the stream has ended.
    pub(super) fn finish(mut self) -> CapturedStream {
        let undecoded = std::mem::take(&mut self.undecoded);
        if !undecoded.is_empty() {
            self.push_text(&String::from_utf8_lossy(&undecoded));
        }
        self.flush();
        let text_chars = if self.seen_text {
            self.text_end.unwrap_or(self.kept_chars)
        } else {
            0
        };
        CapturedStream {
            is_empty: self.raw_bytes == 0,
            raw_chars: self.raw_chars,
            raw_head: self.raw_head,
            raw_tail: self.raw_tail.into_iter().collect(),
            found: self.found,
            has_text: self.seen_text,
            text_chars,
            text_newlines: self.text_newlines,
            rest_chars: self.kept_chars,
            rest_newlines: self.kept_newlines,
            max_kept_chars: self.max_kept_chars,
            storage: self.storage,
        }
    }

    fn push_text(&mut self, text: &str) {
        self.record_needles(text);
        for character in text.chars() {
            self.push_raw(character);
            self.push_kept(character);
        }
        if self.pending.len() >= WRITE_CHUNK_BYTES {
            self.flush();
        }
    }

    fn record_needles(&mut self, text: &str) {
        if self.needles.is_empty() {
            return;
        }
        self.window.push_str(text);
        for (needle, found) in self.needles.iter().zip(self.found.iter_mut()) {
            *found |= self.window.contains(needle);
        }
        // Keep only the end a needle split across reads could start in.
        let keep = self
            .needles
            .iter()
            .map(|needle| needle.len())
            .max()
            .unwrap_or(0);
        if self.window.len() > keep {
            let mut start = self.window.len() - keep;
            while !self.window.is_char_boundary(start) {
                start -= 1;
            }
            self.window.drain(..start);
        }
    }

    fn push_raw(&mut self, character: char) {
        self.raw_chars += 1;
        if self.raw_chars <= 2 * RAW_END_CHARS {
            self.raw_head.push(character);
        }
        if self.raw_tail.len() == RAW_END_CHARS {
            self.raw_tail.pop_front();
        }
        self.raw_tail.push_back(character);
    }

    /// Track the stream as `strip_empty_lines` reads it: the kept text starts
    /// on the line of the first non-blank character, and its stripped end is
    /// the line break after the last one.
    fn push_kept(&mut self, character: char) {
        if !self.seen_text {
            if character == '\n' {
                self.pending.clear();
                self.storage.clear();
                self.kept_chars = 0;
                return;
            }
            self.seen_text = !character.is_whitespace();
        }
        if self.kept_chars < self.max_kept_chars {
            self.pending
                .extend_from_slice(character.encode_utf8(&mut [0; 4]).as_bytes());
        }
        if character == '\n' {
            self.text_end.get_or_insert(self.kept_chars);
            self.kept_newlines += 1;
        } else if !character.is_whitespace() {
            self.text_end = None;
            self.text_newlines = self.kept_newlines;
        }
        self.kept_chars += 1;
    }

    fn flush(&mut self) {
        if !self.pending.is_empty() {
            self.storage.write(&self.pending);
            self.pending.clear();
        }
    }
}
