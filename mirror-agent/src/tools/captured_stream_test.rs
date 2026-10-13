use assert_matches::assert_matches;
use pretty_assertions::assert_eq;

use super::*;
use crate::tools::stream_capture::StreamCapture;

/// The in-memory reading the capture must reproduce: drop leading and
/// trailing blank lines, keeping inner whitespace.
fn strip_empty_lines(content: &str) -> String {
    let is_text = |character: char| !character.is_whitespace();
    let (Some(first), Some(last)) = (content.find(is_text), content.rfind(is_text)) else {
        return String::new();
    };
    let start = content[..first]
        .rfind('\n')
        .map_or(0, |newline| newline + 1);
    let end = content[last..]
        .find('\n')
        .map_or(content.len(), |newline| last + newline);
    content[start..end].to_string()
}

/// Capture `bytes` read in chunks of `chunk` bytes.
fn captured(bytes: &[u8], chunk: usize, max_kept_chars: usize) -> CapturedStream {
    let mut capture = StreamCapture::new(max_kept_chars, vec!["Permission denied"]);
    for piece in bytes.chunks(chunk) {
        capture.push(piece);
    }
    capture.finish()
}

const STREAMS: &[&[u8]] = &[
    b"",
    b" \n\t\n",
    b"\n\n  first line\n\n inner\n last  \n\n\t\n",
    b"no newline at all",
    b"   leading spaces kept\ntrailing\n   ",
    "caf\u{e9} \u{1f600} na\u{ef}ve\n\n".as_bytes(),
    b"cut \xe2\x82 sequence \xff and \xf0\x9f\x98 end \xe2",
];

#[test]
fn stripped_streams_read_as_the_whole_stream_reads_in_any_chunking() {
    for stream in STREAMS {
        let lossy = String::from_utf8_lossy(stream);
        let expected = strip_empty_lines(&lossy);
        for chunk in [1, 2, 3, 7, 4096] {
            let mut capture = captured(stream, chunk, 1 << 20);

            let stripped = capture.stripped().expect("read kept text");

            assert_eq!(stripped.kept, expected, "chunk {chunk} of {lossy:?}");
            assert_eq!(stripped.chars, expected.chars().count());
            assert_eq!(stripped.newlines, expected.matches('\n').count());
            assert_eq!(capture.raw_chars, lossy.chars().count());
            assert_eq!(capture.raw_head(), lossy, "chunk {chunk} of {lossy:?}");
        }
    }
}

#[test]
fn a_suffix_keeps_the_trailing_blank_lines_before_it() {
    let suffix = "\n[Sandbox note: denied]";
    for stream in STREAMS {
        let lossy = String::from_utf8_lossy(stream);
        let expected = strip_empty_lines(&format!("{lossy}{suffix}"));

        let stripped = captured(stream, 3, 1 << 20)
            .stripped_with_suffix(suffix)
            .expect("read kept text");

        assert_eq!(stripped.kept, expected, "{lossy:?}");
        assert_eq!(stripped.chars, expected.chars().count());
        assert_eq!(stripped.newlines, expected.matches('\n').count());
    }
}

#[test]
fn text_past_the_cap_keeps_a_prefix_and_its_whole_counts() {
    let stream = "\n\nhead\nmiddle\ntail\n\n";

    let stripped = captured(stream.as_bytes(), 2, 6)
        .stripped()
        .expect("read kept text");

    assert_eq!(stripped.kept, "head\nm");
    assert_eq!(stripped.chars, "head\nmiddle\ntail".len());
    assert_eq!(stripped.newlines, 2);
    assert!(!stripped.is_whole());
}

#[test]
fn a_long_stream_keeps_its_ends_for_a_timed_out_result() {
    let stream = format!(
        "{}{}{}",
        "h".repeat(RAW_END_CHARS),
        "m".repeat(10),
        "t".repeat(RAW_END_CHARS)
    );

    let capture = captured(stream.as_bytes(), 1000, 8);

    assert_eq!(capture.raw_chars, 2 * RAW_END_CHARS + 10);
    assert_eq!(capture.raw_head(), "h".repeat(RAW_END_CHARS));
    assert_eq!(capture.raw_tail, "t".repeat(RAW_END_CHARS));
}

#[test]
fn a_needle_split_across_reads_is_found() {
    let capture = captured(b"ls: cannot open: Permission denied\n", 5, 8);

    assert_matches!(capture.first_needle(), Some(0));
}

#[test]
fn a_stream_without_the_needle_finds_none() {
    assert_matches!(captured(b"Permission granted", 4, 8).first_needle(), None);
}
