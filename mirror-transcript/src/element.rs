//! Logical lines, wrapping and retained row caches.

use unicode_segmentation::UnicodeSegmentation as _;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::{protocol::block::Overflow, Content, Row, Source, Span};

#[derive(Debug)]
pub struct Element {
    pub id: u32,
    lines: Vec<Line>,
    pub content: Option<Content>,
    pub sticky: Option<Span>,
    content_width: u16,
    cache_bytes: usize,
    layout_bytes: usize,
    cache: Vec<Option<(u16, Vec<Row>)>>, // wrapped rows per logical line, keyed by width
    ends: Vec<usize>,                    // cumulative row counts; only a changed suffix is rebuilt
    count_width: u16,
}

pub type Line = Vec<Span>;

impl Element {
    pub fn lines(&self) -> &[Line] {
        &self.lines
    }

    pub fn cache_bytes(&self) -> usize {
        self.cache_bytes
            + self.layout_bytes
            + self.cache.capacity() * std::mem::size_of::<Option<(u16, Vec<Row>)>>()
            + self.ends.capacity() * std::mem::size_of::<usize>()
    }

    pub fn evict_layout(&mut self) {
        self.cache = Vec::new();
        self.ends = Vec::new();
        self.cache_bytes = 0;
        self.content_width = 0;
        if self.content.is_some() {
            self.lines = Vec::new();
            self.layout_bytes = 0;
        }
    }

    pub fn clear_cache(&mut self) {
        self.cache.iter_mut().for_each(|rows| *rows = None);
        self.cache_bytes = 0;
    }

    pub fn new(id: u32, lines: Vec<Line>) -> Element {
        Element {
            id,
            lines,
            content: None,
            sticky: None,
            content_width: 0,
            cache_bytes: 0,
            layout_bytes: 0,
            cache: Vec::new(),
            ends: Vec::new(),
            count_width: 0,
        }
    }
    pub fn from_content(id: u32, content: Content) -> Self {
        let mut element = Self::new(id, Vec::new());
        element.content = Some(content);
        element
    }

    pub fn prepare(&mut self, width: u16) {
        let width = width.clamp(1, 4096);
        if self.content_width == width {
            return;
        }
        if let Some(content) = &self.content {
            self.lines = content.render(width);
            self.layout_bytes = line_bytes(&self.lines)
                + (self.lines.capacity() - self.lines.len()) * std::mem::size_of::<Line>();
        }
        self.clear_cache();
        self.cache.resize_with(self.lines.len(), || None);
        self.cache.truncate(self.lines.len());
        self.ends.clear();
        self.content_width = width;
    }

    /// Append text (may contain '\n') with one style; only the touched lines lose their cache.
    pub fn append(&mut self, sgr: &str, text: &str) {
        self.cache.resize_with(self.lines.len(), || None);
        if self.lines.is_empty() {
            self.lines.push(vec![]);
            self.cache.push(None);
        }
        self.ends.truncate(self.lines.len() - 1);
        for (k, part) in text.split('\n').enumerate() {
            if k > 0 {
                self.lines.push(vec![]);
                self.cache.push(None);
            }
            let last = self.lines.len() - 1;
            if part.is_empty() {
                continue;
            }
            let line = &mut self.lines[last];
            match line.last_mut() {
                Some(s) if s.sgr == sgr => s.text.push_str(part),
                _ => line.push(Span {
                    action: None,
                    source: None,
                    sgr: sgr.into(),
                    text: part.into(),
                }),
            }
            self.cache_bytes -= self.cache[last]
                .as_ref()
                .map_or(0, |(_, rows)| rows.iter().map(Row::bytes).sum());
            match &mut self.cache[last] {
                // re-wrap only from the last row on
                Some((w, rows)) if rows.len() >= 2 => {
                    let from = rows
                        .pop()
                        .expect("cache has at least two rows in this branch")
                        .s as usize;
                    rows.extend(wrap_from(&self.lines[last], *w, true, from));
                }
                c => {
                    *c = None;
                }
            }
            self.cache_bytes += self.cache[last]
                .as_ref()
                .map_or(0, |(_, rows)| rows.iter().map(Row::bytes).sum());
        }
    }
    /// Index row counts without encoding lines that fit on one row.
    pub fn total(&mut self, width: u16) -> usize {
        let width = width.clamp(1, 4096);
        self.prepare(width);
        if self.count_width != width {
            self.ends.clear();
            self.count_width = width;
        }
        while self.ends.len() < self.lines.len() {
            let count = self.count1(self.ends.len(), width);
            self.ends
                .push(self.ends.last().copied().unwrap_or(0) + count);
        }
        self.ends.last().copied().unwrap_or(0)
    }

    /// Find the logical line and its first row without walking preceding lines.
    pub fn locate(&mut self, row: usize, width: u16) -> Option<(usize, usize)> {
        if row >= self.total(width) {
            return None;
        }
        let line = self.ends.partition_point(|&end| end <= row);
        Some((line, self.start(line)))
    }

    /// Resolve a source byte even when nested layout has changed its logical line.
    pub fn source_position(&self, source: Source) -> Option<(usize, u32)> {
        let (line, index, origin) = self
            .lines
            .iter()
            .enumerate()
            .flat_map(|(line, spans)| {
                spans.iter().enumerate().filter_map(move |(index, span)| {
                    span.source
                        .filter(|origin| origin.block == source.block)
                        .map(|origin| (line, index, origin))
                })
            })
            .min_by_key(|(_, _, origin)| {
                (
                    origin.offset > source.offset,
                    origin.offset.abs_diff(source.offset),
                )
            })?;
        let spans = &self.lines[line];
        let text = &spans[index].text;
        let mut bytes = (source.offset.saturating_sub(origin.offset) as usize).min(text.len());
        while !text.is_char_boundary(bytes) {
            bytes -= 1;
        }
        let character = spans[..index]
            .iter()
            .map(|span| span.text.chars().count())
            .sum::<usize>()
            + text[..bytes].chars().count();
        Some((line, character as u32))
    }

    pub fn start(&self, line: usize) -> usize {
        self.ends.get(line.wrapping_sub(1)).copied().unwrap_or(0)
    }
    fn count1(&mut self, i: usize, width: u16) -> usize {
        if let Some((w, r)) = &self.cache[i] {
            if *w == width {
                return r.len();
            }
        }
        let fits = ascii_width(&self.lines[i]);
        match fits {
            Some(w) if w <= width.max(1) as usize => 1,
            _ => wrap_output::<()>(&self.lines[i], width, true, 0, Overflow::Fold).len(),
        }
    }
    pub fn rows(&mut self, i: usize, width: u16) -> &[Row] {
        self.prepare(width);
        if i >= self.lines.len() {
            return &[];
        }
        let width = width.clamp(1, 4096);
        if !matches!(&self.cache[i], Some((w, _)) if *w == width) {
            self.cache_bytes -= self.cache[i]
                .as_ref()
                .map_or(0, |(_, rows)| rows.iter().map(Row::bytes).sum());
            let rows = wrap(&self.lines[i], width, true);
            self.cache_bytes += rows.iter().map(Row::bytes).sum::<usize>();
            self.cache[i] = Some((width, rows));
        }
        &self.cache[i]
            .as_ref()
            .expect("rows populated the cache above")
            .1
    }
}

pub fn line_bytes(lines: &[Line]) -> usize {
    lines
        .iter()
        .map(|line| {
            std::mem::size_of::<Line>()
                + (line.capacity() - line.len()) * std::mem::size_of::<Span>()
                + line.iter().map(Span::bytes).sum::<usize>()
        })
        .sum()
}

pub fn cw(c: char) -> u16 {
    if c.is_control() {
        0
    } else {
        c.width().unwrap_or(0) as u16
    }
}
pub fn width(s: &str) -> usize {
    UnicodeWidthStr::width(s)
}
/// Cell width of styled source text.
pub fn line_width(line: &[Span]) -> usize {
    width(
        &line
            .iter()
            .map(|span| span.text.as_str())
            .collect::<String>(),
    )
}

/// Clip without wrapping, preserving styles and complete graphemes.
pub fn clip_line(line: &[Span], columns: usize) -> Line {
    let text: String = line.iter().map(|span| span.text.as_str()).collect();
    let mut cells = 0;
    let mut remaining = 0;
    for grapheme in text.graphemes(true) {
        cells += width(grapheme);
        if cells > columns {
            break;
        }
        remaining += grapheme.len();
    }
    line.iter()
        .map_while(|span| {
            if remaining == 0 {
                return None;
            }
            let length = remaining.min(span.text.len());
            remaining -= length;
            Some(Span {
                action: span.action.clone(),
                source: span.source,
                text: span.text[..length].into(),
                sgr: span.sgr.clone(),
            })
        })
        .collect()
}

/// Clip a nonwrapping cell, keeping the final visible style on its ellipsis.
pub fn clip_cell(line: &[Span], columns: usize, overflow: Overflow) -> Line {
    let ellipsis = overflow == Overflow::Ellipsis && line_width(line) > columns && columns > 0;
    let mut clipped = clip_line(line, columns.saturating_sub(usize::from(ellipsis)));
    if ellipsis {
        clipped.push(Span {
            action: None,
            source: None,
            text: "…".into(),
            sgr: clipped
                .last()
                .or_else(|| line.first())
                .map_or(String::new(), |span| span.sgr.clone()),
        });
    }
    clipped
}

/// Wrap styled spans into rows (word wrap, or char wrap for the input box). Continuation rows get
/// a hanging indent equal to the line's leading spaces. Control characters are dropped.
pub fn wrap(line: &Line, width: u16, words: bool) -> Vec<Row> {
    wrap_from(line, width, words, 0)
}
/// Wrap from char `from` on, which must start a row (used to re-wrap only the tail on append).
fn wrap_from(line: &Line, width: u16, words: bool, from: usize) -> Vec<Row> {
    wrap_output(line, width, words, from, Overflow::Fold)
}

/// Use the terminal wrapper for nested logical surfaces without encoding ANSI.
pub fn wrap_lines(line: &Line, width: u16, words: bool) -> Vec<Line> {
    wrap_output(line, width, words, 0, Overflow::Fold)
}

/// Word-wrap a cell, applying its overflow policy to words wider than the cell.
pub fn wrap_cell(line: &Line, width: u16, overflow: Overflow) -> Vec<Line> {
    wrap_output(line, width, true, 0, overflow)
}

type Slice<'a> = (&'a Span, &'a str, Option<Source>);

// Both projections use exactly the same grapheme boundaries and whitespace rules.
trait Wrapped {
    fn spans<'a>(spans: impl Iterator<Item = Slice<'a>>, width: u16, start: u32) -> Self;
}

impl Wrapped for () {
    fn spans<'a>(_: impl Iterator<Item = Slice<'a>>, _: u16, _: u32) {}
}

impl Wrapped for Row {
    fn spans<'a>(spans: impl Iterator<Item = Slice<'a>>, width: u16, start: u32) -> Self {
        encode(spans, width, start)
    }
}

impl Wrapped for Line {
    fn spans<'a>(spans: impl Iterator<Item = Slice<'a>>, _: u16, _: u32) -> Self {
        spans
            .map(|(span, text, source)| Span {
                text: text.into(),
                source,
                ..span.clone()
            })
            .collect()
    }
}

fn wrap_output<T: Wrapped>(
    line: &Line,
    width: u16,
    words: bool,
    from: usize,
    overflow: Overflow,
) -> Vec<T> {
    let width = width.clamp(1, 4096);
    if let Some(cells) = ascii_width(line).filter(|&cells| from == 0 && cells <= usize::from(width))
    {
        return vec![T::spans(
            line.iter()
                .map(|span| (span, span.text.as_str(), span.source)),
            cells as u16,
            0,
        )];
    }
    let indent = if words { hang(line, width) } else { 0 };
    // Segment across style boundaries so a combining mark cannot become a new cell.
    type Glyph<'a> = (Vec<Slice<'a>>, u16, u32);
    type Token<'a> = (Vec<Glyph<'a>>, u16, bool);
    let text: String = line.iter().map(|span| span.text.as_str()).collect();
    let (mut span_index, mut span_start, mut character) = (0, 0, 0);
    let mut graphemes = text.grapheme_indices(true).peekable();
    // Retain only the current word, not every glyph of a large paragraph.
    let toks = std::iter::from_fn(|| {
        let mut token: Option<Token<'_>> = None;
        while let Some(&(offset, grapheme)) = graphemes.peek() {
            let space = grapheme == " ";
            if token
                .as_ref()
                .is_some_and(|(_, _, previous)| !words || *previous != space)
            {
                break;
            }
            graphemes.next();
            let count = grapheme.chars().count();
            character += count;
            if character <= from {
                continue;
            }
            let end = offset + grapheme.len();
            let mut parts = Vec::new();
            while span_index < line.len() && span_start < end {
                let span = &line[span_index];
                let span_end = span_start + span.text.len();
                if span_end > offset {
                    parts.push((
                        span,
                        &text[offset.max(span_start)..end.min(span_end)],
                        span.source
                            .map(|source| source.shifted(offset.max(span_start) - span_start)),
                    ));
                }
                if span_end >= end {
                    break;
                }
                span_start = span_end;
                span_index += 1;
            }
            let cells = if grapheme.chars().all(char::is_control) {
                0
            } else {
                UnicodeWidthStr::width(grapheme).min(u16::MAX as usize) as u16
            };
            let (glyphs, cells_so_far, _) = token.get_or_insert_with(|| (Vec::new(), 0, space));
            glyphs.push((parts, cells, count as u32));
            *cells_so_far = cells_so_far.saturating_add(cells);
        }
        token
    });
    let mut rows = Vec::new();
    let padding = Span::default();
    let mut segs: Vec<(&Span, String, Option<Source>)> = Vec::new();
    let (mut w, mut ci, mut st): (u16, u32, Option<u32>) = (0, from as u32, None); // ci: char index, st: row start
    if from > 0 && indent > 0 {
        segs.push((&padding, " ".repeat(indent as usize), None));
        w = indent;
    }
    let flush =
        |segs: &mut Vec<(_, String, Option<Source>)>, w: &mut u16, rows: &mut Vec<T>, s: u32| {
            rows.push(T::spans(
                segs.iter()
                    .map(|(style, text, source)| (*style, text.as_str(), *source)),
                *w,
                s,
            ));
            segs.clear();
            *w = 0;
            if indent > 0 {
                segs.push((&padding, " ".repeat(indent as usize), None));
                *w = indent;
            }
        };
    for (mut tok, mut tw, space) in toks {
        if !space && tw > width && overflow != Overflow::Fold {
            let ellipsis = overflow == Overflow::Ellipsis;
            let limit = width - u16::from(ellipsis);
            let count = tok.iter().map(|glyph| glyph.2).sum::<u32>();
            let style = tok
                .first()
                .and_then(|glyph| glyph.0.first())
                .map_or(&padding, |part| part.0);
            tw = 0;
            let keep = tok
                .iter()
                .take_while(|glyph| {
                    if tw + glyph.1 > limit {
                        return false;
                    }
                    tw += glyph.1;
                    true
                })
                .count();
            tok.truncate(keep);
            let hidden = count - tok.iter().map(|glyph| glyph.2).sum::<u32>();
            if ellipsis {
                tok.push((vec![(style, "…", None)], 1, hidden));
                tw += 1;
            } else if let Some(last) = tok.last_mut() {
                last.2 += hidden;
            }
        }
        if w.saturating_add(tw) > width && w > indent && !space {
            flush(&mut segs, &mut w, &mut rows, st.take().unwrap_or(ci));
        }
        if words && space && w.saturating_add(tw) > width {
            ci += tok.iter().map(|glyph| glyph.2).sum::<u32>();
            continue;
        }
        for (parts, cells, count) in tok {
            let too_wide = cells > width;
            let cells = if too_wide { 1 } else { cells };
            if w + cells > width {
                flush(&mut segs, &mut w, &mut rows, st.take().unwrap_or(ci));
            }
            st.get_or_insert(ci);
            for (sgr, text, source) in parts
                .into_iter()
                .take(if too_wide { 1 } else { usize::MAX })
            {
                let text = if too_wide {
                    "�".into()
                } else {
                    text.chars()
                        .filter(|character| !character.is_control())
                        .collect::<String>()
                };
                match segs.last_mut() {
                    Some((style, previous, start))
                        if style.sgr == sgr.sgr
                            && style.action == sgr.action
                            && start.map(|start| start.shifted(previous.len())) == source =>
                    {
                        previous.push_str(&text)
                    }
                    _ => segs.push((sgr, text, source)),
                }
            }
            w += cells;
            ci += count;
        }
    }
    rows.push(T::spans(
        segs.iter()
            .map(|(style, text, source)| (*style, text.as_str(), *source)),
        w,
        st.unwrap_or(ci),
    ));
    rows.shrink_to_fit();
    rows
}

/// Hanging indent of continuation rows: the line's leading spaces, at most half the width.
pub fn hang(line: &Line, width: u16) -> u16 {
    line.iter()
        .flat_map(|s| s.text.chars())
        .take_while(|c| *c == ' ')
        .take(usize::from(width / 2))
        .count() as u16
}
pub fn plain(line: &Line) -> Vec<char> {
    line.iter().flat_map(|s| s.text.chars()).collect()
}
fn ascii_width(line: &Line) -> Option<usize> {
    line.iter().try_fold(0, |width, span| {
        span.text
            .bytes()
            .all(|byte| (0x20..0x7f).contains(&byte))
            .then_some(width + span.text.len())
    })
}

fn encode<'a>(spans: impl Iterator<Item = Slice<'a>>, w: u16, s: u32) -> Row {
    let mut bytes = Vec::new();
    let mut plain = String::new();
    let mut actions = Vec::new();
    let mut column = 0;
    let mut source = None;
    for (span, text, origin) in spans {
        source = source.or(origin);
        if text.is_empty() {
            continue;
        }
        let end = column + text.width();
        if let Some(action) = &span.action {
            actions.push(crate::Hit {
                start: column,
                end,
                action: action.clone(),
            });
            if let crate::Action::Link(url) = action.as_ref() {
                bytes.extend_from_slice(format!("\x1b]8;;{url}\x1b\\").as_bytes());
            }
        }
        if !span.sgr.is_empty() {
            bytes.extend_from_slice(format!("\x1b[{}m", span.sgr).as_bytes());
        }
        bytes.extend_from_slice(text.as_bytes());
        if !span.sgr.is_empty() {
            bytes.extend_from_slice(b"\x1b[0m");
        }
        if span
            .action
            .as_deref()
            .is_some_and(|action| matches!(action, crate::Action::Link(_)))
        {
            bytes.extend_from_slice(b"\x1b]8;;\x1b\\");
        }
        plain.push_str(text);
        column = end;
    }
    Row {
        b: bytes.into(),
        w,
        s,
        source,
        text: plain.into(),
        actions: actions.into(),
        selection: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    #[test]
    fn clip_preserves_graphemes_across_style_boundaries() {
        let source = vec![
            Span {
                action: None,
                source: None,
                text: "a👩".into(),
                sgr: "31".into(),
            },
            Span {
                action: None,
                source: None,
                text: "🏽‍💻b".into(),
                sgr: "32".into(),
            },
        ];
        assert_eq!(
            clip_line(&source, 2),
            vec![Span {
                action: None,
                source: None,
                text: "a".into(),
                sgr: "31".into()
            }]
        );
        assert_eq!(
            clip_line(&source, 3),
            vec![
                source[0].clone(),
                Span {
                    action: None,
                    source: None,
                    text: "🏽‍💻".into(),
                    sgr: "32".into()
                },
            ]
        );
    }

    #[test]
    fn cell_ellipsizes_long_words_without_folding_them() {
        let source = vec![Span {
            action: None,
            source: None,
            text: "Turn in flight".into(),
            sgr: "31".into(),
        }];
        let rows = wrap_cell(&source, 4, Overflow::Ellipsis);
        let text: Vec<String> = rows
            .iter()
            .map(|line| line.iter().map(|span| span.text.as_str()).collect())
            .collect();
        assert_eq!(text, ["Turn", "in ", "fli…"]);
        assert_eq!(rows[2][0].sgr, "31");
    }

    #[test]
    fn wrap_keeps_a_styled_emoji_cluster_on_one_row() {
        let rows = wrap(
            &vec![
                Span {
                    action: None,
                    source: None,
                    text: "abc👩".into(),
                    sgr: String::new(),
                },
                Span {
                    action: None,
                    source: None,
                    text: "🏽‍💻!".into(),
                    sgr: "31".into(),
                },
            ],
            4,
            false,
        );
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].w, 3);
        assert_eq!(rows[1].w, 3);
        assert_eq!(rows[1].s, 3);
        assert_eq!(rows[1].b.as_ref(), "👩\x1b[31m🏽‍💻!\x1b[0m".as_bytes());
    }

    #[test]
    fn wrap_drops_terminal_controls_without_changing_geometry() {
        let rows = wrap(
            &vec![Span {
                action: None,
                source: None,
                text: "ab\x1b\x7f\u{85}\u{9b}cde\u{301}".into(),
                sgr: String::new(),
            }],
            4,
            false,
        );
        assert_matches!(rows.as_slice(), [first, second] if first.w == 4 && second.w == 1);
        assert_eq!(rows[0].b.as_ref(), b"abcd");
        assert_eq!(rows[1].b.as_ref(), "e\u{301}".as_bytes());
    }

    #[test]
    fn wrap_preserves_styles_across_word_boundaries() {
        let rows = wrap(
            &vec![
                Span {
                    action: None,
                    source: None,
                    text: "ab cd".into(),
                    sgr: "32".into(),
                },
                Span {
                    action: None,
                    source: None,
                    text: "ef".into(),
                    sgr: "1;34".into(),
                },
            ],
            4,
            true,
        );
        assert_matches!(rows.as_slice(), [first, second] if first.w == 3 && second.w == 4);
        assert_eq!(rows[0].b.as_ref(), b"\x1b[32mab \x1b[0m");
        assert_eq!(rows[1].b.as_ref(), b"\x1b[32mcd\x1b[0m\x1b[1;34mef\x1b[0m");
    }

    #[test]
    fn character_wrap_preserves_spaces_at_the_margin() {
        let rows = wrap(
            &vec![Span {
                action: None,
                source: None,
                text: "ab c".into(),
                sgr: String::new(),
            }],
            2,
            false,
        );
        assert_matches!(rows.as_slice(), [first, second] if first.w == 2 && second.w == 2);
        assert_eq!(rows[0].b.as_ref(), b"ab");
        assert_eq!(rows[1].b.as_ref(), b" c");
    }
}
