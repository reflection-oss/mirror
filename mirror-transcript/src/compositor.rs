//! One writer composes Python chrome with an independently moving transcript.

use std::io::{self, Write};

use crate::{invalid, protocol, validate_style, Row, Viewport};

/// Retained terminal shadow; Python chrome is never reconstructed from policy.
#[derive(Debug, Default)]
pub struct Compositor {
    frame: Option<protocol::ChromeFrame>,
    chrome: Vec<Row>,
    body: Vec<Row>,
    previous_chrome: Vec<Row>,
    previous_body: Vec<Row>,
    thumb: Vec<Option<String>>,
    previous_thumb: Vec<Option<String>>,
    rectangle: Option<protocol::TranscriptRect>,
}

impl Compositor {
    pub fn frame(&self) -> Option<&protocol::ChromeFrame> {
        self.frame.as_ref()
    }

    pub fn set_frame(&mut self, frame: protocol::ChromeFrame) -> io::Result<()> {
        validate_frame(&frame)?;
        let size = frame
            .size
            .as_ref()
            .ok_or_else(|| invalid("chrome frame needs dimensions"))?;
        let chrome: Vec<_> = frame
            .rows
            .iter()
            .map(|line| Row {
                b: line.clone().into(),
                w: size.columns as u16,
                ..Default::default()
            })
            .collect();
        if self.frame.as_ref().is_none_or(|previous| {
            previous.size != frame.size || previous.overlay_rows != frame.overlay_rows
        }) {
            self.invalidate();
        }
        self.chrome = chrome;
        self.frame = Some(frame);
        Ok(())
    }

    pub fn invalidate(&mut self) {
        self.previous_chrome.clear();
        self.previous_body.clear();
        self.previous_thumb.clear();
        self.rectangle = None;
    }

    /// Paint changed rows, marking the transcript's position on its right edge
    /// in the `thumb` SGR style when given.
    pub fn present(
        &mut self,
        output: &mut impl Write,
        viewport: Option<&mut Viewport>,
        thumb: Option<&str>,
    ) -> io::Result<()> {
        let Some(frame) = &self.frame else {
            return Ok(());
        };
        let Some(size) = frame.size else {
            return Ok(());
        };
        self.body = Vec::new();
        self.thumb = Vec::new();
        if let (Some(rect), Some(viewport)) = (frame.transcript.as_ref(), viewport) {
            let width = rect.width as u16;
            self.body = viewport
                .rows(width, rect.height as u16)
                .into_iter()
                .map(|row| row.on_canvas(width, &frame.canvas))
                .collect();
            if let (Some(style), Some((share, offset))) = (thumb, viewport.scroll_position()) {
                self.thumb = thumb_column(&self.body, width, style, share, offset);
            }
        }
        let changed_rectangle = self.rectangle != frame.transcript;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x1b[?2026h\x1b[?25l\x1b[?7l");
        for (y, row) in self.chrome.iter().enumerate() {
            if changed_rectangle || self.previous_chrome.get(y) != Some(row) {
                write_row(&mut bytes, 0, y, size.columns as usize, row)?;
            }
        }
        if let Some(rect) = &frame.transcript {
            for (y, row) in self.body.iter().enumerate() {
                let screen_y = y + rect.y as usize;
                let thumb = self.thumb.get(y).and_then(Option::as_ref);
                if changed_rectangle
                    || self.previous_body.get(y) != Some(row)
                    || self.previous_thumb.get(y).and_then(Option::as_ref) != thumb
                    || self.previous_chrome.get(screen_y) != self.chrome.get(screen_y)
                {
                    write_row(
                        &mut bytes,
                        rect.x as usize,
                        screen_y,
                        rect.width as usize,
                        row,
                    )?;
                    if let Some(thumb) = thumb {
                        write!(
                            bytes,
                            "\x1b[{};{}H{thumb}",
                            screen_y + 1,
                            (rect.x + rect.width) as usize
                        )?;
                    }
                    if let Some(overlay) = frame.overlay_rows.get(screen_y) {
                        bytes.extend_from_slice(overlay);
                    }
                }
            }
        }
        write!(
            bytes,
            "\x1b[0m\x1b[{};{}H\x1b[{} q\x1b[?7h",
            frame.cursor_y + 1,
            frame.cursor_x + 1,
            frame.cursor_shape
        )?;
        if frame.cursor_visible {
            bytes.extend_from_slice(b"\x1b[?25h");
        }
        bytes.extend_from_slice(b"\x1b[?2026l");
        if let Err(error) = output.write_all(&bytes).and_then(|()| output.flush()) {
            self.invalidate();
            return Err(error);
        }
        self.previous_chrome.clone_from(&self.chrome);
        self.previous_body.clone_from(&self.body);
        self.previous_thumb.clone_from(&self.thumb);
        self.rectangle.clone_from(&frame.transcript);
        Ok(())
    }
}

const EIGHTHS: usize = 8;
const LOWER_BLOCKS: [char; EIGHTHS + 1] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// Encoded thumb cells for each row's last column, for a thumb in the `style`
/// SGR `share` of the track long and `offset` of the way down it.
///
/// Ends are drawn in eighths of a row so the thumb glides rather than jumps,
/// and each cell keeps its row's background.
fn thumb_column(
    rows: &[Row],
    width: u16,
    style: &str,
    share: f64,
    offset: f64,
) -> Vec<Option<String>> {
    let track = rows.len() * EIGHTHS;
    let least = (2 * EIGHTHS).max(track / 16).min(track);
    let size = ((track as f64 * share).round() as usize).clamp(least, track);
    let top = ((track - size) as f64 * offset).round() as usize;
    rows.iter()
        .enumerate()
        .map(|(y, row)| {
            let start = top.max(y * EIGHTHS) - y * EIGHTHS;
            let end = (top + size)
                .min((y + 1) * EIGHTHS)
                .saturating_sub(y * EIGHTHS);
            if start >= end {
                return None;
            }
            // Lower blocks fill a cell from its bottom: the thumb's top end.
            // Reversed, one paints the background below the thumb's bottom
            // end; a full cell is the empty block reversed.
            let (glyph, reverse) = if start > 0 {
                (LOWER_BLOCKS[EIGHTHS - start], "")
            } else {
                (LOWER_BLOCKS[EIGHTHS - end], "7")
            };
            let background = if row.w < width {
                String::new()
            } else {
                background(&row.b)
            };
            let sgr = ["0", &background, style, reverse]
                .into_iter()
                .filter(|parameter| !parameter.is_empty())
                .collect::<Vec<_>>()
                .join(";");
            Some(format!("\x1b[{sgr}m{glyph}\x1b[0m"))
        })
        .collect()
}

/// SGR parameters for the background of an encoded row's last printed cell,
/// so the thumb sits on code fills.
fn background(row: &[u8]) -> String {
    let mut current = String::new();
    let mut last = String::new();
    let mut index = 0;
    while index < row.len() {
        if row[index] != 0x1b {
            // Continuation bytes belong to the character already counted.
            if row[index] >= 0x20 && row[index] & 0xc0 != 0x80 {
                last.clone_from(&current);
            }
            index += 1;
            continue;
        }
        let rest = &row[index + 2.min(row.len() - index)..];
        match row.get(index + 1) {
            Some(b'[') => {
                let Some(end) = rest.iter().position(|byte| (0x40..=0x7e).contains(byte)) else {
                    break;
                };
                if rest[end] == b'm' {
                    set_background(&mut current, &rest[..end]);
                }
                index += 2 + end + 1;
            }
            Some(b']') => {
                let Some(end) = rest.windows(2).position(|pair| pair == b"\x1b\\") else {
                    break;
                };
                index += 2 + end + 2;
            }
            _ => index += 2,
        }
    }
    last
}

fn set_background(background: &mut String, parameters: &[u8]) {
    if parameters.is_empty() {
        background.clear();
        return;
    }
    let mut parameters = std::str::from_utf8(parameters).unwrap_or("").split(';');
    while let Some(parameter) = parameters.next() {
        match parameter {
            "" | "0" | "49" => background.clear(),
            "38" => {
                let skip = match parameters.next() {
                    Some("5") => 1,
                    Some("2") => 3,
                    _ => 0,
                };
                parameters.by_ref().take(skip).for_each(drop);
            }
            "48" => {
                let count = match parameters.next() {
                    Some("5") => 1,
                    Some("2") => 3,
                    _ => continue,
                };
                let color: Vec<_> = parameters.by_ref().take(count).collect();
                *background = format!("48;{};{}", if count == 1 { 5 } else { 2 }, color.join(";"));
            }
            _ if matches!(parameter.parse::<u8>(), Ok(40..=47 | 100..=107)) => {
                parameter.clone_into(background);
            }
            _ => {}
        }
    }
}

fn write_row(
    output: &mut impl Write,
    x: usize,
    y: usize,
    width: usize,
    row: &Row,
) -> io::Result<()> {
    write!(output, "\x1b[0m\x1b[{};{}H", y + 1, x + 1)?;
    output.write_all(&row.b)?;
    write!(
        output,
        "\x1b[0m{}",
        " ".repeat(width.saturating_sub(row.w as usize))
    )?;
    if let Some((start, text)) = &row.selection {
        write!(
            output,
            "\x1b[{};{}H\x1b[7m{}\x1b[0m",
            y + 1,
            x + start + 1,
            text
        )?;
    }
    Ok(())
}

/// Check the complete frame before committing its source or terminal geometry.
pub fn validate_frame(frame: &protocol::ChromeFrame) -> io::Result<()> {
    validate_style(&frame.canvas)?;
    if frame
        .update
        .as_ref()
        .is_some_and(|update| update.view != frame.view)
        || frame.moves.len() > 256
        || (frame.transcript.is_none() && (frame.update.is_some() || !frame.moves.is_empty()))
    {
        return Err(invalid("invalid chrome transcript publication"));
    }
    let Some(size) = frame.size else {
        return Err(invalid("chrome frame needs dimensions"));
    };
    if !(1..=4096).contains(&size.columns)
        || !(1..=256).contains(&size.rows)
        || frame.rows.len() != size.rows as usize
        || frame.cursor_x >= size.columns
        || frame.cursor_y >= size.rows
        || frame.cursor_shape > 6
    {
        return Err(invalid("invalid chrome frame geometry"));
    }
    if let Some(rect) = &frame.transcript {
        if rect.x >= size.columns
            || rect.y >= size.rows
            || rect.width == 0
            || rect.height == 0
            || rect.width > size.columns - rect.x
            || rect.height > size.rows - rect.y
        {
            return Err(invalid("transcript rectangle exceeds chrome frame"));
        }
    }
    if (!frame.overlay_rows.is_empty() && frame.overlay_rows.len() != frame.rows.len())
        || frame.occlusions.len() > (size.rows * size.columns) as usize
        || frame.occlusions.iter().any(|rect| {
            rect.x >= size.columns
                || rect.y >= size.rows
                || rect.width > size.columns - rect.x
                || rect.height > size.rows - rect.y
        })
    {
        return Err(invalid("invalid chrome overlay geometry"));
    }
    if frame
        .rows
        .iter()
        .chain(&frame.overlay_rows)
        .any(|row| row.len() > 256 * 1024)
    {
        return Err(invalid("encoded chrome row exceeds 256 KiB"));
    }
    if frame.routes.len() > 256
        || frame.routes.iter().any(|route| {
            route.input.is_empty() || route.input.len() > 128 || route.r#move.is_none()
        })
    {
        return Err(invalid("invalid transcript input routes"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    #[test]
    fn scrolling_repaints_only_the_transcript_and_restores_the_python_cursor() {
        let mut view = Viewport::default();
        view.document
            .update(protocol::TranscriptUpdate {
                set: vec![protocol::Element {
                    id: 1,
                    lines: vec![
                        protocol::Line {
                            spans: vec![protocol::Span {
                                text: "first".into(),
                                sgr: "31".into(),
                            }],
                        },
                        protocol::Line {
                            spans: vec![protocol::Span {
                                text: "second".into(),
                                sgr: "32".into(),
                            }],
                        },
                    ],
                    ..Default::default()
                }],
                order: Some(protocol::TranscriptOrder { ids: vec![1] }),
                ..Default::default()
            })
            .unwrap();
        let mut compositor = Compositor::default();
        compositor
            .set_frame(protocol::ChromeFrame {
                size: Some(protocol::Size {
                    columns: 8,
                    rows: 2,
                }),
                rows: vec![b"        ".to_vec(), b"\x1b[34mdraft   \x1b[0m".to_vec()],
                cursor_x: 5,
                cursor_y: 1,
                cursor_visible: true,
                transcript: Some(protocol::TranscriptRect {
                    x: 0,
                    y: 0,
                    width: 8,
                    height: 1,
                }),
                ..Default::default()
            })
            .unwrap();
        compositor
            .present(&mut Vec::new(), Some(&mut view), None)
            .unwrap();
        view.navigate(&protocol::TranscriptMove {
            amount: -1,
            ..Default::default()
        });
        let mut bytes = Vec::new();
        compositor
            .present(&mut bytes, Some(&mut view), None)
            .unwrap();
        let painted = String::from_utf8(bytes).unwrap();
        assert_eq!(painted.contains("draft"), false);
        assert_eq!(painted.contains("first"), true);
        assert_eq!(
            painted.ends_with("\x1b[0m\x1b[2;6H\x1b[0 q\x1b[?7h\x1b[?25h\x1b[?2026l"),
            true
        );
        assert_matches!(compositor.frame(), Some(frame) if frame.cursor_x == 5);
    }

    #[test]
    fn thumb_ends_glide_in_eighths_and_keep_a_minimum_length() {
        let rows = vec![Row::default(); 4];
        assert_eq!(
            thumb_column(&rows, 10, "90", 0.5, 0.25),
            [
                Some("\x1b[0;90m▄\x1b[0m".into()),
                Some("\x1b[0;90;7m \x1b[0m".into()),
                Some("\x1b[0;90;7m▄\x1b[0m".into()),
                None,
            ]
        );
        assert_eq!(
            thumb_column(&rows, 10, "90", 0.001, 1.0)
                .iter()
                .map(Option::is_some)
                .collect::<Vec<_>>(),
            [false, false, true, true]
        );
    }

    #[test]
    fn thumb_takes_the_background_of_the_cell_it_covers() {
        assert_eq!(
            background(b"\x1b[0m\x1b[48;2;1;2;3mcode\x1b[0m"),
            "48;2;1;2;3"
        );
        assert_eq!(background(b"\x1b[38;5;9;48;5;236mcode\x1b[0m"), "48;5;236");
        assert_eq!(
            background(b"\x1b]8;;https://a.test\x1b\\\x1b[44mlink\x1b]8;;\x1b\\"),
            "44"
        );
        assert_eq!(background(b"\x1b[41mred\x1b[49mplain"), "");
    }

    #[test]
    fn moving_thumb_repaints_rows_and_sits_on_the_last_column() {
        let mut view = Viewport::default();
        view.document
            .update(protocol::TranscriptUpdate {
                set: vec![protocol::Element {
                    id: 1,
                    lines: (0..8)
                        .map(|line| protocol::Line {
                            spans: vec![protocol::Span {
                                text: format!("line {line}"),
                                sgr: String::new(),
                            }],
                        })
                        .collect(),
                    ..Default::default()
                }],
                order: Some(protocol::TranscriptOrder { ids: vec![1] }),
                ..Default::default()
            })
            .unwrap();
        let mut compositor = Compositor::default();
        compositor
            .set_frame(protocol::ChromeFrame {
                size: Some(protocol::Size {
                    columns: 10,
                    rows: 2,
                }),
                rows: vec![Vec::new(); 2],
                canvas: "40".into(),
                transcript: Some(protocol::TranscriptRect {
                    x: 0,
                    y: 0,
                    width: 10,
                    height: 2,
                }),
                ..Default::default()
            })
            .unwrap();
        let mut bytes = Vec::new();
        compositor
            .present(&mut bytes, Some(&mut view), Some("90"))
            .unwrap();
        let painted = String::from_utf8(bytes).unwrap();
        assert!(painted.contains("\x1b[2;10H\x1b[0;40;90;7m \x1b[0m"));
        let mut bytes = Vec::new();
        compositor
            .present(&mut bytes, Some(&mut view), Some("90"))
            .unwrap();
        assert!(!String::from_utf8(bytes).unwrap().contains("line"));
        let mut bytes = Vec::new();
        compositor
            .present(&mut bytes, Some(&mut view), None)
            .unwrap();
        let painted = String::from_utf8(bytes).unwrap();
        assert!(painted.contains("line 7"));
        assert!(!painted.contains("\x1b[2;10H"));
    }
}
