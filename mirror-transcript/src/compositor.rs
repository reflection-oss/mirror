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
        self.rectangle = None;
    }

    pub fn present(
        &mut self,
        output: &mut impl Write,
        viewport: Option<&mut Viewport>,
    ) -> io::Result<()> {
        let Some(frame) = &self.frame else {
            return Ok(());
        };
        let Some(size) = frame.size else {
            return Ok(());
        };
        self.body = match (frame.transcript.as_ref(), viewport) {
            (Some(rect), Some(viewport)) => viewport
                .rows(rect.width as u16, rect.height as u16)
                .into_iter()
                .map(|row| row.on_canvas(rect.width as u16, &frame.canvas))
                .collect(),
            _ => Vec::new(),
        };
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
                if changed_rectangle
                    || self.previous_body.get(y) != Some(row)
                    || self.previous_chrome.get(screen_y) != self.chrome.get(screen_y)
                {
                    write_row(
                        &mut bytes,
                        rect.x as usize,
                        screen_y,
                        rect.width as usize,
                        row,
                    )?;
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
        self.rectangle.clone_from(&frame.transcript);
        Ok(())
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
            .present(&mut Vec::new(), Some(&mut view))
            .unwrap();
        view.navigate(&protocol::TranscriptMove {
            amount: -1,
            ..Default::default()
        });
        let mut bytes = Vec::new();
        compositor.present(&mut bytes, Some(&mut view)).unwrap();
        let painted = String::from_utf8(bytes).unwrap();
        assert_eq!(painted.contains("draft"), false);
        assert_eq!(painted.contains("first"), true);
        assert_eq!(
            painted.ends_with("\x1b[0m\x1b[2;6H\x1b[0 q\x1b[?7h\x1b[?25h\x1b[?2026l"),
            true
        );
        assert_matches!(compositor.frame(), Some(frame) if frame.cursor_x == 5);
    }
}
