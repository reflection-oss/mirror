//! A read-only selection pins the painted rows, not mutable application sources.

use unicode_segmentation::UnicodeSegmentation as _;
use unicode_width::UnicodeWidthStr as _;

use crate::Row;

#[derive(Debug)]
pub struct Selection {
    pub start: (usize, usize),
    pub end: (usize, usize),
    pub rows: Vec<Row>,
}

impl Selection {
    pub fn range(&self, y: usize, width: usize) -> Option<(usize, usize)> {
        let (start, end) = if self.start <= self.end {
            (self.start, self.end)
        } else {
            (self.end, self.start)
        };
        (start != end && start.0 <= y && y <= end.0).then_some((
            if y == start.0 { start.1 } else { 0 },
            if y == end.0 { end.1 } else { width },
        ))
    }

    pub fn text(&self, width: usize) -> Option<String> {
        let mut lines = Vec::new();
        for (y, row) in self.rows.iter().enumerate() {
            if let Some((start, end)) = self.range(y, width) {
                lines.push(cells(&row.text, start, end).trim_end().to_owned());
            }
        }
        let text = lines.join("\n");
        (!text.is_empty() && text.len() <= 75_000).then_some(text)
    }
}

pub fn cell_start(text: &str, position: usize) -> usize {
    let mut column = 0;
    for glyph in text.graphemes(true) {
        if column + glyph.width() > position {
            break;
        }
        column += glyph.width();
    }
    column
}

pub fn cells(text: &str, start: usize, end: usize) -> String {
    let mut column = 0;
    text.graphemes(true)
        .filter(|glyph| {
            let left = column;
            column += glyph.width();
            column > start && left < end
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    #[test]
    fn selection_keeps_graphemes_and_omits_unselected_rows() {
        let selection = Selection {
            start: (0, 1),
            end: (1, 3),
            rows: vec![
                Row {
                    text: "a界e\u{301}  ".into(),
                    ..Default::default()
                },
                Row {
                    text: "next".into(),
                    ..Default::default()
                },
            ],
        };
        assert_eq!(cells("a界e\u{301}", 1, 3), "界");
        assert_matches!(selection.text(8), Some(text) if text == "界e\u{301}\nnex");
    }
}
