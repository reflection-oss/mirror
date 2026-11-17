//! Transcript navigation and reading anchors; no conversation or editor policy.

use crate::{clip_cell, protocol, wrap, Document, Position, Row};

/// One retained reading position per Python-owned workspace view.
#[derive(Debug, Default)]
pub struct Viewport {
    pub document: Document,
    pub revision: u64,
    anchor: Option<Position>,
    top: usize,
    width: u16,
    height: u16,
    header: bool,
    visible: Vec<Row>,
    selection: Option<crate::Selection>,
}

impl Viewport {
    pub fn reading(&self) -> bool {
        self.anchor.is_some()
    }

    pub fn settle(&mut self, before: u32, after: u32) {
        if let Some(anchor) = &mut self.anchor {
            if anchor.id == before {
                anchor.id = after;
            }
        }
    }

    pub fn navigate(&mut self, movement: &protocol::TranscriptMove) {
        self.selection = None;
        let total = self.document.total(self.width);
        let page = usize::from(self.height)
            .saturating_sub(1 + usize::from(self.header))
            .max(1);
        let target = match protocol::transcript_move::Kind::try_from(movement.kind) {
            Ok(protocol::transcript_move::Kind::Follow) => {
                self.anchor = None;
                return;
            }
            Ok(protocol::transcript_move::Kind::First) => 0,
            Ok(protocol::transcript_move::Kind::Rows) => {
                self.top.saturating_add_signed(movement.amount as isize)
            }
            Ok(protocol::transcript_move::Kind::Page) => self
                .top
                .saturating_add_signed((movement.amount as isize).saturating_mul(page as isize)),
            Err(_) => return,
        };
        let header = self.document.sticky(target, self.width).is_some();
        let bottom =
            total.saturating_sub(usize::from(self.height).saturating_sub(usize::from(header)));
        self.top = target.min(bottom);
        self.anchor = if target >= bottom {
            None
        } else {
            self.document
                .row(target, self.width)
                .map(|(_, position)| position)
        };
    }

    pub fn rows(&mut self, width: u16, height: u16) -> Vec<Row> {
        self.width = width;
        self.height = height;
        if width == 0 || height == 0 {
            return Vec::new();
        }
        let total = self.document.total(width);
        let following = total.saturating_sub(height as usize);
        let mut top = self
            .anchor
            .as_ref()
            .and_then(|anchor| self.document.resolve(anchor, width))
            .unwrap_or_else(|| {
                if self.anchor.is_some() {
                    self.top.min(total.saturating_sub(1))
                } else {
                    following
                }
            });
        let mut sticky = self.document.sticky(top, width);
        if self.anchor.is_none() && sticky.is_some() {
            top = (top + 1).min(total.saturating_sub(1));
        }
        // Preserve reading anchors only while they can fill the viewport.
        loop {
            let bottom = total.saturating_sub(height as usize - usize::from(sticky.is_some()));
            if top <= bottom {
                break;
            }
            top = bottom;
            sticky = self.document.sticky(top, width);
        }
        self.header = sticky.is_some();
        self.top = top;
        let body = self.document.rows(top, width, height);
        if self.anchor.is_some() {
            self.anchor = body.first().map(|(_, position)| position.clone());
        }
        let mut rows = Vec::with_capacity(height as usize);
        if let Some(sticky) = sticky {
            let style = sticky.sgr.clone();
            let line = clip_cell(
                &[sticky],
                width as usize,
                protocol::block::Overflow::Ellipsis,
            );
            rows.extend(
                wrap(&line, width, false)
                    .into_iter()
                    .map(|row| row.on_canvas(width, &style)),
            );
        }
        rows.extend(
            body.into_iter()
                .take(height as usize - rows.len())
                .map(|(row, _)| row),
        );
        rows.resize_with(height as usize, Row::default);
        if self
            .selection
            .as_ref()
            .is_some_and(|selection| selection.rows != rows)
        {
            self.selection = None;
        }
        self.visible.clone_from(&rows);
        if let Some(selection) = &self.selection {
            for (y, row) in rows.iter_mut().enumerate() {
                if let Some((start, end)) = selection.range(y, width as usize) {
                    row.selection = Some((
                        crate::cell_start(&row.text, start),
                        crate::cells(&row.text, start, end),
                    ));
                }
            }
        }
        rows
    }

    /// Keep reading anchors, but release layout when the view leaves the screen.
    pub fn evict_layout(&mut self) {
        self.clear_selection();
        self.visible = Vec::new();
        self.document.trim_caches(0);
    }

    pub fn clear_selection(&mut self) {
        self.selection = None;
    }

    pub fn pointer(
        &mut self,
        x: usize,
        y: usize,
        button: u32,
        released: bool,
    ) -> Option<crate::Action> {
        if button == 35 {
            return None;
        }
        if button == 0 && !released {
            self.selection = Some(crate::Selection {
                start: (y, x),
                end: (y, x),
                rows: self.visible.clone(),
            });
            return None;
        }
        let selection = self.selection.as_mut()?;
        selection.end = (y, x);
        if !released {
            return None;
        }
        if selection.start != selection.end {
            return selection.text(self.width as usize).map(crate::Action::Copy);
        }
        self.selection = None;
        if let Some(hit) = self
            .visible
            .get(y)
            .and_then(|row| row.actions.iter().find(|hit| hit.start <= x && x < hit.end))
        {
            return Some((*hit.action).clone());
        }
        self.toggle(y);
        None
    }

    pub fn toggle(&mut self, y: usize) -> bool {
        let Some(body) = y.checked_sub(usize::from(self.header)) else {
            return false;
        };
        let Some((_, position)) = self.document.row(self.top + body, self.width) else {
            return false;
        };
        let anchor = self
            .document
            .row(self.top, self.width)
            .map(|(_, position)| position);
        if !self.document.toggle(position.id) {
            return false;
        }
        if self.anchor.is_none() {
            self.anchor = anchor;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Span;
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    fn source(text: &str) -> protocol::Element {
        protocol::Element {
            id: 1,
            lines: text
                .split('\n')
                .map(|text| protocol::Line {
                    spans: vec![Span {
                        text: text.into(),
                        sgr: String::new(),
                    }],
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn clicking_plain_text_does_not_stop_following_output() {
        let mut view = Viewport::default();
        view.document
            .update(protocol::TranscriptUpdate {
                set: vec![source("one\ntwo\nthree")],
                order: Some(protocol::TranscriptOrder { ids: vec![1] }),
                ..Default::default()
            })
            .unwrap();
        view.rows(20, 2);
        view.pointer(1, 0, 0, false);
        assert_matches!(view.pointer(1, 0, 0, true), None);
        assert!(!view.reading());
        view.document
            .update(protocol::TranscriptUpdate {
                set: vec![source("one\ntwo\nthree\nfour")],
                ..Default::default()
            })
            .unwrap();
        assert_eq!(view.rows(20, 2)[1].text.as_ref(), "four");
    }

    #[test]
    fn selection_copies_whole_graphemes_and_rejects_replaced_rows() {
        let mut view = Viewport::default();
        view.document
            .update(protocol::TranscriptUpdate {
                set: vec![source("a界e\u{301}")],
                order: Some(protocol::TranscriptOrder { ids: vec![1] }),
                ..Default::default()
            })
            .unwrap();
        view.rows(20, 2);
        view.pointer(2, 0, 0, false);
        view.pointer(4, 0, 32, false);
        assert_eq!(
            view.rows(20, 2)[0].selection,
            Some((1, "界e\u{301}".into()))
        );
        assert_matches!(view.pointer(4, 0, 0, true), Some(crate::Action::Copy(text)) if text == "界e\u{301}");
        view.pointer(0, 0, 0, false);
        view.document
            .update(protocol::TranscriptUpdate {
                set: vec![source("new frame")],
                ..Default::default()
            })
            .unwrap();
        view.rows(20, 2);
        assert_matches!(view.pointer(4, 0, 0, true), None);
    }

    #[test]
    fn markdown_link_and_code_copy_targets_survive_wrapping() {
        let mut view = Viewport::default();
        view.document
            .update(protocol::TranscriptUpdate {
                set: vec![protocol::Element {
                    id: 1,
                    blocks: vec![protocol::Block {
                        kind: protocol::block::Kind::Markdown as i32,
                        text: "[link](https://example.test)\n\n```rust\nlet answer = 42;\n```"
                            .into(),
                        ..Default::default()
                    }],
                    ..Default::default()
                }],
                order: Some(protocol::TranscriptOrder { ids: vec![1] }),
                ..Default::default()
            })
            .unwrap();
        let rows = view.rows(30, 12);
        let link = rows
            .iter()
            .position(|row| row.text.starts_with("link"))
            .unwrap();
        let copy = rows
            .iter()
            .position(|row| row.text.contains("Copy"))
            .unwrap();
        view.pointer(1, link, 0, false);
        assert_matches!(view.pointer(1, link, 0, true), Some(crate::Action::Link(url)) if url == "https://example.test");
        view.pointer(27, copy, 0, false);
        assert_matches!(view.pointer(27, copy, 0, true), Some(crate::Action::Copy(text)) if text == "let answer = 42;");
    }

    #[test]
    fn sticky_summary_fills_the_viewport_width_with_its_style() {
        let mut view = Viewport::default();
        let mut element = source("first\nsecond\nthird");
        element.sticky = Some(Span {
            text: "summary".into(),
            sgr: "41".into(),
        });
        view.document
            .update(protocol::TranscriptUpdate {
                set: vec![element],
                order: Some(protocol::TranscriptOrder { ids: vec![1] }),
                ..Default::default()
            })
            .unwrap();
        let rows = view.rows(10, 2);
        assert_eq!(rows[0].w, 10);
        assert!(rows[0].b.ends_with(b"\x1b[41m   \x1b[0m"));
        assert_eq!(rows[1].text.as_ref(), "third");
    }

    #[test]
    fn streaming_does_not_move_a_reading_anchor() {
        let mut view = Viewport::default();
        view.document
            .update(protocol::TranscriptUpdate {
                set: vec![source("one\ntwo\nthree\nfour")],
                order: Some(protocol::TranscriptOrder { ids: vec![1] }),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(view.rows(20, 2)[0].b.as_ref(), b"three");
        view.navigate(&protocol::TranscriptMove {
            amount: -1,
            ..Default::default()
        });
        assert_eq!(view.rows(20, 2)[0].b.as_ref(), b"two");
        view.document
            .update(protocol::TranscriptUpdate {
                set: vec![source("one\ntwo\nthree\nfour\nfive")],
                ..Default::default()
            })
            .unwrap();
        assert_eq!(view.rows(20, 2)[0].b.as_ref(), b"two");
        assert_matches!(view.document.row(4, 20), Some((row, _)) if row.b.as_ref() == b"five");
        view.navigate(&protocol::TranscriptMove {
            kind: protocol::transcript_move::Kind::Follow as i32,
            amount: 0,
        });
        assert_eq!(view.rows(20, 2)[0].b.as_ref(), b"four");
    }

    #[test]
    fn reading_tail_fills_a_larger_viewport_after_eviction() {
        let mut view = Viewport::default();
        view.document
            .update(protocol::TranscriptUpdate {
                set: vec![source("one\ntwo\nthree\nfour\nfive\nsix")],
                order: Some(protocol::TranscriptOrder { ids: vec![1] }),
                ..Default::default()
            })
            .unwrap();
        view.rows(20, 2);
        view.navigate(&protocol::TranscriptMove {
            amount: -1,
            ..Default::default()
        });
        assert_eq!(view.rows(20, 2)[0].text.as_ref(), "four");
        view.evict_layout();
        let rows = view.rows(20, 4);
        assert_eq!(
            rows.iter().map(|row| row.text.as_ref()).collect::<Vec<_>>(),
            ["three", "four", "five", "six"]
        );
        assert!(view.reading());
    }

    #[test]
    fn clamping_the_tail_rechecks_the_sticky_header_at_the_new_top() {
        let mut view = Viewport::default();
        let mut first = source("one\ntwo\nthree");
        first.sticky = Some(Span {
            text: "old summary".into(),
            ..Default::default()
        });
        let mut second = source("prompt\nbody\nlast");
        second.id = 2;
        second.sticky = Some(Span {
            text: "new summary".into(),
            ..Default::default()
        });
        view.document
            .update(protocol::TranscriptUpdate {
                set: vec![first, second],
                order: Some(protocol::TranscriptOrder { ids: vec![1, 2] }),
                ..Default::default()
            })
            .unwrap();
        view.rows(20, 2);
        view.navigate(&protocol::TranscriptMove {
            amount: -1,
            ..Default::default()
        });
        let rows = view.rows(20, 4);
        assert_eq!(
            rows.iter().map(|row| row.text.as_ref()).collect::<Vec<_>>(),
            ["old summary", "three", "prompt", "body"]
        );
        assert!(view.reading());
    }

    #[test]
    fn settled_frame_keeps_its_source_anchor_after_eviction_and_resize() {
        let mut view = Viewport::default();
        let mut element = protocol::Element {
            id: 1,
            blocks: vec![protocol::Block {
                kind: protocol::block::Kind::Frame as i32,
                insets: vec![0; 4],
                children: vec![protocol::Block {
                    text: "abcdefghijklmnop\nlast".into(),
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        };
        view.document
            .update(protocol::TranscriptUpdate {
                set: vec![element.clone()],
                order: Some(protocol::TranscriptOrder { ids: vec![1] }),
                ..Default::default()
            })
            .unwrap();
        view.rows(4, 2);
        view.navigate(&protocol::TranscriptMove {
            amount: -1,
            ..Default::default()
        });
        assert_eq!(view.rows(4, 2)[0].text.as_ref(), "ijkl");
        element.id = 2;
        view.settle(1, 2);
        view.document
            .update(protocol::TranscriptUpdate {
                set: vec![element],
                order: Some(protocol::TranscriptOrder { ids: vec![2] }),
                ..Default::default()
            })
            .unwrap();
        view.evict_layout();
        assert_eq!(view.rows(8, 2)[0].text.as_ref(), "ijklmnop");
        assert!(view.reading());
    }

    #[test]
    fn resize_keeps_the_logical_character_at_the_top() {
        let mut view = Viewport::default();
        view.document
            .update(protocol::TranscriptUpdate {
                set: vec![source("abcdefghijklmnop\nlast")],
                order: Some(protocol::TranscriptOrder { ids: vec![1] }),
                ..Default::default()
            })
            .unwrap();
        view.rows(4, 2);
        view.navigate(&protocol::TranscriptMove {
            amount: -1,
            ..Default::default()
        });
        assert_eq!(view.rows(4, 2)[0].b.as_ref(), b"ijkl");
        assert_eq!(view.rows(8, 2)[0].b.as_ref(), b"ijklmnop");
    }
}
