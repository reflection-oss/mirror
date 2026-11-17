//! Source blocks retained independently of their width-specific logical projection.

use std::sync::Arc;

use unicode_segmentation::UnicodeSegmentation as _;
use unicode_width::UnicodeWidthStr as _;

use crate::{
    protocol::{
        block::{Alignment, Kind, Overflow},
        Block, Theme,
    },
    Line, Markdown, Source, Span,
};

#[derive(Debug, Clone)]
pub struct Content {
    pub blocks: Arc<Vec<Block>>,
    pub theme: Arc<Theme>,
    pub expanded: bool,
}

impl Content {
    pub fn render(&self, width: u16) -> Vec<Line> {
        indexed(&self.blocks, 0)
            .flat_map(|(source, block)| self.block(block, width, source))
            .collect()
    }

    pub fn collapsible(&self) -> bool {
        fn collapsible(block: &Block) -> bool {
            block.collapse != 0
                || !block.collapsed.is_empty()
                || block.children.iter().any(collapsible)
        }
        self.blocks.iter().any(collapsible)
    }

    fn block(&self, block: &Block, width: u16, source: u32) -> Vec<Line> {
        if !self.expanded && !block.collapsed.is_empty() {
            return indexed(&block.collapsed, source + 1 + node_count(&block.children))
                .flat_map(|(source, child)| self.block(child, width, source))
                .collect();
        }
        let padding = (block.padding as usize).min(width as usize / 4);
        let inner = width.saturating_sub(2 * padding as u16).max(1);
        let origin = Source {
            block: source,
            offset: 0,
        };
        let mut lines = match Kind::try_from(block.kind).unwrap_or(Kind::Text) {
            Kind::Markdown => Markdown::render(&block.text, &block.sgr, &self.theme, inner, origin),
            Kind::Frame => self.frame(block, inner, source),
            Kind::Panel => self.panel(block, inner, source),
            Kind::Rule => self.rule(block, inner),
            Kind::Grid => self.grid(block, inner, source),
            Kind::Table => crate::table_lines(
                indexed(&block.children, source + 1)
                    .map(|(source, row)| {
                        indexed(&row.children, source + 1)
                            .map(|(source, cell)| {
                                let mut result = Vec::new();
                                for line in self.block(cell, inner, source) {
                                    if !result.is_empty() {
                                        result.push(Span {
                                            action: None,
                                            source: None,
                                            text: " ".into(),
                                            sgr: String::new(),
                                        });
                                    }
                                    result.extend(line);
                                }
                                result
                            })
                            .collect()
                    })
                    .collect(),
                inner,
                &self.theme,
            ),
            Kind::Code => {
                let mut theme = (*self.theme).clone();
                if !block.sgr.is_empty() {
                    theme.roles.insert("code".into(), block.sgr.clone());
                }
                crate::code_lines(
                    &block.text,
                    &block.language,
                    &theme,
                    &block.syntax,
                    Some(origin),
                )
            }
            Kind::Column => {
                let mut lines = Vec::new();
                for (source, child) in indexed(&block.children, source + 1) {
                    let child = self.block(child, inner, source);
                    if child.is_empty() {
                        continue;
                    }
                    if !lines.is_empty() {
                        lines.extend(vec![vec![]; block.gap.min(256) as usize]);
                    }
                    lines.extend(child);
                }
                lines
            }
            Kind::Row => self.columns(&block.children, inner, block.spread, source + 1),
            Kind::Inline => vec![indexed(&block.children, source + 1)
                .flat_map(|(source, child)| self.block(child, inner, source).into_iter().flatten())
                .collect()],
            Kind::Diff => crate::diff_lines(
                &block.text,
                block.show_file_headers.unwrap_or(true),
                inner,
                &self.theme,
                Some(origin),
            ),
            _ => {
                let mut lines: Vec<Line> = if block.lines.is_empty() {
                    block
                        .text
                        .split('\n')
                        .map(|text| {
                            vec![Span {
                                text: text.into(),
                                sgr: block.sgr.clone(),
                                ..Default::default()
                            }]
                        })
                        .collect()
                } else {
                    block
                        .lines
                        .iter()
                        .map(|line| line.spans.iter().cloned().map(Span::from).collect())
                        .collect()
                };
                crate::annotate_lines(&mut lines, origin);
                lines
            }
        };
        if !block.word_wrap.unwrap_or(true) && !block.clip && !block.no_wrap {
            lines = lines
                .iter()
                .flat_map(|line| crate::wrap_lines(line, inner, false))
                .collect();
        }
        if block.no_wrap {
            lines = lines
                .iter()
                .map(|line| {
                    crate::clip_cell(
                        line,
                        inner as usize,
                        crate::protocol::block::Overflow::try_from(block.overflow)
                            .unwrap_or_default(),
                    )
                })
                .collect();
        }
        if block.clip {
            lines = lines
                .iter()
                .map(|line| {
                    crate::wrap_lines(line, inner, false)
                        .into_iter()
                        .next()
                        .unwrap_or_default()
                })
                .collect();
        }
        if !block.title.is_empty()
            && block.kind != Kind::Panel as i32
            && block.kind != Kind::Rule as i32
        {
            lines.insert(
                0,
                vec![Span {
                    action: None,
                    source: None,
                    text: format!(
                        "{}{}",
                        if block.collapse == 0 {
                            ""
                        } else if self.expanded {
                            "▾ "
                        } else {
                            "▸ "
                        },
                        block.title
                    ),
                    sgr: if block.title_sgr.is_empty() {
                        self.theme.accent.clone()
                    } else {
                        block.title_sgr.clone()
                    },
                }],
            );
        }
        if !self.expanded && block.collapse != 0 && lines.len() > block.collapse as usize + 1 {
            let hidden = lines.len() - block.collapse as usize;
            lines.truncate(block.collapse as usize);
            lines.push(vec![Span {
                action: None,
                source: None,
                text: format!("  … {hidden} more lines · click to expand"),
                sgr: self.theme.muted.clone(),
            }]);
        }
        if padding != 0 {
            for line in &mut lines {
                line.insert(
                    0,
                    Span {
                        action: None,
                        source: None,
                        text: " ".repeat(padding),
                        sgr: block.sgr.clone(),
                    },
                );
            }
        }
        if let Some(height) = block.height {
            lines = lines
                .iter()
                .flat_map(|line| crate::wrap_lines(line, inner, true))
                .collect();
            lines.resize(height.min(256) as usize, vec![]);
        }
        if block.trim_end {
            while lines.last().is_some_and(|line| {
                line.iter()
                    .all(|span| span.sgr.is_empty() && span.text.trim().is_empty())
            }) {
                lines.pop();
            }
        }
        lines
    }

    fn frame(&self, block: &Block, width: u16, source: u32) -> Vec<Line> {
        let [top, right, bottom, left] = block.insets.as_slice() else {
            return vec![];
        };
        let left = (*left).min(16).min(width.saturating_sub(1) as u32) as u16;
        let inner = width.saturating_sub(left + (*right).min(16) as u16).max(1);
        let blank = vec![Span {
            action: None,
            source: None,
            text: " ".repeat(width as usize),
            sgr: block.sgr.clone(),
        }];
        let mut result = vec![blank.clone(); (*top).min(16) as usize];
        let mut first = true;
        for line in indexed(&block.children, source + 1)
            .flat_map(|(source, child)| self.block(child, inner, source))
        {
            let overflow =
                crate::protocol::block::Overflow::try_from(block.overflow).unwrap_or_default();
            let lines = if block.no_wrap {
                vec![crate::clip_cell(&line, inner as usize, overflow)]
            } else {
                crate::wrap_cell(&line, inner, overflow)
            };
            for mut line in lines {
                let marker = block.marker.as_ref().filter(|_| first);
                let mut text = marker.map_or(String::new(), |marker| {
                    fit(&marker.text, left as usize, false)
                });
                if marker.is_some() {
                    text.push_str(&" ".repeat((left as usize).saturating_sub(1 + text.width())));
                }
                let marker_width = text.width();
                let used: usize = line.iter().map(|span| span.text.width()).sum();
                for span in &mut line {
                    span.sgr = format!("{};{}", block.sgr, span.sgr)
                        .trim_matches(';')
                        .into();
                }
                let mut row = vec![
                    Span {
                        action: None,
                        source: None,
                        text,
                        sgr: format!("{};{}", block.sgr, marker.map_or("", |marker| &marker.sgr))
                            .trim_matches(';')
                            .into(),
                    },
                    Span {
                        action: None,
                        source: None,
                        text: " ".repeat(left as usize - marker_width),
                        sgr: block.sgr.clone(),
                    },
                ];
                row.extend(line);
                row.push(Span {
                    action: None,
                    source: None,
                    text: " ".repeat((width as usize).saturating_sub(left as usize + used)),
                    sgr: block.sgr.clone(),
                });
                result.push(row);
                first = false;
            }
        }
        result.extend(vec![blank; (*bottom).min(16) as usize]);
        result
    }

    fn rule(&self, block: &Block, width: u16) -> Vec<Line> {
        let width = width as usize;
        let characters = if block.text.width() == 0 {
            "─"
        } else {
            &block.text
        };
        let stroke = |columns: usize| {
            let text = characters.repeat(columns / characters.width() + 1);
            let mut line = crate::clip_line(
                &[Span {
                    action: None,
                    source: None,
                    text,
                    sgr: block.sgr.clone(),
                }],
                columns,
            );
            line.push(Span {
                action: None,
                source: None,
                text: " ".repeat(columns.saturating_sub(crate::line_width(&line))),
                sgr: block.sgr.clone(),
            });
            line
        };
        let alignment = Alignment::try_from(block.alignment).unwrap_or_default();
        let available = width.saturating_sub(if alignment == Alignment::Center { 4 } else { 2 });
        let title = block.lines.first().map_or_else(
            || {
                vec![Span {
                    action: None,
                    source: None,
                    text: block.title.clone(),
                    sgr: block.title_sgr.clone(),
                }]
            },
            |line| line.spans.iter().cloned().map(Span::from).collect(),
        );
        if crate::line_width(&title) == 0 || available == 0 {
            return vec![stroke(width)];
        }
        let title = crate::clip_cell(&title, available, Overflow::Ellipsis);
        let spare = width - crate::line_width(&title);
        let left = match alignment {
            Alignment::Left => 0,
            Alignment::Center => spare / 2 - 1,
            Alignment::Right => spare - 1,
        };
        let space = Span {
            action: None,
            source: None,
            text: " ".into(),
            sgr: if alignment == Alignment::Center {
                block.sgr.clone()
            } else {
                String::new()
            },
        };
        let mut line = stroke(left);
        if alignment != Alignment::Left {
            line.push(space.clone());
        }
        line.extend(title);
        if alignment != Alignment::Right {
            line.push(space);
        }
        line.extend(stroke(width - crate::line_width(&line)));
        vec![line]
    }

    fn panel(&self, block: &Block, width: u16, source: u32) -> Vec<Line> {
        let border = |text: String| Span {
            action: None,
            source: None,
            text,
            sgr: block.border_sgr.clone(),
        };
        let inner = width.saturating_sub(2);
        let title = crate::clip_line(
            &[Span {
                action: None,
                source: None,
                text: format!(" {} ", block.title),
                sgr: format!("{};{}", block.border_sgr, block.title_sgr),
            }],
            width.saturating_sub(4) as usize,
        );
        let mut top = vec![border("╭─".into())];
        if !block.title.is_empty() && width > 4 {
            top.extend(title);
        }
        let used = crate::line_width(&top);
        top.push(border(format!(
            "{}╮",
            "─".repeat((width as usize).saturating_sub(used + 1))
        )));
        let mut result = vec![top];
        for mut line in self.frame(block, inner, source) {
            line.insert(0, border("│".into()));
            line.push(border("│".into()));
            result.push(line);
        }
        result.push(vec![border(format!("╰{}╯", "─".repeat(inner as usize)))]);
        result
    }

    fn grid(&self, block: &Block, width: u16, source: u32) -> Vec<Line> {
        let Some(first) = block.children.first() else {
            return vec![];
        };
        let mut ratios: Vec<usize> = first
            .children
            .iter()
            .map(|cell| cell.ratio as usize)
            .collect();
        let intrinsic: Vec<_> = (0..ratios.len())
            .map(|index| {
                indexed(&block.children, source + 1)
                    .filter_map(|(source, row)| indexed(&row.children, source + 1).nth(index))
                    .map(|(source, cell)| {
                        let padding = cell.insets.get(1).copied().unwrap_or(0)
                            + cell.insets.get(3).copied().unwrap_or(0);
                        indexed(&cell.children, source + 1)
                            .flat_map(|(source, child)| self.block(child, width, source))
                            .map(|line| crate::line_width(&line))
                            .max()
                            .unwrap_or(0)
                            + padding as usize
                    })
                    .max()
                    .unwrap_or(1)
            })
            .collect();
        if ratios.iter().all(|ratio| *ratio == 0) {
            ratios.clone_from(&intrinsic);
        }
        let mut remaining = if block.spread {
            width as usize
        } else {
            (width as usize).min(intrinsic.iter().sum())
        };
        let mut widths = vec![0; ratios.len()];
        let flexible = first.children.iter().filter(|cell| !cell.no_wrap).count();
        for (index, cell) in first.children.iter().enumerate() {
            if cell.no_wrap {
                widths[index] = intrinsic[index].min(remaining.saturating_sub(flexible));
                remaining -= widths[index];
                ratios[index] = 0;
            }
        }
        let mut total = ratios.iter().sum::<usize>();
        for (index, ratio) in ratios.iter().enumerate() {
            let assigned = (remaining * ratio).div_ceil(total.max(1)).min(remaining);
            widths[index] += assigned;
            remaining -= assigned;
            total -= ratio;
        }
        let mut result = Vec::new();
        for (source, row) in indexed(&block.children, source + 1) {
            let cells: Vec<_> = indexed(&row.children, source + 1)
                .zip(&widths)
                .map(|((source, cell), width)| self.block(cell, *width as u16, source))
                .collect();
            for index in 0..cells.iter().map(Vec::len).max().unwrap_or(0) {
                let mut line = Vec::new();
                for ((cell, width), source) in cells.iter().zip(&widths).zip(&row.children) {
                    let part = cell.get(index).cloned().unwrap_or_default();
                    let used = crate::line_width(&part);
                    line.extend(part);
                    line.push(Span {
                        action: None,
                        source: None,
                        text: " ".repeat(width.saturating_sub(used)),
                        sgr: source.sgr.clone(),
                    });
                }
                result.push(line);
            }
        }
        result
    }

    fn columns(&self, children: &[Block], width: u16, spread: bool, source: u32) -> Vec<Line> {
        if children.is_empty() {
            return vec![];
        }
        let mut columns: Vec<_> = indexed(children, source)
            .map(|(source, child)| self.block(child, width, source))
            .collect();
        let mut widths: Vec<usize> = columns
            .iter()
            .map(|lines| {
                lines
                    .iter()
                    .map(|line| line.iter().map(|span| span.text.width()).sum())
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        if widths.iter().sum::<usize>() > width as usize {
            let preferred = widths.clone();
            widths.fill(0);
            let mut remaining = width as usize;
            while remaining > 0 {
                let active = widths
                    .iter()
                    .zip(&preferred)
                    .filter(|(assigned, preferred)| assigned < preferred)
                    .count();
                let step = (remaining / active.max(1)).max(1);
                for (assigned, preferred) in widths.iter_mut().zip(&preferred) {
                    let added = (*preferred - *assigned).min(step).min(remaining);
                    *assigned += added;
                    remaining -= added;
                }
            }
            columns = indexed(children, source)
                .zip(&widths)
                .map(|((source, child), width)| {
                    if *width == 0 {
                        return vec![];
                    }
                    self.block(child, *width as u16, source)
                        .iter()
                        .flat_map(|line| crate::wrap_lines(line, *width as u16, true))
                        .collect()
                })
                .collect();
        }
        let weight: usize = children.iter().map(|child| child.ratio as usize).sum();
        let mut spare = (width as usize).saturating_sub(widths.iter().sum());
        let mut remaining_weight = weight;
        for (assigned, child) in widths.iter_mut().zip(children) {
            if child.ratio != 0 && remaining_weight != 0 {
                let added = spare * child.ratio as usize / remaining_weight;
                *assigned += added;
                spare -= added;
                remaining_weight -= child.ratio as usize;
            }
        }
        let height = columns.iter().map(Vec::len).max().unwrap_or(0);
        let mut result = vec![vec![]; height];
        let gaps = children.len().saturating_sub(1).max(1);
        let spare = if spread {
            width as usize - widths.iter().sum::<usize>()
        } else {
            0
        };
        for (index, (lines, columns)) in columns.into_iter().zip(widths).enumerate() {
            let gap = if index + 1 < children.len() {
                spare / gaps + usize::from(index < spare % gaps)
            } else {
                0
            };
            let mut lines = lines.into_iter();
            for row in &mut result {
                let line = lines.next().unwrap_or_default();
                let used: usize = line.iter().map(|span| span.text.width()).sum();
                row.extend(line);
                row.push(Span {
                    action: None,
                    source: None,
                    text: " ".repeat(columns.saturating_sub(used) + gap),
                    sgr: String::new(),
                });
            }
        }
        result
    }
}

// Include hidden alternatives so expansion never renumbers subsequent blocks.
fn indexed(blocks: &[Block], start: u32) -> impl Iterator<Item = (u32, &Block)> {
    blocks.iter().scan(start, |next, block| {
        let source = *next;
        *next += 1 + node_count(&block.children) + node_count(&block.collapsed);
        Some((source, block))
    })
}

fn node_count(blocks: &[Block]) -> u32 {
    blocks
        .iter()
        .map(|block| 1 + node_count(&block.children) + node_count(&block.collapsed))
        .sum()
}

/// Fit a terminal label without splitting a grapheme, preserving a meaningful suffix.
pub fn fit(text: &str, columns: usize, suffix: bool) -> String {
    if text.width() <= columns {
        return text.into();
    }
    if columns == 0 {
        return String::new();
    }
    let graphemes: Vec<_> = text.graphemes(true).collect();
    let mut width = 1;
    let mut selected = Vec::new();
    for index in 0..graphemes.len() {
        let grapheme = graphemes[if suffix {
            graphemes.len() - index - 1
        } else {
            index
        }];
        if width + grapheme.width() > columns {
            break;
        }
        width += grapheme.width();
        selected.push(grapheme);
    }
    if suffix {
        selected.reverse();
        format!("…{}", selected.concat())
    } else {
        format!("{}…", selected.concat())
    }
}

/// Heap storage owned by a block, excluding the block's inline struct.
pub fn block_heap_bytes(block: &Block) -> usize {
    block.text.capacity()
        + block.sgr.capacity()
        + block.title.capacity()
        + block.language.capacity()
        + block.syntax.capacity()
        + block.title_sgr.capacity()
        + block.border_sgr.capacity()
        + block.insets.capacity() * std::mem::size_of::<u32>()
        + block
            .marker
            .as_ref()
            .map_or(0, |marker| marker.text.capacity() + marker.sgr.capacity())
        + block.lines.capacity() * std::mem::size_of::<crate::protocol::Line>()
        + block
            .lines
            .iter()
            .map(source_line_heap_bytes)
            .sum::<usize>()
        + (block.children.capacity() + block.collapsed.capacity()) * std::mem::size_of::<Block>()
        + block.children.iter().map(block_heap_bytes).sum::<usize>()
        + block.collapsed.iter().map(block_heap_bytes).sum::<usize>()
}

/// Charge the larger display span, also covering conversion from wire lines.
pub fn source_line_heap_bytes(line: &crate::protocol::Line) -> usize {
    line.spans.capacity() * std::mem::size_of::<Span>()
        + line
            .spans
            .iter()
            .map(|span| span.text.capacity() + span.sgr.capacity())
            .sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    #[test]
    fn rule_preserves_styled_titles_and_fits_graphemes_on_resize() {
        let content = Content {
            blocks: Arc::new(vec![Block {
                kind: Kind::Rule as i32,
                text: "─".into(),
                sgr: "34".into(),
                lines: vec![crate::protocol::Line {
                    spans: vec![crate::protocol::Span {
                        text: "界e\u{301}abc".into(),
                        sgr: "1".into(),
                    }],
                }],
                ..Default::default()
            }]),
            theme: Arc::default(),
            expanded: false,
        };
        let rows = content.render(9);
        assert_eq!(
            crate::plain(&rows[0]).iter().collect::<String>(),
            "─ 界e\u{301}a… ─"
        );
        assert!(rows[0]
            .iter()
            .any(|span| span.text == "…" && span.sgr == "1"));
        assert_eq!(
            rows[0][0],
            Span {
                action: None,
                source: None,
                text: "─".into(),
                sgr: "34".into()
            }
        );
        assert_eq!(
            crate::plain(&content.render(3)[0])
                .iter()
                .collect::<String>(),
            "───"
        );
    }

    #[test]
    fn rule_aligns_titles_and_pads_wide_strokes() {
        let content = Content {
            blocks: Arc::new(vec![
                Block {
                    kind: Kind::Rule as i32,
                    text: "界".into(),
                    title: "Tag".into(),
                    alignment: Alignment::Left as i32,
                    ..Default::default()
                },
                Block {
                    kind: Kind::Rule as i32,
                    text: "界".into(),
                    title: "Tag".into(),
                    alignment: Alignment::Right as i32,
                    ..Default::default()
                },
            ]),
            theme: Arc::default(),
            expanded: false,
        };
        let rows = content.render(9);
        assert_eq!(
            crate::plain(&rows[0]).iter().collect::<String>(),
            "Tag 界界 "
        );
        assert_eq!(
            crate::plain(&rows[1]).iter().collect::<String>(),
            "界界  Tag"
        );
    }

    #[test]
    fn character_wrapping_keeps_styles_and_does_not_move_words() {
        let content = Content {
            blocks: Arc::new(vec![Block {
                text: "alpha beta".into(),
                sgr: "1;31".into(),
                word_wrap: Some(false),
                ..Default::default()
            }]),
            theme: Arc::default(),
            expanded: false,
        };
        assert_eq!(
            content.render(7),
            vec![
                vec![Span {
                    action: None,
                    source: Some(Source {
                        block: 0,
                        offset: 0
                    }),
                    text: "alpha b".into(),
                    sgr: "1;31".into()
                }],
                vec![Span {
                    action: None,
                    source: Some(Source {
                        block: 0,
                        offset: 7
                    }),
                    text: "eta".into(),
                    sgr: "1;31".into()
                }],
            ]
        );
    }

    #[test]
    fn column_gaps_ignore_hidden_children_and_height_counts_wrapped_rows() {
        let content = Content {
            blocks: Arc::new(vec![Block {
                kind: Kind::Column as i32,
                gap: 1,
                children: vec![
                    Block {
                        text: "hidden".into(),
                        height: Some(0),
                        ..Default::default()
                    },
                    Block {
                        text: "abcdefgh".into(),
                        height: Some(1),
                        ..Default::default()
                    },
                    Block {
                        text: "hidden".into(),
                        height: Some(0),
                        ..Default::default()
                    },
                    Block {
                        text: "last".into(),
                        height: Some(2),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }]),
            theme: Arc::default(),
            expanded: false,
        };
        let rows = content.render(4);
        assert_eq!(rows.len(), 4);
        assert_eq!(crate::plain(&rows[0]).iter().collect::<String>(), "abcd");
        assert!(rows[1].is_empty());
        assert_eq!(crate::plain(&rows[2]).iter().collect::<String>(), "last");
        assert!(rows[3].is_empty());
    }

    #[test]
    fn nested_collapsed_source_expands_without_truncating_its_summary() {
        let mut content = Content {
            blocks: Arc::new(vec![Block {
                kind: Kind::Column as i32,
                children: vec![Block {
                    kind: Kind::Column as i32,
                    collapsed: vec![Block {
                        text: "first … (+1 lines)".into(),
                        ..Default::default()
                    }],
                    children: vec![Block {
                        text: "first\nsecond".into(),
                        ..Default::default()
                    }],
                    ..Default::default()
                }],
                ..Default::default()
            }]),
            theme: Arc::default(),
            expanded: false,
        };
        assert!(content.collapsible());
        assert_eq!(content.render(80).len(), 1);
        assert_eq!(
            crate::plain(&content.render(80)[0])
                .iter()
                .collect::<String>(),
            "first … (+1 lines)"
        );
        content.expanded = true;
        assert_eq!(content.render(80).len(), 2);
        assert_eq!(
            crate::plain(&content.render(80)[1])
                .iter()
                .collect::<String>(),
            "second"
        );
    }

    #[test]
    fn nonwrapping_frame_clips_the_whole_line_with_styled_ellipsis() {
        let content = Content {
            blocks: Arc::new(vec![Block {
                kind: Kind::Frame as i32,
                insets: vec![0, 1, 0, 2],
                no_wrap: true,
                overflow: crate::protocol::block::Overflow::Ellipsis as i32,
                children: vec![Block {
                    lines: vec![crate::protocol::Line {
                        spans: vec![
                            crate::protocol::Span {
                                text: "task ".into(),
                                sgr: "2".into(),
                            },
                            crate::protocol::Span {
                                text: "界界界界界界界".into(),
                                sgr: "31".into(),
                            },
                        ],
                    }],
                    ..Default::default()
                }],
                ..Default::default()
            }]),
            theme: Arc::default(),
            expanded: true,
        };
        let lines = content.render(16);
        assert_eq!(lines.len(), 1);
        assert_eq!(
            crate::plain(&lines[0]).iter().collect::<String>(),
            "  task 界界界…  "
        );
        assert!(lines[0]
            .iter()
            .any(|span| span.text == "…" && span.sgr == "31"));
    }

    #[test]
    fn grid_preserves_nowrap_labels_before_folding_values() {
        let cell = |text: &str, no_wrap, right| Block {
            kind: Kind::Frame as i32,
            no_wrap,
            insets: vec![0, right, 0, 0],
            children: vec![Block {
                text: text.into(),
                ..Default::default()
            }],
            ..Default::default()
        };
        let content = Content {
            blocks: Arc::new(vec![Block {
                kind: Kind::Grid as i32,
                children: vec![Block {
                    children: vec![
                        cell("directory", true, 2),
                        cell("123456789012345678", false, 0),
                    ],
                    ..Default::default()
                }],
                ..Default::default()
            }]),
            theme: Arc::default(),
            expanded: true,
        };
        let lines = content.render(20);
        assert_eq!(lines.len(), 2);
        assert_eq!(
            crate::plain(&lines[0]).iter().collect::<String>(),
            "directory  123456789"
        );
        assert_eq!(
            crate::plain(&lines[1]).iter().collect::<String>(),
            "           012345678"
        );
    }

    #[test]
    fn unexpanded_grid_keeps_intrinsic_column_widths() {
        let content = Content {
            blocks: Arc::new(vec![Block {
                kind: Kind::Grid as i32,
                children: vec![Block {
                    children: vec![
                        Block {
                            children: vec![Block {
                                text: "└ ".into(),
                                ..Default::default()
                            }],
                            kind: Kind::Frame as i32,
                            insets: vec![0; 4],
                            ..Default::default()
                        },
                        Block {
                            children: vec![Block {
                                text: "Reading file.py".into(),
                                ..Default::default()
                            }],
                            kind: Kind::Frame as i32,
                            insets: vec![0; 4],
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                }],
                ..Default::default()
            }]),
            theme: Arc::default(),
            expanded: true,
        };
        let lines = content.render(80);
        assert_eq!(lines.len(), 1);
        assert_matches!(lines.first(), Some(line) if crate::plain(line).iter().collect::<String>() == "└ Reading file.py");
    }
}
