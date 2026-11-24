//! Python-ordered transcript sources, with retained layout and row indexes.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::io;
use std::sync::Arc;

use prost::Message as _;

use crate::{
    invalid, protocol, retained_source_bytes, retained_theme_bytes, validate_element,
    validate_theme, Content, Element, Row, Span, MAX_CACHE, MAX_ELEMENTS, MAX_RETAINED_SOURCE,
    MAX_SOURCE,
};

/// Mutations preserve caller-supplied order; IDs never imply an ordering.
#[derive(Debug, Default)]
pub struct Document {
    elements: HashMap<u32, Element>,
    order: Vec<u32>,
    indexes: HashMap<u32, usize>,
    sizes: HashMap<u32, SourceSize>,
    theme: Arc<protocol::Theme>,
    ends: Vec<usize>,
    width: u16,
    cache_bytes: usize,
    cached: BTreeSet<u32>,
    sticky: BTreeSet<usize>,
}

#[derive(Debug)]
struct SourceSize {
    wire: usize,
    retained: usize,
}

/// A logical character position survives wrapping and changes preceding its block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Position {
    pub id: u32,
    pub line: usize,
    pub character: u32,
    pub source: Option<crate::Source>,
}

impl Document {
    /// Validate the entire update before mutating committed sources or geometry.
    pub fn update(&mut self, update: protocol::TranscriptUpdate) -> io::Result<()> {
        self.update_bounded(update, MAX_RETAINED_SOURCE)
    }

    pub fn update_bounded(
        &mut self,
        update: protocol::TranscriptUpdate,
        budget: usize,
    ) -> io::Result<()> {
        let order = update
            .order
            .as_ref()
            .map_or(self.order.as_slice(), |order| &order.ids);
        if order.len() > MAX_ELEMENTS {
            return Err(invalid("transcript exceeds 100000 elements"));
        }
        let ids: HashSet<_> = order.iter().copied().collect();
        if ids.len() != order.len() || ids.contains(&0) {
            return Err(invalid("transcript order contains duplicate or zero IDs"));
        }
        let mut replacements = HashMap::new();
        for element in &update.set {
            validate_element(element)?;
            if !ids.contains(&element.id)
                || replacements
                    .insert(
                        element.id,
                        SourceSize {
                            wire: element.encoded_len(),
                            retained: retained_source_bytes(element),
                        },
                    )
                    .is_some()
            {
                return Err(invalid(
                    "transcript update has duplicate or unordered replacements",
                ));
            }
        }
        let mut wire_bytes = 0usize;
        let mut retained = retained_theme_bytes(update.theme.as_ref().unwrap_or(&self.theme));
        for id in order {
            let size = replacements
                .get(id)
                .or_else(|| self.sizes.get(id))
                .ok_or_else(|| invalid("transcript order references a missing element"))?;
            wire_bytes += size.wire;
            retained += size.retained;
        }
        if wire_bytes > MAX_SOURCE {
            return Err(invalid("transcript exceeds 32 MiB serialized source"));
        }
        if retained > budget {
            return Err(invalid("transcript exceeds retained source budget"));
        }
        if let Some(theme) = &update.theme {
            validate_theme(theme)?;
        }
        self.apply(update, ids, replacements);
        Ok(())
    }

    fn apply(
        &mut self,
        update: protocol::TranscriptUpdate,
        ids: HashSet<u32>,
        sizes: HashMap<u32, SourceSize>,
    ) {
        let reordered = update.order.is_some();
        if let Some(order) = update.order {
            let unchanged = self
                .order
                .iter()
                .zip(&order.ids)
                .take_while(|(a, b)| a == b)
                .count();
            self.ends.truncate(unchanged);
            self.order = order.ids;
            self.indexes = self
                .order
                .iter()
                .enumerate()
                .map(|(index, &id)| (id, index))
                .collect();
            self.elements.retain(|id, element| {
                if ids.contains(id) {
                    return true;
                }
                self.cache_bytes -= element.cache_bytes();
                self.cached.remove(id);
                false
            });
            self.sizes.retain(|id, _| ids.contains(id));
            self.elements.shrink_to_fit();
            self.sizes.shrink_to_fit();
            self.order.shrink_to_fit();
            self.ends.shrink_to_fit();
        }
        if let Some(theme) = update.theme.filter(|theme| theme != self.theme.as_ref()) {
            self.theme = Arc::new(theme);
            for element in self.elements.values_mut() {
                if let Some(content) = &element.content {
                    let mut content = content.clone();
                    content.theme = self.theme.clone();
                    let sticky = element.sticky.clone();
                    self.cache_bytes -= element.cache_bytes();
                    self.cached.remove(&element.id);
                    *element = Element::from_content(element.id, content);
                    element.sticky = sticky;
                }
            }
            self.ends.clear();
        }
        for source in update.set {
            let expanded = self
                .elements
                .get(&source.id)
                .and_then(|element| element.content.as_ref())
                .is_some_and(|content| content.expanded);
            let mut element = if source.blocks.is_empty() {
                Element::new(
                    source.id,
                    source
                        .lines
                        .into_iter()
                        .map(|line| line.spans.into_iter().map(Span::from).collect())
                        .collect(),
                )
            } else {
                Element::from_content(
                    source.id,
                    Content {
                        blocks: Arc::new(source.blocks),
                        theme: self.theme.clone(),
                        expanded,
                    },
                )
            };
            element.sticky = source.sticky.map(Span::from);
            if let Some(&index) = self.indexes.get(&source.id) {
                self.ends.truncate(index);
                if element.sticky.is_some() {
                    self.sticky.insert(index);
                } else {
                    self.sticky.remove(&index);
                }
            }
            if let Some(previous) = self.elements.insert(source.id, element) {
                self.cache_bytes -= previous.cache_bytes();
                self.cached.remove(&source.id);
            }
        }
        self.sizes.extend(sizes);
        if reordered {
            self.sticky = self
                .order
                .iter()
                .enumerate()
                .filter_map(|(index, id)| {
                    self.elements
                        .get(id)
                        .and_then(|element| element.sticky.as_ref())
                        .map(|_| index)
                })
                .collect();
        }
    }

    pub fn theme(&self) -> &protocol::Theme {
        &self.theme
    }

    pub fn source_bytes(&self) -> usize {
        self.sizes.values().map(|size| size.retained).sum::<usize>()
            + retained_theme_bytes(&self.theme)
    }

    pub fn contains(&self, id: u32) -> bool {
        self.elements.contains_key(&id)
    }

    pub fn total(&mut self, width: u16) -> usize {
        let width = width.clamp(1, 4096);
        if width != self.width {
            self.width = width;
            self.ends.clear();
        }
        while self.ends.len() < self.order.len() {
            let id = self.order[self.ends.len()];
            let before = self.elements.get(&id).map_or(0, Element::cache_bytes);
            let height = self
                .elements
                .get_mut(&id)
                .map_or(0, |element| element.total(width));
            self.account_cache(id, before);
            self.ends
                .push(self.ends.last().copied().unwrap_or(0) + height);
        }
        self.ends.last().copied().unwrap_or(0)
    }

    pub fn row(&mut self, row: usize, width: u16) -> Option<(Row, Position)> {
        self.rows(row, width, 1).into_iter().next()
    }

    /// Project a viewport before evicting each element's temporary layout.
    pub fn rows(&mut self, mut row: usize, width: u16, height: u16) -> Vec<(Row, Position)> {
        let end = row.saturating_add(height as usize).min(self.total(width));
        let mut rendered = Vec::with_capacity(end.saturating_sub(row));
        let first = self.ends.partition_point(|&end| end <= row);
        for index in first..self.order.len() {
            if row >= end {
                break;
            }
            let id = self.order[index];
            let start = self.before(index);
            let Some(element) = self.elements.get_mut(&id) else {
                break;
            };
            let before = element.cache_bytes();
            let boundary = end.min(self.ends[index]);
            while row < boundary {
                let Some((line, offset)) = element.locate(row - start, width) else {
                    break;
                };
                let Some(content) = element.rows(line, width).get(row - start - offset) else {
                    break;
                };
                rendered.push((
                    content.clone(),
                    Position {
                        id,
                        line,
                        character: content.s,
                        source: content.source,
                    },
                ));
                row += 1;
            }
            self.account_cache(id, before);
        }
        rendered
    }

    pub fn resolve(&mut self, position: &Position, width: u16) -> Option<usize> {
        self.total(width);
        let index = *self.indexes.get(&position.id)?;
        let start = self.before(index);
        let element = self.elements.get_mut(&position.id)?;
        let before = element.cache_bytes();
        element.total(width);
        let resolved = element.lines().len().checked_sub(1).map(|last| {
            let (line, character) = position
                .source
                .and_then(|source| element.source_position(source))
                .unwrap_or((position.line.min(last), position.character));
            let first = element.start(line);
            let rows = element.rows(line, width);
            let offset = rows
                .partition_point(|row| row.s <= character)
                .saturating_sub(1);
            start + first + offset
        });
        self.account_cache(position.id, before);
        resolved
    }

    pub fn sticky(&mut self, row: usize, width: u16) -> Option<Span> {
        self.total(width);
        let index = self.ends.partition_point(|&end| end <= row);
        let offset = *self.sticky.range(..=index).next_back()?;
        let sticky = self.elements.get(&self.order[offset])?.sticky.as_ref()?;
        (offset != index || row != self.before(index)).then(|| sticky.clone())
    }

    pub fn toggle(&mut self, id: u32) -> bool {
        let Some(element) = self.elements.get_mut(&id) else {
            return false;
        };
        let Some(mut content) = element.content.clone().filter(Content::collapsible) else {
            return false;
        };
        content.expanded = !content.expanded;
        let sticky = element.sticky.clone();
        self.cache_bytes -= element.cache_bytes();
        self.cached.remove(&id);
        *element = Element::from_content(id, content);
        element.sticky = sticky;
        self.ends.truncate(self.indexes[&id]);
        true
    }

    pub fn cache_bytes(&self) -> usize {
        self.cache_bytes
    }

    pub fn trim_caches(&mut self, budget: usize) {
        while self.cache_bytes > budget {
            let Some(id) = self.cached.pop_first() else {
                break;
            };
            if let Some(element) = self.elements.get_mut(&id) {
                self.cache_bytes -= element.cache_bytes();
                element.evict_layout();
            }
        }
    }

    fn account_cache(&mut self, id: u32, before: usize) {
        let after = self.elements.get(&id).map_or(0, Element::cache_bytes);
        self.cache_bytes = self.cache_bytes - before + after;
        if after == 0 {
            self.cached.remove(&id);
        } else {
            self.cached.insert(id);
        }
        self.trim_caches(MAX_CACHE);
    }

    fn before(&self, index: usize) -> usize {
        self.ends.get(index.wrapping_sub(1)).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    fn element(id: u32, text: &str) -> protocol::Element {
        protocol::Element {
            id,
            lines: vec![protocol::Line {
                spans: vec![protocol::Span {
                    text: text.into(),
                    sgr: String::new(),
                }],
            }],
            ..Default::default()
        }
    }

    #[test]
    fn explicit_order_prevents_queued_messages_overtaking_settled_answers() {
        let mut doc = Document::default();
        doc.update(protocol::TranscriptUpdate {
            set: vec![element(9, "queued"), element(2, "answer")],
            order: Some(protocol::TranscriptOrder { ids: vec![2, 9] }),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(doc.row(0, 80).unwrap().0.b.as_ref(), b"answer");
        doc.update(protocol::TranscriptUpdate {
            set: vec![element(2, "completed answer")],
            ..Default::default()
        })
        .unwrap();
        assert_eq!(doc.row(1, 80).unwrap().0.b.as_ref(), b"queued");
    }

    #[test]
    fn viewport_rows_cross_elements_after_eviction_and_stop_at_the_end() {
        let mut doc = Document::default();
        doc.update(protocol::TranscriptUpdate {
            set: vec![element(9, "end"), element(2, "abcdefgh")],
            order: Some(protocol::TranscriptOrder { ids: vec![2, 9] }),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(doc.total(4), 3);
        doc.trim_caches(0);
        let rows = doc.rows(1, 4, 3);
        let positions = rows
            .iter()
            .map(|(row, position)| (row.text.as_ref(), position.id, position.character))
            .collect::<Vec<_>>();
        assert_eq!(positions, [("efgh", 2, 4), ("end", 9, 0)]);
    }

    #[test]
    fn rejected_update_leaves_order_and_sources_unchanged() {
        let mut doc = Document::default();
        doc.update(protocol::TranscriptUpdate {
            set: vec![element(1, "before")],
            order: Some(protocol::TranscriptOrder { ids: vec![1] }),
            ..Default::default()
        })
        .unwrap();
        let result = doc.update(protocol::TranscriptUpdate {
            set: vec![element(1, "after")],
            order: Some(protocol::TranscriptOrder { ids: vec![1, 7] }),
            ..Default::default()
        });
        assert_matches!(result, Err(_));
        assert_eq!(doc.row(0, 80).unwrap().0.b.as_ref(), b"before");
    }

    #[test]
    fn retained_budget_charges_empty_source_structures_and_rejects_atomically() {
        let mut doc = Document::default();
        doc.update(protocol::TranscriptUpdate {
            set: vec![element(1, "before")],
            order: Some(protocol::TranscriptOrder { ids: vec![1] }),
            ..Default::default()
        })
        .unwrap();
        let before = doc.source_bytes();
        let result = doc.update_bounded(
            protocol::TranscriptUpdate {
                set: vec![protocol::Element {
                    id: 2,
                    blocks: vec![protocol::Block::default(); 100],
                    ..Default::default()
                }],
                order: Some(protocol::TranscriptOrder { ids: vec![2] }),
                ..Default::default()
            },
            4096,
        );
        assert_matches!(result, Err(error) if error.to_string().contains("retained source budget"));
        assert_eq!(doc.source_bytes(), before);
        assert_eq!(doc.row(0, 80).unwrap().0.b.as_ref(), b"before");
        doc.update_bounded(
            protocol::TranscriptUpdate {
                order: Some(protocol::TranscriptOrder { ids: vec![] }),
                ..Default::default()
            },
            before,
        )
        .unwrap();
        assert!(doc.source_bytes() < before);
        assert_eq!(doc.total(80), 0);
    }

    #[test]
    fn evicted_layout_rebuilds_anchors_and_rows_after_resize() {
        let mut doc = Document::default();
        doc.update(protocol::TranscriptUpdate {
            set: vec![protocol::Element {
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
            }],
            order: Some(protocol::TranscriptOrder { ids: vec![1] }),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(doc.cache_bytes(), 0);
        let position = doc.row(2, 4).unwrap().1;
        assert!(doc.cache_bytes() > 0);
        doc.trim_caches(0);
        assert_eq!(doc.cache_bytes(), 0);
        assert_eq!(doc.resolve(&position, 4), Some(2));
        assert_eq!(doc.row(2, 4).unwrap().0.b.as_ref(), b"ijkl");
        doc.trim_caches(0);
        assert_eq!(doc.resolve(&position, 8), Some(1));
        assert_eq!(doc.row(1, 8).unwrap().0.b.as_ref(), b"ijklmnop");
    }

    fn content_document(block: protocol::Block) -> Document {
        let mut document = Document::default();
        document
            .update(protocol::TranscriptUpdate {
                set: vec![protocol::Element {
                    id: 1,
                    blocks: vec![block],
                    ..Default::default()
                }],
                order: Some(protocol::TranscriptOrder { ids: vec![1] }),
                ..Default::default()
            })
            .unwrap();
        document
    }

    #[test]
    fn nested_panel_reflow_resolves_unicode_source_bytes() {
        let mut doc = content_document(protocol::Block {
            kind: protocol::block::Kind::Panel as i32,
            insets: vec![0; 4],
            children: vec![protocol::Block {
                kind: protocol::block::Kind::Frame as i32,
                insets: vec![0; 4],
                children: vec![protocol::Block {
                    text: "ab界e\u{301}👩🏽‍💻mnop\nlast".into(),
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        });
        let (row, position) = doc.row(2, 7).unwrap();
        assert_eq!(row.text.as_ref(), "│👩🏽‍💻mno│");
        doc.trim_caches(0);
        assert_eq!(doc.resolve(&position, 12), Some(1));
        assert_eq!(
            doc.row(1, 12).unwrap().0.text.as_ref(),
            "│ab界e\u{301}👩🏽‍💻mno│"
        );
    }

    #[test]
    fn markdown_list_reflow_preserves_the_item_source_position() {
        let mut doc = content_document(protocol::Block {
            kind: protocol::block::Kind::Markdown as i32,
            text: "- abcdefghijklmnop\n\nlast".into(),
            ..Default::default()
        });
        let (row, position) = doc.row(2, 7).unwrap();
        assert_eq!(row.text.as_ref(), "   ijkl");
        assert_eq!(doc.resolve(&position, 11), Some(1));
        assert_eq!(doc.row(1, 11).unwrap().0.text.as_ref(), "   ijklmnop");
    }

    #[test]
    fn markdown_record_anchor_tracks_the_value_not_its_repeated_label() {
        let mut doc = content_document(protocol::Block {
            kind: protocol::block::Kind::Markdown as i32,
            text: "| Key | Value |\n| --- | --- |\n| a | one |\n| b | two |".into(),
            ..Default::default()
        });
        let (row, position) = doc.row(5, 10).unwrap();
        assert_eq!(row.text.as_ref(), "Value: two");
        assert_eq!(doc.resolve(&position, 30), Some(4));
        assert_eq!(doc.row(4, 30).unwrap().0.text.trim_end(), "b   │ two");
    }

    #[test]
    fn diff_anchor_survives_changed_gutter_and_code_width() {
        let mut doc = content_document(protocol::Block {
            kind: protocol::block::Kind::Diff as i32,
            text: "@@ -1 +1 @@\n-abcdefghijklmnop\n+mnopqrstuvwx".into(),
            ..Default::default()
        });
        let (row, position) = doc.row(3, 13).unwrap();
        assert_eq!(row.text.as_ref(), "┃        ijkl");
        assert_eq!(doc.resolve(&position, 21), Some(1));
        assert_eq!(
            doc.row(1, 21).unwrap().0.text.as_ref(),
            "┃    1 - abcdefghijkl"
        );
    }

    #[test]
    fn fenced_code_anchor_survives_reflow_and_cache_eviction() {
        let mut doc = content_document(protocol::Block {
            kind: protocol::block::Kind::Markdown as i32,
            text: "```text\nabcdefghijklmnop\n```\n\nlast".into(),
            ..Default::default()
        });
        let (row, position) = doc.row(4, 7).unwrap();
        assert_eq!(row.text.as_ref(), "┃ ijkl ");
        doc.trim_caches(0);
        assert_eq!(doc.resolve(&position, 11), Some(3));
        assert_eq!(doc.row(3, 11).unwrap().0.text.as_ref(), "┃ ijklmnop ");
    }

    #[test]
    fn indexing_expanded_layout_enforces_the_cache_budget() {
        let mut doc = Document::default();
        doc.update(protocol::TranscriptUpdate {
            set: vec![protocol::Element {
                id: 1,
                blocks: vec![
                    protocol::Block {
                        text: "row".into(),
                        height: Some(256),
                        ..Default::default()
                    };
                    2048
                ],
                ..Default::default()
            }],
            order: Some(protocol::TranscriptOrder { ids: vec![1] }),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(doc.total(80), 2048 * 256);
        assert!(doc.cache_bytes() <= MAX_CACHE);
        assert_eq!(doc.row(256, 80).unwrap().0.b.as_ref(), b"row");
        assert!(doc.cache_bytes() <= MAX_CACHE);
    }

    #[test]
    fn evicted_plain_lines_rebuild_counts_without_losing_source() {
        let mut doc = Document::default();
        doc.update(protocol::TranscriptUpdate {
            set: vec![element(1, "abcdefgh")],
            order: Some(protocol::TranscriptOrder { ids: vec![1] }),
            ..Default::default()
        })
        .unwrap();
        let position = doc.row(1, 4).unwrap().1;
        doc.trim_caches(0);
        assert_eq!(doc.cache_bytes(), 0);
        assert_eq!(doc.resolve(&position, 4), Some(1));
        assert_eq!(doc.row(1, 4).unwrap().0.b.as_ref(), b"efgh");
    }

    #[test]
    fn reorder_preserves_the_source_anchor_and_removes_absent_blocks() {
        let mut doc = Document::default();
        doc.update(protocol::TranscriptUpdate {
            set: vec![element(1, "one"), element(2, "abcdefghi")],
            order: Some(protocol::TranscriptOrder { ids: vec![1, 2] }),
            ..Default::default()
        })
        .unwrap();
        let position = doc.row(2, 4).unwrap().1;
        doc.update(protocol::TranscriptUpdate {
            order: Some(protocol::TranscriptOrder { ids: vec![2] }),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(doc.resolve(&position, 8), Some(0));
        assert_eq!(doc.total(8), 2);
        assert_eq!(doc.contains(1), false);
    }
}
