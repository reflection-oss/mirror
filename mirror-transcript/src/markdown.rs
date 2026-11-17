//! Markdown source becomes styled logical lines, never terminal escape input.

use std::sync::Arc;

use pulldown_cmark::{CodeBlockKind, Event, Options, Parser, Tag, TagEnd};

use crate::{protocol::Theme, Line, Span};

#[derive(Debug)]
pub struct Markdown<'a> {
    theme: &'a Theme,
    lines: Vec<Line>,
    styles: Vec<String>,
    lists: Vec<Option<u64>>,
    items: Vec<(usize, String)>,
    quote: usize,
    code: bool,
    language: String,
    code_source: String,
    links: Vec<Option<Arc<crate::Action>>>,
    columns: u16,
    table: Option<Vec<Vec<Line>>>,
    cell_start: usize,
    source: crate::Source,
    code_origin: Option<crate::Source>,
}

impl<'a> Markdown<'a> {
    pub fn render(
        text: &str,
        sgr: &str,
        theme: &'a Theme,
        columns: u16,
        source: crate::Source,
    ) -> Vec<Line> {
        let mut markdown = Self {
            theme,
            lines: vec![vec![]],
            styles: vec![sgr.into()],
            lists: vec![],
            items: vec![],
            quote: 0,
            code: false,
            language: String::new(),
            code_source: String::new(),
            links: vec![],
            columns,
            table: None,
            cell_start: 0,
            source,
            code_origin: None,
        };
        let options =
            Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TASKLISTS;
        for (event, range) in Parser::new_ext(text, options).into_offset_iter() {
            markdown.source = source.shifted(range.start);
            markdown.event(event);
        }
        while markdown.lines.last().is_some_and(Vec::is_empty) {
            markdown.lines.pop();
        }
        markdown.lines
    }

    fn role(&self, role: &str) -> String {
        self.theme.roles.get(role).cloned().unwrap_or_default()
    }

    fn push(&mut self, text: &str, sgr: &str) {
        let action = self.links.last().cloned().flatten();
        let mut offset = 0;
        for (index, part) in text.split('\n').enumerate() {
            let source = Some(self.source.shifted(offset));
            offset += part.len() + 1;
            if index != 0 {
                self.lines.push(vec![]);
            }
            let line = self
                .lines
                .last_mut()
                .expect("render starts with a line and never removes it");
            if line.is_empty() && self.quote != 0 {
                line.push(Span {
                    action: None,
                    source: None,
                    text: "▌ ".repeat(self.quote.min(16)),
                    sgr: self
                        .theme
                        .roles
                        .get("quote-rail")
                        .cloned()
                        .unwrap_or_default(),
                });
            }
            if !part.is_empty() {
                match line.last_mut() {
                    Some(span)
                        if span.sgr == sgr
                            && span.action == action
                            && span.source.map(|source| source.shifted(span.text.len()))
                                == source =>
                    {
                        span.text.push_str(part)
                    }
                    _ => line.push(Span {
                        action: action.clone(),
                        source,
                        text: part.into(),
                        sgr: sgr.into(),
                    }),
                }
            }
        }
    }

    fn newline(&mut self, blank: bool) {
        if self.lines.last().is_some_and(|line| !line.is_empty()) {
            self.lines.push(vec![]);
        }
        if blank && self.lines.len() > 1 && !self.lines[self.lines.len() - 2].is_empty() {
            self.lines.push(vec![]);
        }
    }

    fn event(&mut self, event: Event<'_>) {
        let current = self.styles.last().cloned().unwrap_or_default();
        match event {
            Event::Start(tag) => {
                let mut style = current;
                match tag {
                    Tag::Heading { .. } => {
                        self.newline(true);
                        style = self.role("heading.bold");
                    }
                    Tag::Emphasis => style.push_str(";3"),
                    Tag::Strong => style.push_str(";1"),
                    Tag::Strikethrough => style.push_str(";9"),
                    Tag::Link { dest_url, .. } | Tag::Image { dest_url, .. } => {
                        style = self.role("link");
                        self.links.push(crate::link_action(&dest_url).map(Arc::new));
                    }
                    Tag::BlockQuote(_) => {
                        self.newline(true);
                        self.quote += 1;
                    }
                    Tag::List(start) => {
                        self.newline(false);
                        self.lists.push(start);
                    }
                    Tag::Item => {
                        self.newline(false);
                        let indent = "  ".repeat(self.lists.len().saturating_sub(1).min(16));
                        let marker = match self.lists.last_mut() {
                            Some(Some(number)) => {
                                let text = format!(" {number}. ");
                                *number = number.saturating_add(1);
                                text
                            }
                            _ => " • ".into(),
                        };
                        self.items
                            .push((self.lines.len() - 1, format!("{indent}{marker}")));
                    }
                    Tag::CodeBlock(kind) => {
                        self.newline(true);
                        self.code = true;
                        style = self.role("code");
                        let language = match kind {
                            CodeBlockKind::Fenced(language) => language.to_string(),
                            _ => String::new(),
                        };
                        self.language = language.split_whitespace().next().unwrap_or("").into();
                        self.code_source.clear();
                        self.code_origin = None;
                    }
                    Tag::Table(_) => {
                        self.newline(true);
                        self.table = Some(vec![]);
                    }
                    Tag::TableHead | Tag::TableRow => {
                        if let Some(table) = &mut self.table {
                            table.push(vec![]);
                        }
                    }
                    Tag::TableCell => self.cell_start = self.lines.len().saturating_sub(1),
                    _ => {}
                }
                self.styles.push(style.trim_start_matches(';').into());
            }
            Event::End(tag) => {
                self.styles.pop();
                match tag {
                    TagEnd::Paragraph => self.newline(self.lists.is_empty()),
                    TagEnd::Heading(_) => self.newline(true),
                    TagEnd::Item => {
                        if let Some((start, marker)) = self.items.pop() {
                            let mut body = self.lines.split_off(start);
                            while body.last().is_some_and(Vec::is_empty) {
                                body.pop();
                            }
                            if body.is_empty() {
                                body.push(vec![]);
                            }
                            let inset =
                                (crate::width(&marker) as u16).min(self.columns.saturating_sub(1));
                            let mut first = true;
                            for line in body {
                                for mut row in crate::wrap_lines(
                                    &line,
                                    self.columns.saturating_sub(inset).max(1),
                                    true,
                                ) {
                                    row.insert(
                                        0,
                                        Span {
                                            action: None,
                                            source: None,
                                            text: if first {
                                                marker.clone()
                                            } else {
                                                " ".repeat(inset as usize)
                                            },
                                            sgr: self.theme.muted.clone(),
                                        },
                                    );
                                    self.lines.push(row);
                                    first = false;
                                }
                            }
                        }
                        self.newline(false);
                    }
                    TagEnd::TableCell => {
                        let cell = self
                            .lines
                            .split_off(self.cell_start)
                            .into_iter()
                            .flatten()
                            .collect();
                        if let Some(row) = self.table.as_mut().and_then(|table| table.last_mut()) {
                            row.push(cell);
                        }
                        self.lines.push(vec![]);
                    }
                    TagEnd::Table => {
                        if let Some(table) = self.table.take() {
                            self.lines
                                .extend(crate::table_lines(table, self.columns, self.theme));
                            self.newline(true);
                        }
                    }
                    TagEnd::List(_) => {
                        self.lists.pop();
                        self.newline(self.lists.is_empty());
                    }
                    TagEnd::BlockQuote(_) => {
                        self.quote = self.quote.saturating_sub(1);
                        self.newline(true);
                    }
                    TagEnd::CodeBlock => {
                        self.code = false;
                        self.lines.pop();
                        self.lines.extend(crate::code_block(
                            &self.code_source,
                            &self.language,
                            self.theme,
                            self.columns,
                            self.code_origin,
                        ));
                        self.code_source.clear();
                        self.newline(true);
                    }
                    TagEnd::Link | TagEnd::Image => {
                        self.links.pop();
                    }
                    _ => {}
                }
            }
            Event::Text(text) | Event::Html(text) | Event::InlineHtml(text) => {
                if self.code {
                    self.code_origin.get_or_insert(self.source);
                    self.code_source.push_str(&text);
                } else {
                    self.push(&text, &current);
                }
            }
            Event::Code(text) => self.push(&text, &self.role("inline-code")),
            Event::SoftBreak if !self.code => self.push(" ", &current),
            Event::SoftBreak | Event::HardBreak => self.lines.push(vec![]),
            Event::Rule => {
                self.newline(true);
                self.push("────────", &self.theme.muted.clone());
                self.newline(true);
            }
            Event::TaskListMarker(done) => {
                self.push(if done { "☑ " } else { "☐ " }, &self.theme.accent.clone())
            }
            Event::FootnoteReference(text) => self.push(&format!("[{text}]"), &current),
            Event::InlineMath(text) | Event::DisplayMath(text) => self.push(&text, &current),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn empty_list_items_preserve_markers_and_following_content() {
        let lines = Markdown::render(
            "-\n- b\n\n-\n\ntext",
            "",
            &Theme::default(),
            80,
            crate::Source {
                block: 0,
                offset: 0,
            },
        );
        let text = lines
            .iter()
            .map(|line| {
                line.iter()
                    .map(|span| span.text.as_str())
                    .collect::<String>()
            })
            .collect::<Vec<_>>();
        assert_eq!(text, [" • ", " • b", " • ", "", "text"]);
    }

    #[test]
    fn markdown_table_cells_survive_narrow_layout() {
        let lines = Markdown::render(
            "| Key | Value |\n| --- | --- |\n| a | **bold** |",
            "",
            &Theme::default(),
            8,
            crate::Source {
                block: 0,
                offset: 0,
            },
        );
        let text = lines
            .iter()
            .map(|line| {
                line.iter()
                    .map(|span| span.text.as_str())
                    .collect::<String>()
            })
            .collect::<Vec<_>>();
        assert_eq!(text, ["", "Key: a", "Value: bold"]);
        assert_eq!(lines[2].last().expect("bold cell").sgr, "1");
    }
}
