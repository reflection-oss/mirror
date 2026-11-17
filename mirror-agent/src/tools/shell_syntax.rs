use std::collections::HashMap;

use tree_sitter::{Node, Parser};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Word {
    Literal(String),
    Expanded,
}

impl Word {
    pub(super) fn literal(&self) -> Option<&str> {
        match self {
            Self::Literal(text) => Some(text),
            Self::Expanded => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ShellCommand {
    pub(super) words: Vec<Word>,
    pub(super) upstream: Option<usize>,
    pub(super) subshells: Vec<usize>,
}

pub(super) fn commands(script: &str) -> Vec<ShellCommand> {
    let mut parser = Parser::new();
    if parser
        .set_language(&tree_sitter_bash::LANGUAGE.into())
        .is_err()
    {
        return Vec::new();
    }
    let Some(tree) = parser.parse(script, None) else {
        return Vec::new();
    };
    let mut collector = Collector {
        source: script.as_bytes(),
        commands: Vec::new(),
        index_by_node: HashMap::new(),
        subshells: Vec::new(),
        next_subshell: 0,
    };
    collector.visit(tree.root_node());
    collector.commands
}

struct Collector<'a> {
    source: &'a [u8],
    commands: Vec<ShellCommand>,
    index_by_node: HashMap<usize, usize>,
    subshells: Vec<usize>,
    next_subshell: usize,
}

impl Collector<'_> {
    fn visit(&mut self, node: Node<'_>) {
        match node.kind() {
            "command" => self.command(node),
            "redirected_statement" => self.redirected_statement(node),
            "pipeline" => self.pipeline(node),
            "heredoc_redirect" => self.heredoc_redirect(node),
            "subshell" | "command_substitution" | "process_substitution" => {
                self.in_subshell(|collector| collector.visit_children(node));
            }
            "comment" => {}
            _ => self.visit_children(node),
        }
    }

    fn in_subshell(&mut self, visit: impl FnOnce(&mut Self)) {
        self.subshells.push(self.next_subshell);
        self.next_subshell += 1;
        visit(self);
        self.subshells.pop();
    }

    fn heredoc_redirect(&mut self, node: Node<'_>) {
        let mut cursor = node.walk();
        let quoted = node
            .named_children(&mut cursor)
            .find(|child| child.kind() == "heredoc_start")
            .is_some_and(|start| self.text(start).contains(['\'', '"', '\\']));
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            if !(quoted && child.kind() == "heredoc_body") {
                self.visit(child);
            }
        }
    }

    fn visit_children(&mut self, node: Node<'_>) {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            self.visit(child);
        }
    }

    fn command(&mut self, node: Node<'_>) {
        let mut words = Vec::new();
        let mut cursor = node.walk();
        for (index, child) in node.children(&mut cursor).enumerate() {
            match node.field_name_for_child(index as u32) {
                Some("name" | "argument") => words.push(self.word(child)),
                Some("redirect") => words.extend(self.arguments_after_redirect(child)),
                _ => {}
            }
        }
        self.index_by_node.insert(node.id(), self.commands.len());
        self.commands.push(ShellCommand {
            words,
            upstream: None,
            subshells: self.subshells.clone(),
        });
        self.visit_children(node);
    }

    fn redirected_statement(&mut self, node: Node<'_>) {
        let mut target = None;
        let mut cursor = node.walk();
        for (index, child) in node.children(&mut cursor).enumerate() {
            match node.field_name_for_child(index as u32) {
                Some("body") => {
                    self.visit(child);
                    target = self.last_command(child);
                }
                Some("redirect") => {
                    let arguments = self.arguments_after_redirect(child);
                    if let Some(target) = target {
                        self.commands[target].words.extend(arguments);
                    }
                    self.visit(child);
                }
                _ if child.is_named() => self.visit(child),
                _ => {}
            }
        }
    }

    fn arguments_after_redirect(&self, redirect: Node<'_>) -> Vec<Word> {
        if redirect.kind() != "file_redirect" {
            return Vec::new();
        }
        let mut cursor = redirect.walk();
        redirect
            .children_by_field_name("destination", &mut cursor)
            .skip(1)
            .map(|destination| self.word(destination))
            .collect()
    }

    fn pipeline(&mut self, node: Node<'_>) {
        let mut upstream = None;
        let mut cursor = node.walk();
        for stage in node.named_children(&mut cursor) {
            self.visit(stage);
            if let (Some(command), Some(upstream)) = (self.first_command(stage), upstream) {
                self.commands[command].upstream = Some(upstream);
            }
            upstream = self.last_command(stage);
        }
    }

    fn first_command(&self, node: Node<'_>) -> Option<usize> {
        self.end_command(node, |node| node.named_child(0))
    }

    fn last_command(&self, node: Node<'_>) -> Option<usize> {
        self.end_command(node, |node| {
            node.named_child(node.named_child_count().checked_sub(1)?)
        })
    }

    fn end_command(
        &self,
        node: Node<'_>,
        end: impl Fn(Node<'_>) -> Option<Node<'_>> + Copy,
    ) -> Option<usize> {
        match node.kind() {
            "command" => self.index_by_node.get(&node.id()).copied(),
            "redirected_statement" => self.end_command(node.child_by_field_name("body")?, end),
            "list" | "pipeline" | "subshell" | "compound_statement" => {
                self.end_command(end(node)?, end)
            }
            _ => None,
        }
    }

    fn word(&self, node: Node<'_>) -> Word {
        match node.kind() {
            "word" | "number" => Word::Literal(unescape(self.text(node))),
            "raw_string" => Word::Literal(trim_quotes(self.text(node), "'", "'").to_string()),
            "ansi_c_string" => Word::Literal(trim_quotes(self.text(node), "$'", "'").to_string()),
            "string" | "translated_string" | "concatenation" | "command_name" => self.joined(node),
            "string_content" => Word::Literal(unescape_double_quoted(self.text(node))),
            "simple_expansion"
            | "expansion"
            | "command_substitution"
            | "process_substitution"
            | "arithmetic_expansion" => Word::Expanded,
            _ => Word::Literal(self.text(node).to_string()),
        }
    }

    fn joined(&self, node: Node<'_>) -> Word {
        let mut text = String::new();
        let mut cursor = node.walk();
        for part in node.named_children(&mut cursor) {
            match self.word(part) {
                Word::Literal(literal) => text.push_str(&literal),
                Word::Expanded => return Word::Expanded,
            }
        }
        Word::Literal(text)
    }

    fn text(&self, node: Node<'_>) -> &str {
        node.utf8_text(self.source).unwrap_or_default()
    }
}

fn trim_quotes<'a>(text: &'a str, open: &str, close: &str) -> &'a str {
    let text = text.strip_prefix(open).unwrap_or(text);
    text.strip_suffix(close).unwrap_or(text)
}

fn unescape(text: &str) -> String {
    let mut unescaped = String::with_capacity(text.len());
    let mut characters = text.chars();
    while let Some(character) = characters.next() {
        if character == '\\' {
            unescaped.extend(characters.next().filter(|escaped| *escaped != '\n'));
        } else {
            unescaped.push(character);
        }
    }
    unescaped
}

fn unescape_double_quoted(text: &str) -> String {
    let mut unescaped = String::with_capacity(text.len());
    let mut characters = text.chars().peekable();
    while let Some(character) = characters.next() {
        match characters.peek() {
            Some(escaped) if character == '\\' && matches!(escaped, '"' | '\\' | '$' | '`') => {
                unescaped.push(*escaped);
                characters.next();
            }
            Some('\n') if character == '\\' => {
                characters.next();
            }
            _ => unescaped.push(character),
        }
    }
    unescaped
}

#[cfg(test)]
#[path = "shell_syntax_test.rs"]
mod tests;
