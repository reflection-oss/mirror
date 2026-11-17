//! Syntax scopes mapped onto the application's already-quantized semantic palette.

use std::sync::{Arc, LazyLock};

use syntect::parsing::{ParseState, ScopeStack, SyntaxSet};

use crate::{protocol::Theme, Line, Span};

pub fn code_lines(
    text: &str,
    language: &str,
    theme: &Theme,
    palette: &str,
    source: Option<crate::Source>,
) -> Vec<Line> {
    let syntax = SYNTAXES
        .find_syntax_by_token(language)
        .unwrap_or_else(|| SYNTAXES.find_syntax_plain_text());
    let mut parser = ParseState::new(syntax);
    let mut scopes = ScopeStack::new();
    let base = theme.roles.get("code").unwrap_or(&theme.text);
    let mut lines: Vec<Line> = text
        .split_inclusive('\n')
        .map(|line| {
            let Ok(operations) = parser.parse_line(line, &SYNTAXES) else {
                parser = ParseState::new(syntax);
                scopes = ScopeStack::new();
                return vec![Span {
                    action: None,
                    source: None,
                    text: line.trim_end_matches('\n').into(),
                    sgr: base.clone(),
                }];
            };
            let mut result: Line = vec![];
            let mut start = 0;
            for (end, operation) in operations {
                push(
                    &mut result,
                    &line[start..end],
                    &scopes,
                    theme,
                    base,
                    palette,
                );
                if scopes.apply(&operation).is_err() {
                    scopes = ScopeStack::new();
                }
                start = end;
            }
            push(&mut result, &line[start..], &scopes, theme, base, palette);
            result
        })
        .collect();
    if let Some(source) = source {
        crate::annotate_lines(&mut lines, source);
    }
    lines
}

pub fn code_block(
    text: &str,
    language: &str,
    theme: &Theme,
    columns: u16,
    source: Option<crate::Source>,
) -> Vec<Line> {
    let padding = usize::from(columns >= 6);
    let width = columns.saturating_sub(1 + 2 * padding as u16).max(1);
    let mut lines = vec![];
    for line in code_lines(text, language, theme, "code", source) {
        for mut row in crate::wrap_lines(&line, width, false) {
            row.insert(
                0,
                Span {
                    action: None,
                    source: None,
                    text: " ".repeat(padding),
                    sgr: String::new(),
                },
            );
            lines.push(row);
        }
    }
    let mut panel = code_panel(lines, columns, theme, 2);
    if columns >= 8 && text.len() <= 75_000 {
        panel[0] = vec![
            Span {
                text: "┃".into(),
                sgr: theme.roles.get("code-rail").cloned().unwrap_or_default(),
                action: None,
                source: None,
            },
            Span {
                text: " ".repeat(columns as usize - 6),
                sgr: theme.roles.get("code").cloned().unwrap_or_default(),
                action: None,
                source: None,
            },
            Span {
                source: None,
                text: "Copy ".into(),
                sgr: theme.muted.clone(),
                action: Some(Arc::new(crate::Action::Copy(
                    text.trim_end_matches('\n').into(),
                ))),
            },
        ];
    }
    panel
}

pub fn code_panel(lines: Vec<Line>, columns: u16, theme: &Theme, top: usize) -> Vec<Line> {
    let rail = Span {
        action: None,
        source: None,
        text: if columns >= 2 { "┃" } else { "" }.into(),
        sgr: theme.roles.get("code-rail").cloned().unwrap_or_default(),
    };
    let background = theme.roles.get("code").cloned().unwrap_or_default();
    let width = columns.saturating_sub(u16::from(columns >= 2)) as usize;
    let blank = vec![
        rail.clone(),
        Span {
            action: None,
            source: None,
            text: " ".repeat(width),
            sgr: background.clone(),
        },
    ];
    let mut result = vec![blank.clone(); top];
    for mut line in lines {
        let used = line
            .iter()
            .map(|span| crate::width(&span.text))
            .sum::<usize>();
        for span in &mut line {
            span.sgr = format!("{background};{}", span.sgr)
                .trim_matches(';')
                .into();
        }
        line.insert(0, rail.clone());
        line.push(Span {
            action: None,
            source: None,
            text: " ".repeat(width.saturating_sub(used)),
            sgr: background.clone(),
        });
        result.push(line);
    }
    result.push(blank);
    result
}

static SYNTAXES: LazyLock<SyntaxSet> = LazyLock::new(SyntaxSet::load_defaults_newlines);

fn push(
    line: &mut Line,
    text: &str,
    scopes: &ScopeStack,
    theme: &Theme,
    base: &str,
    palette: &str,
) {
    let text = text.trim_end_matches('\n');
    if text.is_empty() {
        return;
    }
    let category = scopes
        .as_slice()
        .iter()
        .rev()
        .find_map(|scope| {
            let scope = scope.to_string();
            [
                ("constant.character.escape", "escape"),
                ("comment", "comment"),
                ("string", "string"),
                ("keyword.operator", "operator"),
                ("keyword", "keyword"),
                ("storage", "keyword"),
                ("constant.numeric", "number"),
                ("constant", "constant"),
                ("entity.name.function", "function"),
                ("entity.name.class", "class"),
                ("support", "builtin"),
                ("variable", "name"),
            ]
            .into_iter()
            .find_map(|(prefix, category)| scope.starts_with(prefix).then_some(category))
        })
        .unwrap_or("text");
    let prefix = if palette.is_empty() { "code" } else { palette };
    let syntax = theme
        .roles
        .get(&format!("syntax.{prefix}.{category}"))
        .or_else(|| theme.roles.get(&format!("syntax.{prefix}.text")));
    let role = scopes.as_slice().iter().rev().find_map(|scope| {
        let scope = scope.to_string();
        if scope.starts_with("comment") {
            Some("muted")
        } else if scope.starts_with("string") {
            Some("success")
        } else if scope.starts_with("keyword") || scope.starts_with("storage") {
            Some("accent.bold")
        } else if scope.starts_with("constant") {
            Some("warning")
        } else if scope.starts_with("entity.name") {
            Some("info")
        } else {
            None
        }
    });
    let sgr = syntax
        .or_else(|| role.and_then(|role| theme.roles.get(role)))
        .map_or_else(
            || base.into(),
            |style| format!("{base};{style}").trim_matches(';').into(),
        );
    match line.last_mut() {
        Some(span) if span.sgr == sgr => span.text.push_str(text),
        _ => line.push(Span {
            action: None,
            source: None,
            text: text.into(),
            sgr,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn syntax_palette_is_independent_of_language_and_semantic_colors() {
        let theme = Theme {
            roles: [
                ("code".into(), "49".into()),
                ("syntax.shell.text".into(), "37".into()),
                ("syntax.shell.string".into(), "33".into()),
                ("success".into(), "32".into()),
            ]
            .into(),
            ..Default::default()
        };
        assert_eq!(
            code_lines("echo 'text'", "bash", &theme, "shell", None),
            vec![vec![
                Span {
                    action: None,
                    source: None,
                    text: "echo ".into(),
                    sgr: "49;37".into()
                },
                Span {
                    action: None,
                    source: None,
                    text: "'text'".into(),
                    sgr: "49;33".into()
                },
            ]]
        );
    }

    #[test]
    fn syntax_uses_the_supplied_palette_without_introducing_truecolor() {
        let theme = Theme {
            roles: [
                ("accent.bold".into(), "1;34".into()),
                ("success".into(), "32".into()),
            ]
            .into(),
            ..Default::default()
        };
        let lines = code_lines(
            "def answer():\n    return \"text\"\n",
            "python",
            &theme,
            "code",
            None,
        );
        assert_eq!(lines[0].first().expect("keyword").sgr, "1;34");
        assert_eq!(lines[1].last().expect("string").sgr, "32");
        assert_eq!(
            lines[1]
                .iter()
                .map(|span| span.text.as_str())
                .collect::<String>(),
            "    return \"text\""
        );
    }
}
