//! Unified diff source with numbered, wrapping code rows and full-width tints.

use crate::{protocol::Theme, Line, Span};

pub fn diff_lines(
    source: &str,
    headers: bool,
    columns: u16,
    theme: &Theme,
    origin: Option<crate::Source>,
) -> Vec<Line> {
    let language = source
        .lines()
        .find_map(|line| line.strip_prefix("+++ "))
        .and_then(|path| std::path::Path::new(path.trim()).extension())
        .and_then(|extension| extension.to_str())
        .unwrap_or("");
    let header_padding = if source.lines().any(|line| hunk(line).is_some()) {
        if columns >= 12 {
            8
        } else {
            2
        }
    } else {
        1
    };
    let mut result = vec![];
    let (mut old, mut new) = (0u64, 0u64);
    let mut in_hunk = false;
    let mut offset = 0;
    for part in source.split_inclusive('\n') {
        let origin = origin.map(|origin| origin.shifted(offset));
        offset += part.len();
        let line = part
            .strip_suffix('\n')
            .unwrap_or(part)
            .trim_end_matches('\r');
        if let Some((before, after)) = hunk(line) {
            (old, new) = (before, after);
            in_hunk = true;
            continue;
        }
        let marker = line.chars().next().unwrap_or(' ');
        if !matches!(marker, ' ' | '+' | '-' | '\\') {
            in_hunk = false;
        }
        if !in_hunk {
            if headers || !(line.starts_with("--- ") || line.starts_with("+++ ")) {
                result.extend(header_lines(line, columns, header_padding, origin));
            }
            continue;
        }
        if marker == '\\' {
            continue;
        }
        let number = if marker == '-' { old } else { new };
        if marker != '+' {
            old = old.saturating_add(1);
        }
        if marker != '-' {
            new = new.saturating_add(1);
        }
        let gutter = if columns >= 12 {
            format!(" {number:>4} {marker} ")
        } else {
            format!("{marker} ")
        };
        let gutter_width = crate::width(&gutter).min(columns.saturating_sub(2) as usize);
        let code_width = columns.saturating_sub(1 + gutter_width as u16).max(1);
        let background = theme
            .roles
            .get(match marker {
                '+' => "addition",
                '-' => "deletion",
                _ => "code",
            })
            .cloned()
            .unwrap_or_default();
        let mut palette = theme.clone();
        palette.roles.insert("code".into(), background.clone());
        let code = line.get(1..).unwrap_or("");
        let highlighted = crate::code_lines(
            code,
            language,
            &palette,
            "diff",
            origin.map(|source| source.shifted(1)),
        )
        .into_iter()
        .next()
        .unwrap_or_default();
        let lines = crate::wrap_lines(&highlighted, code_width, false);
        for (index, mut line) in lines.into_iter().enumerate() {
            line.insert(
                0,
                Span {
                    action: None,
                    source: None,
                    text: if index == 0 {
                        gutter.clone()
                    } else {
                        " ".repeat(gutter_width)
                    },
                    sgr: format!("{};{background}", theme.muted),
                },
            );
            let used = line
                .iter()
                .map(|span| crate::width(&span.text))
                .sum::<usize>();
            line.push(Span {
                action: None,
                source: None,
                text: " ".repeat((columns.saturating_sub(1) as usize).saturating_sub(used)),
                sgr: background.clone(),
            });
            result.push(line);
        }
    }
    crate::code_panel(result, columns, theme, 1)
}

fn header_lines(
    text: &str,
    columns: u16,
    padding: u16,
    source: Option<crate::Source>,
) -> Vec<Line> {
    let padding = padding.min(columns.saturating_sub(2));
    let width = columns.saturating_sub(1 + padding).max(1);
    crate::wrap_lines(
        &vec![Span {
            action: None,
            source,
            text: text.into(),
            sgr: "1".into(),
        }],
        width,
        true,
    )
    .into_iter()
    .map(|mut line| {
        let used = line
            .iter()
            .map(|span| crate::width(&span.text))
            .sum::<usize>();
        line.insert(
            0,
            Span {
                action: None,
                source: None,
                text: " ".repeat(padding as usize),
                sgr: String::new(),
            },
        );
        line.push(Span {
            action: None,
            source: None,
            text: " ".repeat((width as usize).saturating_sub(used)),
            sgr: "1".into(),
        });
        line
    })
    .collect()
}

fn hunk(line: &str) -> Option<(u64, u64)> {
    let mut parts = line.strip_prefix("@@ -")?.split_whitespace();
    let old = parts.next()?.split(',').next()?.parse().ok()?;
    let new = parts
        .next()?
        .strip_prefix('+')?
        .split(',')
        .next()?
        .parse()
        .ok()?;
    (parts.next()? == "@@").then_some((old, new))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn hunk_headers_reset_numbers_without_consuming_marker_prefixed_code() {
        let lines = diff_lines(
            "--- f.py\n+++ f.py\n@@ -10,2 +20,2 @@\n---code\n+++code\n@@ -2 +7 @@\n context",
            true,
            32,
            &Theme::default(),
            None,
        );
        let text: Vec<_> = lines
            .iter()
            .map(|line| {
                line.iter()
                    .map(|span| span.text.as_str())
                    .collect::<String>()
                    .trim_end()
                    .to_owned()
            })
            .collect();
        assert_eq!(
            text,
            [
                "┃",
                "┃        --- f.py",
                "┃        +++ f.py",
                "┃   10 - --code",
                "┃   20 + ++code",
                "┃    7   context",
                "┃"
            ]
        );
    }
}
