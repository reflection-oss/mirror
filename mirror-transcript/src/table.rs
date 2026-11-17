//! Logical tables: aligned when they fit, labeled records when the terminal is narrow.

use crate::{fit, line_width, protocol::Theme, Line, Span};

pub fn table_lines(rows: Vec<Vec<Line>>, columns: u16, theme: &Theme) -> Vec<Line> {
    let count = rows.iter().map(Vec::len).max().unwrap_or(0);
    let widths: Vec<usize> = (0..count)
        .map(|index| {
            rows.iter()
                .filter_map(|row| row.get(index))
                .map(|line| line_width(line))
                .max()
                .unwrap_or(0)
        })
        .collect();
    if rows.len() <= 1 {
        return rows.into_iter().flatten().collect();
    }
    if widths.iter().sum::<usize>() + count.saturating_sub(1) * 3 > columns as usize {
        let Some(header) = rows.first() else {
            return vec![];
        };
        return rows
            .iter()
            .skip(1)
            .flat_map(|row| {
                let mut lines = Vec::new();
                for (index, cell) in row.iter().enumerate() {
                    // Repeated record labels must not amplify an arbitrarily large header.
                    let mut line = header.get(index).map_or_else(Vec::new, |label| {
                        let limit = 80;
                        if line_width(label) <= limit {
                            return label.clone();
                        }
                        vec![Span {
                            action: None,
                            source: None,
                            text: fit(
                                &label
                                    .iter()
                                    .map(|span| span.text.as_str())
                                    .collect::<String>(),
                                limit,
                                false,
                            ),
                            sgr: theme.accent.clone(),
                        }]
                    });
                    // Repeated labels describe this record; anchors belong to its value.
                    for span in &mut line {
                        span.source = None;
                    }
                    line.push(Span {
                        action: None,
                        source: None,
                        text: ": ".into(),
                        sgr: theme.muted.clone(),
                    });
                    line.extend(cell.clone());
                    lines.push(line);
                }
                lines.push(vec![]);
                lines
            })
            .collect();
    }
    rows.into_iter()
        .enumerate()
        .flat_map(|(index, row)| {
            let mut line = Vec::new();
            for (column, cell) in row.into_iter().enumerate() {
                let padding = widths[column].saturating_sub(line_width(&cell));
                if column > 0 {
                    line.push(Span {
                        action: None,
                        source: None,
                        text: " │ ".into(),
                        sgr: theme.muted.clone(),
                    });
                }
                line.extend(cell);
                line.push(Span {
                    action: None,
                    source: None,
                    text: " ".repeat(padding),
                    sgr: String::new(),
                });
            }
            if index != 0 {
                return vec![line];
            }
            let rule = widths
                .iter()
                .map(|width| "─".repeat(*width))
                .collect::<Vec<_>>()
                .join("─┼─");
            vec![
                line,
                vec![Span {
                    action: None,
                    source: None,
                    text: rule,
                    sgr: theme.muted.clone(),
                }],
            ]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn narrow_table_keeps_headers_attached_to_values() {
        let rows = vec![
            vec![
                vec![Span {
                    action: None,
                    source: None,
                    text: "Name".into(),
                    sgr: "1".into(),
                }],
                vec![Span {
                    action: None,
                    source: None,
                    text: "Value".into(),
                    sgr: "1".into(),
                }],
            ],
            vec![
                vec![Span {
                    action: None,
                    source: None,
                    text: "alpha".into(),
                    sgr: "".into(),
                }],
                vec![Span {
                    action: None,
                    source: None,
                    text: "👩🏽‍💻".into(),
                    sgr: "32".into(),
                }],
            ],
        ];
        let lines = table_lines(rows, 8, &Theme::default());
        assert_eq!(
            lines[0]
                .iter()
                .map(|span| span.text.as_str())
                .collect::<String>(),
            "Name: alpha"
        );
        assert_eq!(
            lines[1]
                .iter()
                .map(|span| span.text.as_str())
                .collect::<String>(),
            "Value: 👩🏽‍💻"
        );
        assert_eq!(lines[1].last().expect("cell span").sgr, "32");
    }
}
