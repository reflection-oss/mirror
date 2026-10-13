//! Unified diff rendering for write tools.

use similar::TextDiff;

pub(crate) fn unified_diff(before: &str, after: &str, path: &str) -> String {
    TextDiff::from_lines(before, after)
        .unified_diff()
        .header(path, path)
        .to_string()
}

pub(crate) fn diff_display(diff: &str) -> Option<String> {
    if diff.is_empty() {
        return None;
    }
    let lines = diff.lines().collect::<Vec<_>>();
    if lines.len() <= 200 {
        return Some(diff.to_string());
    }
    Some(format!(
        "{}\n… (+{} more diff lines)",
        lines[..200].join("\n"),
        lines.len() - 200
    ))
}
