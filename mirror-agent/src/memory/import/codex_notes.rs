//! Codex memories: `memory_summary.md` (user profile, preferences, general
//! tips) and `MEMORY.md` (per task group: preferences, reusable knowledge,
//! failures). Bullets are the unit; task tags such as `[Task 2]` name the
//! situation. Raw per-thread memories and rollout summaries are episodic and
//! are not imported.

use std::collections::BTreeMap;
use std::io::ErrorKind;
use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;

use super::claude_code_notes::modified_at;
use super::ImportedNote;
use crate::memory::invalid;
use crate::memory::slug_key;
use crate::memory::MemoryResult;
use crate::memory::MemoryType;

/// A `##` heading and the lines under it, up to the next `#` heading of any level.
struct Section {
    heading: String,
    lines: Vec<String>,
}

/// Notes from both files, in file order.
pub fn codex_notes(directory: &Path) -> MemoryResult<Vec<ImportedNote>> {
    let mut notes = Vec::new();
    let mut read_any = false;
    for (name, parse) in [
        (
            "memory_summary.md",
            summary_notes as fn(&str, &str, Option<u64>) -> Vec<ImportedNote>,
        ),
        ("MEMORY.md", memory_notes),
    ] {
        let path = directory.join(name);
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                read_any = true;
                notes.extend(parse(&text, name, modified_at(&path)));
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    if !read_any {
        return Err(invalid(format!(
            "no Codex memory files found in {}",
            directory.display()
        )));
    }
    Ok(notes)
}

fn summary_notes(text: &str, file: &str, observed_at: Option<u64>) -> Vec<ImportedNote> {
    let mut notes = Vec::new();
    for section in sections(text) {
        match section.heading.as_str() {
            "User Profile" => {
                let profile = section.paragraphs().join(" ");
                if !profile.is_empty() {
                    notes.push(note(
                        "codex-user-profile",
                        MemoryType::Fact,
                        "who the user is, as Codex summarized it",
                        &profile,
                        file,
                        observed_at,
                    ));
                }
            }
            "User preferences" => {
                for bullet in section.bullets() {
                    notes.push(note(
                        &format!("codex-pref-{}", slug_key(&bullet)),
                        MemoryType::Preference,
                        "how the user wants work done, as Codex summarized it",
                        &bullet,
                        file,
                        observed_at,
                    ));
                }
            }
            "General Tips" => {
                for bullet in section.bullets() {
                    notes.push(note(
                        &format!("codex-tip-{}", slug_key(&bullet)),
                        MemoryType::Procedure,
                        "working practice Codex learned across tasks",
                        &bullet,
                        file,
                        observed_at,
                    ));
                }
            }
            _ => {}
        }
    }
    notes
}

fn memory_notes(text: &str, file: &str, observed_at: Option<u64>) -> Vec<ImportedNote> {
    static TASK: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"^Task (\d+): (.+)$").expect("task pattern"));
    static TAG: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\s*\[Task (\d+)\]\s*$").expect("tag pattern"));
    let mut tasks: BTreeMap<String, String> = BTreeMap::new();
    let mut notes = Vec::new();
    for section in sections(text) {
        if let Some(capture) = TASK.captures(&section.heading) {
            tasks.insert(capture[1].to_string(), capture[2].trim().to_string());
            continue;
        }
        let kind = match section.heading.as_str() {
            "User preferences" => MemoryType::Preference,
            "Reusable knowledge" => MemoryType::Fact,
            "Failures and how to do differently" => MemoryType::Pitfall,
            _ => continue,
        };
        for bullet in section.bullets() {
            let mut situation = String::from("from the user's Codex task memory");
            let mut content = bullet.clone();
            while let Some(capture) = TAG.captures(&content) {
                if let Some(title) = tasks.get(&capture[1]) {
                    situation = format!("Codex task: {title}");
                }
                let end = capture
                    .get(0)
                    .map(|found| found.start())
                    .unwrap_or(content.len());
                content.truncate(end);
            }
            if content.trim().is_empty() {
                continue;
            }
            let prefix = match kind {
                MemoryType::Preference => "codex-pref",
                MemoryType::Fact => "codex-fact",
                _ => "codex-pitfall",
            };
            notes.push(note(
                &format!("{prefix}-{}", slug_key(&content)),
                kind,
                &situation,
                content.trim(),
                file,
                observed_at,
            ));
        }
    }
    notes
}

fn note(
    key: &str,
    kind: MemoryType,
    situation: &str,
    content: &str,
    file: &str,
    observed_at: Option<u64>,
) -> ImportedNote {
    ImportedNote {
        key: slug_key(key),
        kind,
        situation: situation.to_string(),
        content: content.to_string(),
        keywords: Vec::new(),
        observed_at,
        file: file.to_string(),
    }
}

impl Section {
    /// Bullets, with indented continuation text joined by a space. An
    /// indented list marker starts its own note.
    fn bullets(&self) -> Vec<String> {
        let mut bullets: Vec<String> = Vec::new();
        for line in &self.lines {
            let trimmed = line.trim_start_matches([' ', '\t']);
            if let Some(item) = trimmed.strip_prefix("- ") {
                bullets.push(item.trim().to_string());
            } else if line.starts_with(' ') || line.starts_with('\t') {
                if let Some(last) = bullets.last_mut() {
                    last.push(' ');
                    last.push_str(line.trim());
                }
            }
        }
        bullets
    }

    /// Non-bullet, non-empty lines.
    fn paragraphs(&self) -> Vec<String> {
        self.lines
            .iter()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty() && !line.starts_with("- "))
            .map(str::to_string)
            .collect()
    }
}

fn sections(text: &str) -> Vec<Section> {
    let mut sections: Vec<Section> = Vec::new();
    for line in text.lines() {
        if let Some(heading) = line.strip_prefix("## ") {
            sections.push(Section {
                heading: heading.trim().to_string(),
                lines: Vec::new(),
            });
        } else if line.starts_with('#') {
            // A deeper or shallower heading ends the current section's bullets.
            sections.push(Section {
                heading: String::new(),
                lines: Vec::new(),
            });
        } else if let Some(section) = sections.last_mut() {
            section.lines.push(line.to_string());
        }
    }
    sections.retain(|section| !section.heading.is_empty());
    sections
}
