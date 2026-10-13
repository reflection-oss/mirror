//! Agent-skill discovery and metadata.

use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use serde::Deserialize;

/// Metadata for one discoverable agent skill.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub body: String,
    pub path: PathBuf,
    pub disable_model_invocation: bool,
}

impl Skill {
    /// The contextual user fragment injected for an explicit invocation.
    pub fn invocation_content(&self) -> String {
        format!(
            "<skill>\n<name>{}</name>\n<path>{}</path>\n{}\n</skill>",
            xml_escape(&self.name),
            xml_escape(&self.path.to_string_lossy()),
            self.body
        )
    }
}

/// Return exact whitespace-delimited skill mentions without repeats.
pub fn mentioned_skills<'a>(text: &str, available: &'a [Skill]) -> Vec<&'a Skill> {
    let mut seen = HashSet::new();
    text.split_whitespace()
        .filter_map(|token| token.strip_prefix('$'))
        .filter_map(|name| available.iter().find(|skill| skill.name == name))
        .filter(|skill| seen.insert(skill.name.as_str()))
        .collect()
}

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

const MAX_SCAN_DEPTH: usize = 6;
const MAX_SCAN_DIRS: usize = 2000;
pub const SKILL_ROOT_DIRS: &[&str] = &[".agents/skills", ".claude/skills"];

#[derive(Debug, Default, Deserialize)]
struct Frontmatter {
    name: Option<String>,
    description: Option<String>,
    #[serde(rename = "disable-model-invocation")]
    disable_model_invocation: Option<bool>,
}

/// Discover skills from global and project-local skill roots.
pub fn load_skills(cwd: &Path, global_dir: Option<&Path>) -> Vec<Skill> {
    let mut roots = Vec::new();
    if let Some(global) = global_dir
        .map(Path::to_path_buf)
        .or_else(default_global_dir)
    {
        roots.push(global);
    }
    for directory in super::directories(cwd) {
        roots.extend(SKILL_ROOT_DIRS.iter().map(|root| directory.join(root)));
    }
    read_skills(
        &roots,
        &mut super::load_instructions::ContextFiles::default(),
    )
    .unwrap_or_default()
}

pub(super) fn read_skills(
    roots: &[PathBuf],
    files: &mut super::load_instructions::ContextFiles,
) -> std::io::Result<Vec<Skill>> {
    let mut loaded = Vec::<Skill>::new();
    let mut seen = HashSet::new();
    for root in roots {
        let Some(canonical_root) = files.identity(root) else {
            continue;
        };
        let mut queue = vec![(root.clone(), 0)];
        let mut visited = HashSet::new();
        let mut index = 0;
        while index < queue.len() && visited.len() < MAX_SCAN_DIRS {
            let (current, depth) = queue[index].clone();
            index += 1;
            let Some(canonical) = files.identity(&current) else {
                continue;
            };
            if !canonical.starts_with(&canonical_root) || !visited.insert(canonical.clone()) {
                continue;
            }
            let path = current.join("SKILL.md");
            let raw = files.read(&path)?;
            if raw.is_some() || path.is_file() {
                if let Some(skill) = raw
                    .as_deref()
                    .and_then(|raw| std::str::from_utf8(raw).ok())
                    .and_then(|raw| parse_skill(&path, raw))
                {
                    if seen.insert(files.identity(&path).unwrap_or_else(|| path.clone())) {
                        if let Some(index) = loaded
                            .iter()
                            .position(|existing| existing.name == skill.name)
                        {
                            loaded[index] = skill;
                        } else {
                            loaded.push(skill);
                        }
                    }
                }
                continue;
            }
            if depth >= MAX_SCAN_DEPTH {
                continue;
            }
            let mut directories = fs::read_dir(&current)
                .into_iter()
                .flatten()
                .filter_map(Result::ok)
                .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
                .map(|entry| entry.path())
                .collect::<Vec<_>>();
            directories.sort();
            for directory in directories {
                let name = directory.file_name().unwrap_or_default();
                if !name.to_string_lossy().starts_with('.') && name != "node_modules" {
                    queue.push((directory, depth + 1));
                }
            }
        }
    }
    Ok(loaded)
}

fn default_global_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".mirror/skills"))
}

fn parse_skill(path: &Path, raw: &str) -> Option<Skill> {
    let (frontmatter, body) = split_frontmatter(raw);
    let description = frontmatter
        .description?
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if description.is_empty() {
        return None;
    }
    let name = frontmatter
        .name
        .filter(|name| !name.trim().is_empty())
        .unwrap_or_else(|| {
            path.parent()
                .and_then(Path::file_name)
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned()
        });
    if !valid_name(&name) {
        return None;
    }
    Some(Skill {
        name,
        description,
        body,
        path: path.to_path_buf(),
        disable_model_invocation: frontmatter.disable_model_invocation.unwrap_or(false),
    })
}

fn split_frontmatter(raw: &str) -> (Frontmatter, String) {
    let normalized = raw
        .trim_start_matches('\u{feff}')
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    let mut lines = normalized.lines();
    if lines.next().map(str::trim) != Some("---") {
        return (Frontmatter::default(), normalized.trim().to_string());
    }
    let remaining = lines.collect::<Vec<_>>();
    let Some(end) = remaining.iter().position(|line| line.trim() == "---") else {
        return (Frontmatter::default(), normalized.trim().to_string());
    };
    let block = remaining[..end].join("\n");
    let body = remaining[end + 1..].join("\n").trim().to_string();
    let frontmatter = serde_yaml::from_str(&block)
        .or_else(|_| serde_yaml::from_str(&repair_unquoted_colons(&block)))
        .unwrap_or_default();
    (frontmatter, body)
}

fn repair_unquoted_colons(block: &str) -> String {
    block
        .lines()
        .map(|line| {
            let Some((key, value)) = line.split_once(':') else {
                return line.to_string();
            };
            let value = value.trim();
            if value.contains(": ")
                && !matches!(
                    value.chars().next(),
                    Some('"' | '\'' | '|' | '>' | '[' | '{')
                )
            {
                format!("{key}: '{}'", value.replace('\'', "''"))
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .next()
            .is_some_and(|character| character.is_ascii_alphanumeric())
        && name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use assert_matches::assert_matches;
    use pretty_assertions::{assert_eq, assert_ne};

    use super::*;

    #[cfg(unix)]
    #[test]
    fn invalid_alias_does_not_hide_a_later_valid_skill() {
        let workspace = tempfile::tempdir().expect("workspace");
        let root = workspace.path().join(".agents/skills");
        fs::create_dir_all(root.join("a.invalid")).expect("invalid name");
        fs::create_dir_all(root.join("review")).expect("valid name");
        fs::write(
            root.join("a.invalid/SKILL.md"),
            "---\ndescription: Review code.\n---\nReview body.",
        )
        .expect("skill");
        std::os::unix::fs::symlink("../a.invalid/SKILL.md", root.join("review/SKILL.md"))
            .expect("alias");
        let skills = load_skills(workspace.path(), Some(&workspace.path().join("no-global")));
        assert_matches!(skills.as_slice(), [Skill { name, body, .. }]
            if name == "review" && body == "Review body.");
    }

    #[test]
    fn discovers_project_skill_and_normalizes_description() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        fs::create_dir(workspace.path().join(".git")).expect("create repository marker");
        let skill_dir = workspace.path().join(".agents/skills/pdf");
        fs::create_dir_all(&skill_dir).expect("create skill directory");
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: pdf\ndescription: Work with\n  PDFs.\n---\nbody",
        )
        .expect("write skill");

        let skills = load_skills(workspace.path(), Some(&workspace.path().join("none")));

        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "pdf");
        assert_eq!(skills[0].description, "Work with PDFs.");
        assert_eq!(skills[0].body, "body");
        assert_ne!(skills[0].name, "review");
        assert_matches!(skills.as_slice(), [Skill { name, .. }] if name == "pdf");
    }

    #[test]
    fn project_skill_overrides_global_skill_by_name() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        let global = workspace.path().join("global");
        let local = workspace.path().join(".agents/skills/review");
        fs::create_dir_all(global.join("review")).expect("create global skill");
        fs::create_dir_all(&local).expect("create local skill");
        fs::write(
            global.join("review/SKILL.md"),
            "---\nname: review\ndescription: Global.\n---\nglobal",
        )
        .expect("write global skill");
        fs::write(
            local.join("SKILL.md"),
            "---\nname: review\ndescription: Local.\n---\nlocal",
        )
        .expect("write local skill");

        let skills = load_skills(workspace.path(), Some(&global));

        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].description, "Local.");
        assert_ne!(skills[0].body, "global");
        assert_matches!(skills.as_slice(), [Skill { name, .. }] if name == "review");
    }

    #[test]
    fn discovers_root_nested_and_claude_skills_in_jujutsu_workspace() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        let root = workspace.path().join("repo");
        let child = root.join("packages/app");
        fs::create_dir_all(root.join(".jj")).expect("create repository marker");
        fs::create_dir_all(&child).expect("create child");
        write_skill(&root.join(".agents/skills/root"), "root", "Root skill.");
        write_skill(&child.join(".agents/skills/deep"), "deep", "Deep skill.");
        write_skill(
            &root.join(".claude/skills/legacy"),
            "legacy",
            "Claude skill.",
        );

        let skills = load_skills(&child, Some(&workspace.path().join("no-global")));
        let names = skills
            .iter()
            .map(|skill| skill.name.as_str())
            .collect::<Vec<_>>();

        assert_eq!(names, ["root", "legacy", "deep"]);
        assert_ne!(names, ["deep"]);
        assert_matches!(skills.as_slice(), [.., Skill { name, .. }] if name == "deep");
    }

    #[test]
    fn nearest_project_skill_overrides_root_and_global_versions() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        let global = workspace.path().join("global");
        let root = workspace.path().join("repo");
        let child = root.join("package");
        fs::create_dir_all(root.join(".git")).expect("create repository marker");
        fs::create_dir_all(&child).expect("create child");
        write_skill(&global.join("review"), "review", "Global review.");
        write_skill(
            &root.join(".agents/skills/review"),
            "review",
            "Root review.",
        );
        write_skill(
            &child.join(".agents/skills/review"),
            "review",
            "Nearest review.",
        );

        let skills = load_skills(&child, Some(&global));

        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].description, "Nearest review.");
        assert_ne!(skills[0].description, "Root review.");
        assert_matches!(skills.as_slice(), [Skill { name, .. }] if name == "review");
    }

    #[test]
    fn recursively_discovers_skills_but_does_not_descend_into_a_skill() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        fs::create_dir(workspace.path().join(".git")).expect("create repository marker");
        let collection = workspace.path().join(".agents/skills/collection");
        write_skill(
            &collection.join("nested"),
            "nested",
            "Nested collection skill.",
        );
        let outer = workspace.path().join(".agents/skills/outer");
        write_skill(&outer, "outer", "Outer skill.");
        write_skill(
            &outer.join("references/hidden"),
            "hidden",
            "Hidden nested skill.",
        );

        let skills = load_skills(workspace.path(), Some(&workspace.path().join("no-global")));
        let names = skills
            .iter()
            .map(|skill| skill.name.as_str())
            .collect::<Vec<_>>();

        assert_eq!(names, ["outer", "nested"]);
        assert_ne!(names, ["outer", "nested", "hidden"]);
        assert_matches!(skills.as_slice(), [Skill { name: outer, .. }, Skill { name: nested, .. }] if outer == "outer" && nested == "nested");
    }

    #[test]
    fn skips_invalid_missing_and_binary_skill_metadata() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        fs::create_dir(workspace.path().join(".git")).expect("create repository marker");
        write_raw_skill(
            &workspace.path().join(".agents/skills/missing"),
            "---\nname: missing\n---\nbody",
        );
        write_raw_skill(
            &workspace.path().join(".agents/skills/invalid"),
            "---\nname: not invokable\ndescription: Invalid.\n---\nbody",
        );
        let binary = workspace.path().join(".agents/skills/binary");
        fs::create_dir_all(&binary).expect("create binary skill");
        fs::write(binary.join("SKILL.md"), b"\xff\xfe\x00").expect("write binary skill");
        write_skill(
            &workspace.path().join(".agents/skills/valid"),
            "valid",
            "Valid skill.",
        );

        let skills = load_skills(workspace.path(), Some(&workspace.path().join("no-global")));

        assert_eq!(skills.len(), 1);
        assert_ne!(skills[0].name, "invalid");
        assert_matches!(skills.as_slice(), [Skill { name, .. }] if name == "valid");
    }

    #[cfg(unix)]
    #[test]
    fn deduplicates_linked_roots_and_rejects_child_links_outside_them() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        fs::create_dir(workspace.path().join(".git")).expect("create repository marker");
        let agents = workspace.path().join(".agents/skills");
        write_skill(&agents.join("pdf"), "pdf", "PDF skill.");
        fs::create_dir_all(workspace.path().join(".claude")).expect("create claude directory");
        std::os::unix::fs::symlink("../.agents/skills", workspace.path().join(".claude/skills"))
            .expect("link skill roots");
        let external = workspace.path().join("external/leaked");
        write_skill(&external, "leaked", "External skill.");
        std::os::unix::fs::symlink(&external, agents.join("linked")).expect("link external skill");

        let skills = load_skills(workspace.path(), Some(&workspace.path().join("no-global")));

        assert_eq!(skills.len(), 1);
        assert_ne!(skills[0].name, "leaked");
        assert_matches!(skills.as_slice(), [Skill { name, .. }] if name == "pdf");
    }

    fn write_skill(directory: &Path, name: &str, description: &str) {
        write_raw_skill(
            directory,
            &format!("---\nname: {name}\ndescription: {description}\n---\nbody"),
        );
    }

    fn write_raw_skill(directory: &Path, content: &str) {
        fs::create_dir_all(directory).expect("create skill directory");
        fs::write(directory.join("SKILL.md"), content).expect("write skill");
    }
}
