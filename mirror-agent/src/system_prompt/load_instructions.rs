use std::collections::HashSet;
use std::fs;
use std::io::{self, Read};
use std::path::Path;
use std::path::PathBuf;

use super::{ResolvedInstructions, SystemPromptConfig};

pub fn resolve_prompt_instructions(config: &SystemPromptConfig) -> ResolvedInstructions {
    if let Some(resolved) = &config.resolved {
        return resolved.clone();
    }
    config
        .working_directory
        .as_deref()
        .map(|directory| ResolvedInstructions {
            instructions: if config.include_project_instructions {
                load_instructions(directory, config.global_instructions_dir.as_deref())
            } else {
                Vec::new()
            },
            skills: if config.include_skills {
                let local = super::load_skills(directory, config.global_skills_dir.as_deref());
                let mut skills = config.additional_skills.clone();
                skills.retain(|skill| !local.iter().any(|local| local.name == skill.name));
                skills.extend(local);
                skills
            } else {
                Vec::new()
            },
        })
        .unwrap_or_default()
}

pub const PINNED_MEMORIES_FILENAME: &str = "PINNED_MEMORIES.md";
pub const MAX_PINNED_MEMORIES_BYTES: usize = 4_096;

pub fn pinned_memories_path() -> io::Result<PathBuf> {
    default_global_dir()
        .map(|directory| directory.join(PINNED_MEMORIES_FILENAME))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "memory needs a home directory to locate PINNED_MEMORIES.md",
            )
        })
}

pub(super) fn load_pinned_memories(path: &Path) -> Option<(String, String)> {
    let content = fs::read_to_string(path).ok()?;
    if content.trim().is_empty() {
        return None;
    }
    let mut end = content.len().min(MAX_PINNED_MEMORIES_BYTES);
    while !content.is_char_boundary(end) {
        end -= 1;
    }
    Some((
        path.display().to_string(),
        format!(
            "Pinned memories from: {}\n{}",
            path.display(),
            &content[..end]
        ),
    ))
}

pub fn load_workspace_context(cwd: &Path, boundary: &Path) -> io::Result<ResolvedInstructions> {
    load_repository_context(cwd, boundary, true)
}

pub fn load_workspace_skills(cwd: &Path, boundary: &Path) -> io::Result<ResolvedInstructions> {
    load_repository_context(cwd, boundary, false)
}

fn load_repository_context(
    cwd: &Path,
    boundary: &Path,
    include_instructions: bool,
) -> io::Result<ResolvedInstructions> {
    let root = fs::canonicalize(boundary)?;
    let cwd = fs::canonicalize(cwd)?;
    if !cwd.starts_with(&root) {
        return Err(io::Error::other("workspace escapes repository"));
    }
    let mut directories = cwd
        .ancestors()
        .take_while(|path| path.starts_with(&root))
        .map(Path::to_path_buf)
        .collect::<Vec<_>>();
    directories.reverse();
    let roots = directories
        .iter()
        .flat_map(|directory| {
            super::SKILL_ROOT_DIRS
                .iter()
                .map(move |name| directory.join(name))
        })
        .collect::<Vec<_>>();
    let mut files = ContextFiles {
        root: Some(root),
        ..Default::default()
    };
    Ok(ResolvedInstructions {
        instructions: if include_instructions {
            read_instructions(&directories, &mut files)?
        } else {
            Vec::new()
        },
        skills: super::skill::read_skills(&roots, &mut files)?,
    })
}

#[derive(Default)]
pub(super) struct ContextFiles {
    root: Option<PathBuf>,
    count: usize,
    bytes: usize,
}

impl ContextFiles {
    pub(super) fn rooted(root: PathBuf) -> Self {
        Self {
            root: Some(root),
            ..Self::default()
        }
    }

    pub(super) fn identity(&self, path: &Path) -> Option<PathBuf> {
        path.canonicalize()
            .ok()
            .filter(|path| self.root.as_ref().is_none_or(|root| path.starts_with(root)))
    }

    pub(super) fn read(&mut self, path: &Path) -> io::Result<Option<Vec<u8>>> {
        if self.root.is_none() {
            return Ok(path.is_file().then(|| fs::read(path).ok()).flatten());
        }
        let resolved = match fs::canonicalize(path) {
            Ok(path) => path,
            Err(error)
                if error.kind() == io::ErrorKind::NotFound
                    && fs::symlink_metadata(path).is_err() =>
            {
                return Ok(None)
            }
            Err(error) => return Err(error),
        };
        if !self
            .root
            .as_ref()
            .is_some_and(|root| resolved.starts_with(root))
        {
            return Err(io::Error::other("instruction source escapes repository"));
        }
        if !fs::metadata(&resolved)?.is_file() {
            return Err(io::Error::other("instruction source is not a regular file"));
        }
        let mut bytes = Vec::new();
        fs::File::open(resolved)?
            .take(256 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        self.count += 1;
        self.bytes += bytes.len();
        if bytes.len() > 256 * 1024 || self.count > 1024 || self.bytes > 4 * 1024 * 1024 {
            return Err(io::Error::other(
                "workspace context exceeds file or byte limit",
            ));
        }
        Ok(Some(bytes))
    }
}

pub fn load_instructions(cwd: &Path, global_dir: Option<&Path>) -> Vec<(String, String)> {
    let mut directories = super::directories(cwd);
    if let Some(global_dir) = global_dir
        .map(Path::to_path_buf)
        .or_else(default_global_dir)
    {
        directories.insert(0, global_dir);
    }
    read_instructions(&directories, &mut ContextFiles::default()).unwrap_or_default()
}

fn read_instructions(
    directories: &[PathBuf],
    files: &mut ContextFiles,
) -> io::Result<Vec<(String, String)>> {
    let mut blocks = Vec::new();
    let mut seen = HashSet::<PathBuf>::new();
    let mut remaining = DEFAULT_MAX_BYTES;
    let mut sources = Vec::new();
    for directory in directories {
        for filename in CANDIDATE_FILENAMES {
            let path = directory.join(filename);
            if let Some(bytes) = files.read(&path)? {
                match String::from_utf8(bytes) {
                    Ok(content) => {
                        sources.push((path, content));
                        break;
                    }
                    Err(error) if files.root.is_some() => {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, error))
                    }
                    Err(_) => continue,
                }
            }
        }
    }
    for (path, content) in sources {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
        if remaining == 0 || !seen.insert(canonical) {
            continue;
        }
        let rendered = format!("Instructions from: {}\n{content}", path.display());
        let mut end = rendered.len().min(remaining);
        while !rendered.is_char_boundary(end) {
            end -= 1;
        }
        blocks.push((path.display().to_string(), rendered[..end].to_string()));
        remaining -= end;
    }
    Ok(blocks)
}

pub const CANDIDATE_FILENAMES: &[&str] = &["AGENTS.md", "CLAUDE.md"];
pub(super) const DEFAULT_MAX_BYTES: usize = 64 * 1024;

fn default_global_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".mirror"))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn loads_global_then_root_to_leaf_instructions() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        let root = workspace.path().join("repo");
        let child = root.join("child");
        let global = workspace.path().join("global");
        fs::create_dir_all(root.join(".git")).expect("create repository marker");
        fs::create_dir_all(&child).expect("create child");
        fs::create_dir_all(&global).expect("create global");
        fs::write(global.join("AGENTS.md"), "global").expect("write global");
        fs::write(root.join("AGENTS.md"), "root").expect("write root");
        fs::write(child.join("CLAUDE.md"), "child").expect("write child");

        let loaded = load_instructions(&child, Some(&global));

        assert_eq!(loaded.len(), 3);
        for ((source, text), (path, content)) in loaded.iter().zip([
            (global.join("AGENTS.md"), "global"),
            (root.join("AGENTS.md"), "root"),
            (child.join("CLAUDE.md"), "child"),
        ]) {
            assert_eq!(source, &path.display().to_string());
            assert_eq!(
                text,
                &format!("Instructions from: {}\n{content}", path.display())
            );
        }
    }

    #[test]
    fn prefers_agents_and_stops_at_the_nearest_repository_root() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        let outer = workspace.path().join("outer");
        let root = outer.join("repo");
        let child = root.join("src/package");
        fs::create_dir_all(root.join(".git")).expect("create repository marker");
        fs::create_dir_all(&child).expect("create child");
        fs::write(outer.join("AGENTS.md"), "outside").expect("write outer instructions");
        fs::write(root.join("AGENTS.md"), "agents").expect("write agents instructions");
        fs::write(root.join("CLAUDE.md"), "claude").expect("write claude instructions");
        fs::write(child.join("CLAUDE.md"), "child").expect("write child instructions");

        let loaded = load_instructions(&child, Some(&workspace.path().join("no-global")));

        assert_eq!(loaded.len(), 2);
        assert!(loaded
            .iter()
            .all(|(_, text)| !text.contains("outside") && !text.contains("claude")));
        assert_eq!(loaded[0].0, root.join("AGENTS.md").display().to_string());
        assert_eq!(loaded[1].0, child.join("CLAUDE.md").display().to_string());
        assert!(loaded[0].1.ends_with("\nagents"));
        assert!(loaded[1].1.ends_with("\nchild"));
    }

    #[test]
    fn non_repository_loads_only_the_working_directory() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        let parent = workspace.path().join("parent");
        let child = parent.join("child");
        fs::create_dir_all(&child).expect("create child");
        fs::write(parent.join("AGENTS.md"), "parent").expect("write parent instructions");
        fs::write(child.join("AGENTS.md"), "child").expect("write child instructions");

        let loaded = load_instructions(&child, Some(&workspace.path().join("no-global")));

        assert_eq!(
            loaded,
            vec![(
                child.join("AGENTS.md").display().to_string(),
                format!(
                    "Instructions from: {}\nchild",
                    child.join("AGENTS.md").display()
                ),
            )]
        );
    }

    #[test]
    fn nested_instructions_load_after_a_root_over_32_kib() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        let root = workspace.path().join("repo");
        let child = root.join("child");
        fs::create_dir_all(root.join(".git")).expect("create repository marker");
        fs::create_dir_all(&child).expect("create child");
        fs::write(root.join("AGENTS.md"), "r".repeat(40 * 1024)).expect("write root");
        fs::write(child.join("AGENTS.md"), "child").expect("write child");

        let loaded = load_instructions(&child, Some(&workspace.path().join("no-global")));

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[1].0, child.join("AGENTS.md").display().to_string());
        assert!(loaded[1].1.ends_with("\nchild"));
    }

    #[test]
    fn missing_instruction_files_return_empty() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        fs::create_dir(workspace.path().join(".git")).expect("create repository marker");

        let loaded = load_instructions(workspace.path(), Some(&workspace.path().join("no-global")));

        assert_matches!(loaded.as_slice(), []);
    }

    #[cfg(unix)]
    #[test]
    fn canonical_paths_deduplicate_global_and_project_instructions() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        let root = workspace.path().join("repo");
        fs::create_dir_all(root.join(".git")).expect("create repository marker");
        fs::write(root.join("AGENTS.md"), "shared").expect("write instructions");
        let global = workspace.path().join("global");
        std::os::unix::fs::symlink(&root, &global).expect("link global directory");

        let loaded = load_instructions(&root, Some(&global));

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, global.join("AGENTS.md").display().to_string());
        assert_eq!(loaded[0].1.matches("shared").count(), 1);
        assert_eq!(loaded[0].1.matches("Instructions from:").count(), 1);
    }
}
