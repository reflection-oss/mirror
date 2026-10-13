use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;

use futures_util::future::BoxFuture;
use mirror_protocol::agent_message;
use mirror_protocol::chat_record;
use mirror_protocol::AgentMessage;
use mirror_protocol::ChatRecord;
use mirror_protocol::UserMessage;

use super::load_instructions::ContextFiles;
use super::load_instructions::DEFAULT_MAX_BYTES;
use super::CANDIDATE_FILENAMES;
use crate::react::ContextHooks;

const SOURCE_PREFIX: &str = "Instructions from: ";

pub struct NestedInstructions {
    workspace: PathBuf,
    loaded_at_start: Vec<PathBuf>,
}

impl NestedInstructions {
    pub fn new(workspace: &Path) -> Self {
        let workspace = workspace
            .canonicalize()
            .unwrap_or_else(|_| workspace.to_path_buf());
        Self {
            loaded_at_start: super::directories(&workspace),
            workspace,
        }
    }

    fn directories_for(&self, touched: &str) -> Vec<PathBuf> {
        let Some(root) = self.loaded_at_start.first() else {
            return Vec::new();
        };
        let path = self.workspace.join(touched);
        let start = if path.is_dir() {
            path.as_path()
        } else {
            let Some(parent) = path.parent() else {
                return Vec::new();
            };
            parent
        };
        let mut directories = start
            .ancestors()
            .take_while(|directory| directory.starts_with(root))
            .filter(|directory| {
                !self
                    .loaded_at_start
                    .iter()
                    .any(|loaded| loaded == directory)
            })
            .map(Path::to_path_buf)
            .collect::<Vec<_>>();
        directories.reverse();
        directories
    }

    fn instructions(
        &self,
        history: &[AgentMessage],
        replies: &[AgentMessage],
    ) -> Vec<AgentMessage> {
        let Some(root) = self.loaded_at_start.first() else {
            return Vec::new();
        };
        let mut directories = Vec::new();
        for touched in replies
            .iter()
            .filter_map(|reply| reply.metadata.as_ref())
            .flat_map(|metadata| &metadata.touched)
        {
            for directory in self.directories_for(&touched.path) {
                if !directories.contains(&directory) {
                    directories.push(directory);
                }
            }
        }
        let mut loaded = loaded_sources(history);
        let mut files = ContextFiles::rooted(root.clone());
        let mut records = Vec::new();
        for directory in directories {
            for filename in CANDIDATE_FILENAMES {
                let path = directory.join(filename);
                let Ok(Some(bytes)) = files.read(&path) else {
                    continue;
                };
                let Ok(content) = String::from_utf8(bytes) else {
                    continue;
                };
                let source = path.display().to_string();
                let identity = files
                    .identity(&path)
                    .unwrap_or_else(|| path.clone())
                    .display()
                    .to_string();
                let fresh = !loaded.contains(&source) && !loaded.contains(&identity);
                loaded.insert(source.clone());
                loaded.insert(identity);
                if fresh {
                    records.push(instruction_record(
                        &self.workspace,
                        &path,
                        &source,
                        &content,
                    ));
                }
                break;
            }
        }
        records
    }
}

impl ContextHooks for NestedInstructions {
    fn on_tool_replies<'a>(
        &'a self,
        history: &'a [AgentMessage],
        replies: &'a [AgentMessage],
    ) -> BoxFuture<'a, Vec<AgentMessage>> {
        let records = self.instructions(history, replies);
        Box::pin(async move { records })
    }
}

fn loaded_sources(history: &[AgentMessage]) -> HashSet<String> {
    let mut sources = HashSet::new();
    collect_sources(history, &mut sources);
    sources
}

fn collect_sources(entries: &[AgentMessage], sources: &mut HashSet<String>) {
    for entry in entries {
        if let Some(agent_message::Payload::ContextWindow(window)) = &entry.payload {
            if let Some(compaction) = &window.compaction {
                collect_sources(&compaction.recent, sources);
            }
        }
        if let Some(chat_record::Message::User(user)) = entry.chat_message() {
            if let Some(source) = user
                .content
                .strip_prefix(SOURCE_PREFIX)
                .and_then(|rest| rest.lines().next())
            {
                sources.insert(source.to_string());
            }
        }
    }
}

fn instruction_record(workspace: &Path, path: &Path, source: &str, content: &str) -> AgentMessage {
    let rendered = format!("{SOURCE_PREFIX}{source}\n{content}");
    let mut end = rendered.len().min(DEFAULT_MAX_BYTES);
    while !rendered.is_char_boundary(end) {
        end -= 1;
    }
    let shown = path.strip_prefix(workspace).unwrap_or(path);
    AgentMessage {
        payload: Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::User(UserMessage {
                checkpoint: None,
                content: rendered[..end].to_string(),
            })),
            success: true,
            display: Some(format!("Loaded {}", shown.display())),
            ..ChatRecord::default()
        })),
        ..AgentMessage::default()
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use mirror_protocol::Compaction;
    use mirror_protocol::ContextWindowRecord;
    use mirror_protocol::MetadataRecord;
    use mirror_protocol::TouchedKind;
    use mirror_protocol::TouchedPath;
    use pretty_assertions::assert_eq;

    use super::*;

    fn reply_touching(path: &str) -> AgentMessage {
        AgentMessage {
            metadata: Some(MetadataRecord {
                touched: vec![TouchedPath {
                    path: path.to_string(),
                    kind: TouchedKind::Read as i32,
                }],
                ..MetadataRecord::default()
            }),
            ..AgentMessage::default()
        }
    }

    fn contents(records: &[AgentMessage]) -> Vec<String> {
        records
            .iter()
            .filter_map(|record| match record.chat_message() {
                Some(chat_record::Message::User(user)) => Some(user.content.clone()),
                _ => None,
            })
            .collect()
    }

    fn repository() -> (tempfile::TempDir, PathBuf) {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        let root = workspace.path().join("repo");
        fs::create_dir_all(root.join(".git")).expect("create repository marker");
        let root = root.canonicalize().expect("canonical repository");
        (workspace, root)
    }

    #[tokio::test]
    async fn loads_a_nested_directory_once_when_a_tool_first_touches_it() {
        let (_workspace, root) = repository();
        fs::create_dir_all(root.join("service/src")).expect("create service");
        fs::write(root.join("AGENTS.md"), "root").expect("write root");
        fs::write(root.join("service/AGENTS.md"), "service").expect("write service");
        let hooks = NestedInstructions::new(&root);
        let replies = vec![reply_touching("service/src/main.rs")];

        let first = hooks.on_tool_replies(&[], &replies).await;
        let second = hooks.on_tool_replies(&first, &replies).await;

        assert_eq!(
            contents(&first),
            vec![format!(
                "Instructions from: {}\nservice",
                root.join("service/AGENTS.md").display()
            )]
        );
        assert_eq!(contents(&second), Vec::<String>::new());
    }

    #[tokio::test]
    async fn loads_the_instructions_of_a_touched_directory() {
        let (_workspace, root) = repository();
        fs::create_dir_all(root.join("service")).expect("create service");
        fs::write(root.join("service/AGENTS.md"), "service").expect("write service");
        let hooks = NestedInstructions::new(&root);

        let records = hooks
            .on_tool_replies(&[], &[reply_touching("service")])
            .await;

        assert_eq!(
            contents(&records),
            vec![format!(
                "Instructions from: {}\nservice",
                root.join("service/AGENTS.md").display()
            )]
        );
    }

    #[tokio::test]
    async fn does_not_reload_instructions_kept_through_compaction() {
        let (_workspace, root) = repository();
        fs::create_dir_all(root.join("service")).expect("create service");
        fs::write(root.join("service/AGENTS.md"), "service").expect("write service");
        let hooks = NestedInstructions::new(&root);
        let replies = vec![reply_touching("service/lib.rs")];
        let loaded = hooks.on_tool_replies(&[], &replies).await;
        let compacted = AgentMessage {
            payload: Some(agent_message::Payload::ContextWindow(ContextWindowRecord {
                compaction: Some(Compaction {
                    recent: loaded,
                    ..Compaction::default()
                }),
                ..ContextWindowRecord::default()
            })),
            ..AgentMessage::default()
        };

        let records = hooks.on_tool_replies(&[compacted], &replies).await;

        assert_eq!(contents(&records), Vec::<String>::new());
    }

    #[tokio::test]
    async fn skips_directories_loaded_when_the_session_started() {
        let (_workspace, root) = repository();
        let service = root.join("service");
        fs::create_dir_all(&service).expect("create service");
        fs::write(service.join("AGENTS.md"), "service").expect("write service");
        let hooks = NestedInstructions::new(&service);

        let records = hooks
            .on_tool_replies(&[], &[reply_touching("lib.rs")])
            .await;

        assert_eq!(contents(&records), Vec::<String>::new());
    }

    #[tokio::test]
    async fn loads_a_sibling_directory_outside_the_working_directory() {
        let (_workspace, root) = repository();
        fs::create_dir_all(root.join("left")).expect("create left");
        fs::create_dir_all(root.join("right")).expect("create right");
        fs::write(root.join("right/CLAUDE.md"), "right").expect("write right");
        let hooks = NestedInstructions::new(&root.join("left"));
        let touched = root.join("right/notes.txt").display().to_string();

        let records = hooks
            .on_tool_replies(&[], &[reply_touching(&touched)])
            .await;

        assert_eq!(
            contents(&records),
            vec![format!(
                "Instructions from: {}\nright",
                root.join("right/CLAUDE.md").display()
            )]
        );
    }

    #[tokio::test]
    async fn ignores_paths_outside_the_repository() {
        let (workspace, root) = repository();
        fs::write(workspace.path().join("AGENTS.md"), "outside").expect("write outside");
        let hooks = NestedInstructions::new(&root);
        let touched = workspace.path().join("notes.txt").display().to_string();

        let records = hooks
            .on_tool_replies(&[], &[reply_touching(&touched)])
            .await;

        assert_eq!(contents(&records), Vec::<String>::new());
    }
}
