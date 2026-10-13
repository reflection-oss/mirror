//! `mirror-agent memory`: manage the personal memory store on this machine.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;
use std::path::PathBuf;

use clap::Parser;
use clap::Subcommand;
use mirror_agent::memory::import::ImportReport;
use mirror_agent::memory::import::ImportSource;
use mirror_agent::memory::import::MemoryImport;
use mirror_agent::memory::SqliteMemoryDatabase;
use mirror_agent::RepositoryIdentity;

#[derive(Parser)]
#[command(
    name = "mirror-agent memory",
    about = "Manage Mirror's personal memory on this machine"
)]
struct MemoryCommand {
    /// The memory store directory; defaults to ~/.mirror/memory.
    #[arg(long, global = true)]
    store: Option<PathBuf>,
    #[command(subcommand)]
    action: MemoryAction,
}

#[derive(Subcommand)]
enum MemoryAction {
    /// Import memories saved by Claude Code or Codex under the user's own name.
    Import {
        /// Scope every imported record to this git checkout instead of all
        /// repositories.
        #[arg(long)]
        repository: Option<PathBuf>,
        /// `kind=path` (kind: claude-code, codex) or a path whose layout is
        /// detected. With none, the harnesses' default locations on this
        /// machine are imported.
        sources: Vec<String>,
    },
}

/// Run `mirror-agent memory …` from this process's arguments.
pub fn run() -> Result<(), Box<dyn Error>> {
    let command = MemoryCommand::parse_from(
        std::iter::once("mirror-agent memory".to_string()).chain(std::env::args().skip(2)),
    );
    let store = match command.store {
        Some(store) => store,
        None => std::env::var_os("HOME")
            .map(|home| PathBuf::from(home).join(".mirror").join("memory"))
            .ok_or("HOME is not set; pass --store")?,
    };
    match command.action {
        MemoryAction::Import {
            repository,
            sources,
        } => import(&store, repository.as_deref(), &sources),
    }
}

fn import(store: &Path, repository: Option<&Path>, specs: &[String]) -> Result<(), Box<dyn Error>> {
    if let Some(repository) = repository {
        if RepositoryIdentity::discover(repository).is_none() {
            return Err(format!("{} is not a git checkout", repository.display()).into());
        }
    }
    let sources = if specs.is_empty() {
        ImportSource::discover(repository)
    } else {
        specs
            .iter()
            .map(|spec| ImportSource::parse(spec, repository))
            .collect::<Result<Vec<_>, _>>()?
    };
    if sources.is_empty() {
        println!("no memories found to import · pass a path or kind=path");
        return Ok(());
    }
    let mut database = SqliteMemoryDatabase::open(store)?;
    let mut providers = BTreeMap::<_, ImportReport>::new();
    for source in &sources {
        let provider = match source {
            ImportSource::ClaudeCode { .. } => "Claude Code",
            ImportSource::Codex { .. } => "Codex",
            _ => return Err("unsupported memory import provider".into()),
        };
        let report = MemoryImport::new(&mut database).run(std::slice::from_ref(source))?;
        let combined = providers.entry(provider).or_default();
        combined.saved += report.saved;
        combined.retired += report.retired;
        combined.unchanged += report.unchanged;
        combined.rejected.extend(report.rejected);
    }
    let mut total = ImportReport::default();
    let mut details = Vec::new();
    for (provider, report) in providers {
        details.push(format!("  {provider}: {}", summary(&report)));
        total.saved += report.saved;
        total.retired += report.retired;
        total.unchanged += report.unchanged;
        total.rejected.extend(report.rejected);
    }
    println!("{}", summary(&total));
    for detail in details {
        println!("{detail}");
    }
    // Keys and reasons come from files on disk, so terminal controls are escaped.
    for (key, reason) in &total.rejected {
        if reason == "another memory already has this key" {
            continue;
        }
        println!(
            "  skipped {}: {}",
            key.escape_debug(),
            reason.escape_debug()
        );
    }
    Ok(())
}

fn summary(report: &ImportReport) -> String {
    let mut headline = vec![match report.saved {
        0 => "nothing new to import".to_string(),
        saved => format!("imported {}", memories(saved)),
    }];
    for (count, label) in [
        (report.retired, "retired"),
        (report.unchanged, "already there"),
        (report.rejected.len(), "skipped"),
    ] {
        if count > 0 {
            headline.push(format!("{count} {label}"));
        }
    }
    headline.join(" · ")
}

fn memories(count: usize) -> String {
    if count == 1 {
        "1 memory".to_string()
    } else {
        format!("{count} memories")
    }
}
