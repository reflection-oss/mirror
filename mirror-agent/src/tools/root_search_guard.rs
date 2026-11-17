use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::slice::Iter;
use std::sync::Arc;

use serde_json::Value;

use super::shell_syntax::{self, Word};
use super::{AnyTool, RemoteToolExecutor, ToolDefinition, ToolOutput};
use crate::ExecutionContext;

const SEARCH_PROGRAMS: &[&str] = &["grep", "egrep", "fgrep", "rg"];
const PREFIX_PROGRAMS: &[&str] = &["sudo", "command", "exec", "nice", "nohup", "env", "time"];
const ROOT_OPERANDS: &[&str] = &["/", "/.", "//", "/*"];
const CURRENT_DIRECTORY_OPERANDS: &[&str] = &[".", "./", "*"];
const FIND_LISTING_FLAGS: &[&str] = &["-print", "-print0", "-follow", "-depth", "-noleaf"];
const FIND_LISTING_VALUE_FLAGS: &[&str] = &["-type", "-maxdepth", "-mindepth"];
const GREP_VALUE_FLAGS: &str = "efmABCdD";
const XARGS_VALUE_FLAGS: &str = "adEILnPs";
const XARGS_LONG_VALUE_FLAGS: &[&str] = &[
    "--arg-file",
    "--delimiter",
    "--max-args",
    "--max-chars",
    "--max-procs",
    "--process-slot-var",
];
const RG_VALUE_FLAGS: &str = "efgrtTmABCEjM";

pub struct RootSearchGuard {
    tool: Arc<dyn AnyTool>,
    working_directory: Option<PathBuf>,
}

impl RootSearchGuard {
    pub fn new(tool: Arc<dyn AnyTool>, working_directory: Option<PathBuf>) -> Self {
        Self {
            tool,
            working_directory,
        }
    }

    fn refusal(&self, arguments: &str) -> Option<ToolOutput> {
        let arguments: Value = serde_json::from_str(arguments).ok()?;
        let command = arguments.get("command")?.as_str()?;
        root_search_refusal(command, self.working_directory.as_deref()).map(ToolOutput::failure)
    }
}

impl AnyTool for RootSearchGuard {
    fn name(&self) -> &str {
        self.tool.name()
    }

    fn needs_approval(&self) -> bool {
        self.tool.needs_approval()
    }

    fn always_asks(&self) -> bool {
        self.tool.always_asks()
    }

    fn definition(&self) -> ToolDefinition {
        self.tool.definition()
    }

    fn supports_remote(&self) -> bool {
        self.tool.supports_remote()
    }

    fn is_concurrency_safe(&self) -> bool {
        self.tool.is_concurrency_safe()
    }

    fn execute_remote<'a>(
        &'a self,
        arguments: &'a str,
        context: &'a ExecutionContext,
        remote: &'a RemoteToolExecutor,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
        match self.refusal(arguments) {
            Some(refusal) => Box::pin(async move { refusal }),
            None => self.tool.execute_remote(arguments, context, remote),
        }
    }

    fn execute<'a>(
        &'a self,
        arguments: &'a str,
        context: &'a ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
        match self.refusal(arguments) {
            Some(refusal) => Box::pin(async move { refusal }),
            None => self.tool.execute(arguments, context),
        }
    }
}

fn root_search_refusal(command: &str, working_directory: Option<&Path>) -> Option<String> {
    let commands = shell_syntax::commands(command);
    let words: Vec<Vec<String>> = commands
        .iter()
        .map(|command| command.words.iter().map(literal_or_empty).collect())
        .collect();
    let mut at_root_by_subshell = HashMap::new();
    commands
        .iter()
        .zip(&words)
        .find_map(|(command, words_of_command)| {
            if let Some(directory) = changed_directory(words_of_command) {
                at_root_by_subshell.insert(
                    command.subshells.last().copied(),
                    ROOT_OPERANDS.contains(&directory),
                );
                return None;
            }
            let at_root = command
                .subshells
                .iter()
                .rev()
                .map(|subshell| Some(*subshell))
                .chain([None])
                .find_map(|subshell| at_root_by_subshell.get(&subshell).copied())
                .unwrap_or(false);
            let upstream = command.upstream.map(|index| words[index].as_slice());
            refusal(words_of_command, at_root, working_directory).or_else(|| {
                find_feed_refusal(words_of_command, upstream, at_root, working_directory)
            })
        })
}

fn literal_or_empty(word: &Word) -> String {
    word.literal().unwrap_or_default().to_string()
}

fn changed_directory(words: &[String]) -> Option<&str> {
    match words {
        [cd, directory, ..] if cd == "cd" => Some(directory),
        [cd] if cd == "cd" => Some("~"),
        _ => None,
    }
}

fn refusal(words: &[String], at_root: bool, working_directory: Option<&Path>) -> Option<String> {
    let (program, args) = without_prefix(words).split_first()?;
    let name = search_program(program)?;
    Search::parse(name, args, at_root)
        .scans_root()
        .then(|| refusal_message(name, working_directory))
}

fn find_feed_refusal(
    words: &[String],
    upstream: Option<&[String]>,
    at_root: bool,
    working_directory: Option<&Path>,
) -> Option<String> {
    let (program, args) = without_prefix(words).split_first()?;
    let search = match program_name(program)? {
        "find" if unfiltered_root_find(args, at_root) => args
            .windows(2)
            .find(|pair| matches!(pair[0].as_str(), "-exec" | "-execdir"))
            .and_then(|pair| search_program(&pair[1])),
        "xargs" => {
            let (upstream_program, upstream_args) = without_prefix(upstream?).split_first()?;
            (program_name(upstream_program)? == "find"
                && unfiltered_root_find(upstream_args, at_root))
            .then(|| xargs_command(args).first())
            .flatten()
            .and_then(|command| search_program(command))
        }
        _ => None,
    }?;
    Some(format!(
        "Command blocked: Blocked: {search} over the files find / lists reads everything under \
         /proc, /sys and /dev, and does not finish within the 10-minute command limit.{} Give \
         find a -name filter, or search a narrower path.",
        location(working_directory)
    ))
}

fn xargs_command(args: &[String]) -> &[String] {
    let mut rest = args;
    while let Some((arg, tail)) = rest.split_first() {
        rest = if arg == "--" {
            return without_prefix(tail);
        } else if XARGS_LONG_VALUE_FLAGS.contains(&arg.as_str()) {
            tail.get(1..).unwrap_or_default()
        } else if arg.starts_with("--") {
            tail
        } else if let Some(flags) = arg.strip_prefix('-').filter(|flags| !flags.is_empty()) {
            if takes_next_word(flags) {
                tail.get(1..).unwrap_or_default()
            } else {
                tail
            }
        } else {
            return without_prefix(rest);
        };
    }
    rest
}

fn takes_next_word(flags: &str) -> bool {
    flags
        .char_indices()
        .find(|(_, flag)| XARGS_VALUE_FLAGS.contains(*flag))
        .is_some_and(|(index, flag)| index + flag.len_utf8() == flags.len())
}

fn unfiltered_root_find(args: &[String], at_root: bool) -> bool {
    let mut args = args
        .iter()
        .skip_while(|arg| matches!(arg.as_str(), "-H" | "-L" | "-P"))
        .peekable();
    let mut starts = Vec::new();
    while let Some(start) = args.next_if(|arg| !arg.starts_with('-') && *arg != "(" && *arg != "!")
    {
        starts.push(start.as_str());
    }
    let rooted = starts.iter().any(|start| ROOT_OPERANDS.contains(start))
        || (at_root
            && (starts.is_empty()
                || starts
                    .iter()
                    .any(|start| CURRENT_DIRECTORY_OPERANDS.contains(start))));
    rooted && lists_everything(args)
}

fn lists_everything<'a>(mut expression: impl Iterator<Item = &'a String>) -> bool {
    while let Some(primary) = expression.next() {
        match primary.as_str() {
            "-exec" | "-execdir" => {
                for word in expression.by_ref() {
                    if word == ";" || word == "+" {
                        break;
                    }
                }
            }
            flag if FIND_LISTING_VALUE_FLAGS.contains(&flag) => {
                expression.next();
            }
            flag if FIND_LISTING_FLAGS.contains(&flag) => {}
            _ => return false,
        }
    }
    true
}

fn program_name(program: &str) -> Option<&str> {
    Path::new(program).file_name()?.to_str()
}

fn search_program(word: &str) -> Option<&str> {
    program_name(word).filter(|name| SEARCH_PROGRAMS.contains(name))
}

fn location(working_directory: Option<&Path>) -> String {
    working_directory
        .map(|directory| format!(" The current working directory is {}.", directory.display()))
        .unwrap_or_default()
}

fn refusal_message(name: &str, working_directory: Option<&Path>) -> String {
    let exclusions = if name == "rg" {
        "-g '!proc' -g '!sys' -g '!dev'"
    } else {
        "--exclude-dir=proc --exclude-dir=sys --exclude-dir=dev"
    };
    format!(
        "Command blocked: Blocked: recursive {name} over / walks the whole filesystem, \
         including /proc, /sys and /dev, and does not finish within the 10-minute command \
         limit.{} Search a narrower path, or add {exclusions}.",
        location(working_directory)
    )
}

#[derive(Debug, Default)]
struct Search {
    recursive: bool,
    rooted: bool,
    excludes_proc: bool,
}

impl Search {
    fn parse(name: &str, args: &[String], at_root: bool) -> Self {
        let rg = name == "rg";
        let mut search = Self {
            recursive: rg,
            ..Self::default()
        };
        let mut pattern_given = false;
        let mut paths_given = false;
        let mut operands_only = false;
        let mut args = args.iter();
        while let Some(arg) = args.next() {
            if operands_only || !arg.starts_with('-') || arg == "-" {
                if pattern_given {
                    paths_given = true;
                    search.rooted |= ROOT_OPERANDS.contains(&arg.as_str())
                        || (at_root && CURRENT_DIRECTORY_OPERANDS.contains(&arg.as_str()));
                } else {
                    pattern_given = true;
                }
            } else if arg == "--" {
                operands_only = true;
            } else if let Some(long) = arg.strip_prefix("--") {
                pattern_given |= search.long_flag(long, &mut args);
            } else {
                pattern_given |= search.short_flags(&arg[1..], rg, &mut args);
            }
        }
        search.rooted |= at_root && !paths_given;
        search
    }

    fn long_flag(&mut self, flag: &str, args: &mut Iter<'_, String>) -> bool {
        let (name, inline) = match flag.split_once('=') {
            Some((name, value)) => (name, Some(value.to_string())),
            None => (flag, None),
        };
        match name {
            "recursive" | "dereference-recursive" => self.recursive = true,
            "directories" => {
                self.recursive |= value(inline, args).as_deref() == Some("recurse");
            }
            "exclude-dir" => {
                self.excludes_proc |= value(inline, args).is_some_and(|dir| dir.contains("proc"));
            }
            "glob" | "iglob" => {
                self.excludes_proc |= value(inline, args).is_some_and(|glob| excludes_proc(&glob));
            }
            "regexp" | "file" => {
                let _ = value(inline, args);
                return true;
            }
            _ => {}
        }
        false
    }

    fn short_flags(&mut self, cluster: &str, rg: bool, args: &mut Iter<'_, String>) -> bool {
        let value_flags = if rg { RG_VALUE_FLAGS } else { GREP_VALUE_FLAGS };
        for (index, flag) in cluster.char_indices() {
            if !rg && (flag == 'r' || flag == 'R') {
                self.recursive = true;
            }
            if !value_flags.contains(flag) {
                continue;
            }
            let rest = &cluster[index + flag.len_utf8()..];
            let inline = (!rest.is_empty()).then(|| rest.to_string());
            let value = value(inline, args);
            match flag {
                'e' | 'f' => return true,
                'g' => self.excludes_proc |= value.is_some_and(|glob| excludes_proc(&glob)),
                'd' => self.recursive |= value.as_deref() == Some("recurse"),
                _ => {}
            }
            return false;
        }
        false
    }

    fn scans_root(&self) -> bool {
        self.recursive && self.rooted && !self.excludes_proc
    }
}

fn value(inline: Option<String>, args: &mut Iter<'_, String>) -> Option<String> {
    inline.or_else(|| args.next().cloned())
}

fn excludes_proc(glob: &str) -> bool {
    glob.starts_with('!') && glob.contains("proc")
}

fn without_prefix(words: &[String]) -> &[String] {
    let mut rest = words;
    while let Some((word, tail)) = rest.split_first() {
        rest = if is_assignment(word) || PREFIX_PROGRAMS.contains(&word.as_str()) {
            tail
        } else if word == "timeout" {
            without_timeout_arguments(tail)
        } else {
            return rest;
        };
    }
    rest
}

fn without_timeout_arguments(words: &[String]) -> &[String] {
    let mut rest = words;
    while let Some((word, tail)) = rest.split_first() {
        rest = match word.as_str() {
            "-s" | "-k" | "--signal" | "--kill-after" => tail.get(1..).unwrap_or_default(),
            option if option.starts_with('-') => tail,
            _ => return tail,
        };
    }
    rest
}

fn is_assignment(word: &str) -> bool {
    word.split_once('=').is_some_and(|(name, _)| {
        name.starts_with(|first: char| first.is_ascii_alphabetic() || first == '_')
            && name
                .chars()
                .all(|character| character.is_ascii_alphanumeric() || character == '_')
    })
}

#[cfg(test)]
#[path = "root_search_guard_test.rs"]
mod tests;
