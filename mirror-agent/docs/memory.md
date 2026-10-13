# Memory

Mirror can remember, across a user's sessions on one machine, how they like to
work, facts about their repositories, and procedures that worked. Memory is
private to the user, stored under their home directory, off until they turn it
on, and shown to the model only where it applies.

## What the user sees

| Capability | Behavior |
| --- | --- |
| Pinned memories | The bullets in `~/.mirror/PINNED_MEMORIES.md` are in the system prompt whenever memory is on, up to 4 KiB. The `save_memory` tool writes them; the file is hand-editable. |
| Recalled by message | Memories whose text matches the user's message follow it into context, at most five per message. |
| Attached by file | A memory tied to files follows the first tool result that touches one of them, at most three per tool batch. |
| `search_memory` | The model's own lookup by words, repository-relative paths, or a record id for the complete text. |
| `save_memory` | Remember, forget, pin and unpin, under the user's authority. |
| Import | `mirror-agent memory import` brings in Claude Code and Codex notes under the user's name. |

In the CLI, `/settings` → **Memory** opens a submenu to toggle memory or
**Import memories** from other harnesses. `/memory` or `/memory status` shows
its status, and `/memory import` also imports notes; see the
[CLI configuration guide](../../mirror-cli/docs/configuration.md).

## How memories are written

A memory is written in one of three ways: the model calls `save_memory` during
a turn, the user runs the import, or the user edits the pinned file. Mirror
does not yet learn from a session on its own; see [Not yet](#not-yet).

Every write is validated (`memory/memory_record.rs`) and appended as an event
that records who made it and in which session. A writer fills in:

| Field | Meaning |
| --- | --- |
| `type` | `preference`, `fact`, `procedure`, `pitfall` or `reference` |
| `key` | A kebab-case handle, used to replace or forget the memory later |
| `repository`, `path_globs` | Where it applies: one repository (empty means all) and, optionally, file patterns within it |
| `situation` | One sentence on when a future session should use it |
| `content` | The memory itself, at most 64 KiB (pitfalls 1 KiB) |

Imported memories also record the harness and file they came from.

## How memories reach the model

- **Pinned bullets** are read into the system prompt when it is built, at
  session start and after compaction.
- **Recall** runs through the runtime's context hooks
  (`react/context_hooks.rs`). After the context commits the user's message,
  memory's hook (`memory/conversation_memory.rs`) searches the store and
  appends what matches right after it. After a tool batch commits, the hook
  looks up memories tied to the files those tools touched, using the paths
  each tool reports in its result's metadata. Recalled text is framed as
  retrieved evidence, not instructions. Anything already in the context
  window is not added again, so resuming a session needs no extra state.
- **`search_memory`** is an ordinary tool the model calls when it wants more.

Memory learns the repository and checks paths through the runtime's
`Workspace` (`workspace.rs`), never by reading the disk or running `git`
itself.

## Where it lives

```text
~/.mirror/memory/memory.sqlite3   # records, events, FTS5 index (0600)
~/.mirror/PINNED_MEMORIES.md      # the always-on bullets
~/.mirror/config.json             # extensions."builtin/memory".config.enabled turns it on
```

The directory is created with mode `0700`. Within a process one thread owns
the SQLite connection, and every read and write goes through it. The store
contract is `MemoryStore` (`memory/memory_store.rs`), three calls: `search`,
`get` and `commit`; `SqliteMemoryStore` is its only implementation, and
ranking lives inside it. The schema is created on open from
`memory/schema.sql`; there are no migrations yet.

The local worker opens memory from `--memory-dir`
(`memory/memory_environment.rs`) and gets back the hooks, the tools and the
pinned file's path.

## Not yet

- **Learning from work.** Nothing extracts memories from a finished turn or
  tidies the store. The `extractor` and `consolidator` actors in the schema
  are reserved for those passes.
- **Hosted sessions** have no memory: a store per principal, a workspace over
  the sandbox and the pinned bullets as prepared instructions are follow-up
  work.
- **Subagents** never see memory.
