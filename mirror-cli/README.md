# Mirror

Mirror is [Reflection's](https://reflection.ai/) open source agent platform for
coding and knowledge work. By building the model and harness together, we
believe we can uncover unique capabilities for agentic workflows.

We are starting with Mirror CLI, available alongside the launch of Reflection's
first model, [Beam](https://reflection.ai/blog/introducing-beam). Mirror CLI is
an agent harness for the terminal that works with Beam and other models. Key
features include:

- **[Parallel agents in worktrees](#parallel-worktrees).** Run independent tasks and switch between them without stopping their work.
- **[Inspect and steer from the terminal](#working-with-mirror).** Follow activity, approve tool calls, switch between manual approval and auto mode, interrupt work, and redirect the agent.
- **[Resume and branch conversations](#conversations-and-context).** Pick up saved work or try another approach from an earlier message.
- **[Beam and other models](#providers-and-models).** Switch providers or use a compatible self-hosted endpoint.
- **[Opt-in memory](#memory).** Keep preferences and useful repository context across sessions.
- **[Project instructions](#project-instructions) and [skills](#skills).** Give the agent your conventions and reusable task procedures.
- **[MCP and Python extensions](docs/extensions.md).** Connect external tools and customize commands, shortcuts, and terminal widgets.

Mirror CLI is still in development. Please [share your feedback](#feedback-and-contributions)
to help us improve it.

[Install](#install) · [Quick start](#quick-start) · [Providers](#providers-and-models)
· [Usage](#working-with-mirror) · [Configuration](#customize-mirror)
· [Security](#local-data-and-safety) · [Documentation](#documentation)

## Install

Install a supplied release wheel with
[uv](https://docs.astral.sh/uv/getting-started/installation/). Replace the path
with the wheel for your platform:

```bash
uv tool install '/path/to/mirror-<version>-py3-none-<platform>.whl'
mirror --version
```

Release wheels include the native agent executable and ripgrep. You do not need
Rust or a separate agent installation to use one.

The public PyPI package named `mirror` is unrelated. Do not install Mirror with
`uv tool install mirror` or `pip install mirror`.

If your shell cannot find the command, run `uv tool update-shell` and restart
the shell.

### Run from source

Requires uv, Rust, a C compiler, `protoc`, and ripgrep (`rg`).

```bash
git clone https://github.com/reflection-oss/mirror.git
cd mirror
./mirror-cli/mirror-dev
```

This builds and runs your checkout without replacing the installed `mirror`.
Rerun after editing. See [development checks](docs/building.md#run-checks).

### Updates

- Requires `uv` on `PATH`; interactive installs check PyPI and ask first.
- Accept → verified side-by-side install; launch immediately and on later runs.
- Decline/failure → current version. Original installation and user data stay intact.
- `--no-update`: skip checking. `--update`: retry with a longer timeout.
- `MIRROR_SKIP_UPDATE=1`: bypass checks and cached releases.
- Noninteractive runs: no checks/prompts. Source checkouts: no opt-in.

### Requirements

| Requirement | Supported |
| --- | --- |
| macOS | macOS 15 or later on Apple Silicon |
| Linux | x86-64 or ARM64, with glibc 2.28 or later |
| Python | 3.12 or later; uv can download a compatible interpreter |
| Model access | Credentials for a supported provider, or a compatible endpoint |

Native Windows is not supported. Git or Jujutsu is needed for repository
worktree management; Mirror can also work in a plain directory.

## Quick start

Start Mirror in the directory you want it to work in:

```bash
cd /path/to/your/project
mirror --api reflection --approval-mode approve
```

Create an API key at
[platform.reflection.ai/api-keys](https://platform.reflection.ai/api-keys).
If a key is missing, Mirror opens a masked sign-in prompt. You can also provide
`REFLECTION_API_KEY` through your shell's secret-management workflow. Reflection
requests go to `https://api.reflection.ai/openai/v1`.

Then give Mirror a task:

```text
Explain how this project is organized and how to run its tests.
```

Or start with a concrete change:

```text
Find why the parser accepts an empty identifier. Fix it, add a regression test,
and run the relevant tests.
```

Review tool requests as they appear and inspect the resulting changes. The
example explicitly selects manual approval. This is also the default unless
automatic approval was previously saved in your configuration. Use
`/settings` → **CLI mode** to change it while Mirror is running.

Press `Escape` to interrupt active work, then give a revised instruction. Type
`/` to browse commands, `/help` for help, and `/model` to choose another model.
Mirror runs tools with your account's permissions; approval does not provide
filesystem or network isolation. See [Local data and safety](#local-data-and-safety).

To look around the interface without a model connection, run
`mirror --showcase` and send a message. It plays a synthetic conversation without
calling a model or running real tools.

## Providers and models

Select a provider with `--api` and a model with `--model`. Exporting a provider's
API key alone does not select that provider.

| Provider | `--api` value | Authentication |
| --- | --- | --- |
| Reflection | `reflection` | `/login`, a saved Reflection key, or `REFLECTION_API_KEY` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` |
| OpenAI Responses | `openai-responses` | `OPENAI_API_KEY` |
| Anthropic Messages | `anthropic-messages` | `ANTHROPIC_API_KEY` |
| ChatGPT plan through Codex | `openai-codex-responses` | Existing Codex CLI login |
| Custom Chat Completions endpoint | `openai-chat-completions` | Endpoint-specific; `--api-key` if required |

For example, with OpenRouter:

```bash
export OPENROUTER_API_KEY="your-api-key"
mirror --api openrouter --model 'provider/model-id'
```

Replace `provider/model-id` with a model ID available to your account. For a
local or self-hosted Chat Completions server:

```bash
mirror --api openai-chat-completions \
  --base-url http://127.0.0.1:8000/v1 --model your-model-name
```

The endpoint must support the tool calls Mirror uses. A local model server
runs separately from Mirror.

For ChatGPT plan access, first sign in with the Codex CLI using
`codex login --device-auth`, then launch:

```bash
mirror --api openai-codex-responses
```

Mirror reads `~/.codex/auth.json` by default. Set `MIRROR_CODEX_AUTH_PATH` to use
another credential file. An OpenAI API key does not replace that plan login.

Inside a conversation, `/model` opens the model picker and `/reasoning` selects
reasoning effort. Model and reasoning changes apply to the next model request.
Each workspace keeps its own selection. You can add named models and custom
endpoints in [`~/.mirror/models.json`](docs/configuration.md#model-picker).

Without an explicit provider, a saved selection, or configured model routing,
Mirror uses Reflection. A custom endpoint selects Chat Completions instead.

### Reflection sign-in

`/login` signs in when no Reflection key is available. If a key already exists,
it reports that without changing your model. Use `/model reflection` to switch
to Reflection; it opens sign-in when a key is missing. Press `Escape` at sign-in
to choose another model.

`/logout` removes the saved key at `~/.mirror/reflection_api_key`. It does not
revoke the key at the provider or cancel requests already in flight. If you
set `REFLECTION_API_KEY`, unset it in your shell and restart Mirror as well.
If Reflection is active and no key remains, Mirror reopens sign-in.

## Working with Mirror

### Terminal layout

The composer stays at the bottom while conversation history scrolls. Use Page Up,
Page Down, or the mouse wheel to scroll; Alt+Home goes to the oldest entry and
Alt+End follows new output. Escape or the **Back to bottom** hint also resumes
following output, unless a menu or editor handles Escape first.

Drag to select visible text and copy it on release. Copying uses OSC 52;
your terminal's clipboard permissions still apply.

### Coding and investigation

Tell Mirror what you need done and how to check the result. For example:

```text
Trace the login flow from the HTTP route to token storage. Show me where expiry
is checked before making any changes.
```

```text
Add a --json option to the status command. Preserve the existing text output,
cover both formats in tests, and run the command's test suite.
```

Use your editor and version control to review the changes. `/editor` or
`Ctrl+P` opens a workspace file in your editor. `/view` or `Ctrl+T` cycles
transcript detail between **compact** (default), **focus**, and **detail**.
The status-bar badge shows the current mode. Compact groups tool activity; focus
emphasizes answers; detail shows full tool output and available reasoning.

### Research and analysis

Mirror can also work with documents and data in your working directory:

```text
Read the design documents in this directory. Compare the proposed approaches,
identify the decisions still open, and write a summary with file and section
references for each finding.
```

Connect services through MCP when a task needs information from other systems.
For web search and browsing, configure a [Firecrawl](https://firecrawl.dev/) key
through `FIRECRAWL_API_KEY` before launching Mirror.

### Conversations and context

Conversations are saved locally, so you can leave and return later:

```bash
cd /path/to/your/project
mirror --continue
```

`--continue` resumes the most recent conversation in each worktree. Use
`/resume` to browse saved conversations, or `/resume <id-or-prefix>` to choose
one directly. `/new` starts a fresh conversation in the current workspace.

`/rewind` branches conversation history from an earlier message. It does not
undo changes to files; use version control or backups for that.

Mirror compacts context automatically as it approaches the configured context
limit. Use `/compact` while idle to compact sooner, keeping a summary and recent
messages. Short conversations retain their recent messages without a
summarization request. `Escape` cancels compaction; a cancelled or failed summary
does not replace the context.

### Parallel worktrees

In a Git or Jujutsu repository, create another workspace:

```text
/worktree parser-fix
```

Mirror creates a separate checkout from `origin/main` and starts its own agent.
This requires access to the repository's `origin` remote and its `main` branch.
Local edits in the current checkout are not copied to the new worktree.

Give the new agent its task, then use `Tab` and `Shift+Tab` to switch worktree
tabs. Agents keep working when you switch tabs. `/workspaces` opens a searchable
workspace picker, and `/status` shows the active conversation and directory.
Each worktree keeps its own conversation, model, and approval settings. Use
`/rename <alias>` to give the current worktree a display name.

### Command reference

| Task | Command |
| --- | --- |
| Show help | `/help` |
| Select a model or reasoning effort | `/model`, `/reasoning` |
| Sign in to Reflection or remove its saved key | `/login`, `/logout` |
| Start or resume a conversation | `/new`, `/resume [id-or-prefix]` |
| Name the current worktree | `/rename <alias>` |
| Branch conversation history | `/rewind` |
| Compact context | `/compact` |
| Create or switch workspaces | `/worktree [name]`, `/workspaces` |
| Open an existing directory | `/folder <path>` |
| Close a workspace without deleting files | `/close` |
| Choose or edit [workspace tags](docs/workspace-tags.md) | `/tag` |
| Show session information | `/status` |
| Choose tool approval mode | `/settings` → **CLI mode** |
| Change transcript detail | `/view` |
| Open a file in your editor | `/editor` |
| Turn memory on or off | `/settings` → Memory |
| Check or import memories | `/memory`, `/memory import` |
| Manage service connections | `/connect`, `/connections`, `/disconnect` |
| Manage extensions | `/extensions`, `/extensions-add`, `/extensions-update`, `/extensions-remove` |
| Clear terminal scrollback | `/clear` |
| Exit | `/quit` or `/exit` |

Extension commands appear when the relevant extension is enabled. Run
`mirror --help` for the full list of launch flags.

### Keyboard shortcuts

| Key | Action |
| --- | --- |
| `Enter` | Send the prompt |
| `Ctrl+J` | Insert a newline when command completion is closed |
| `Escape` | Interrupt work or dismiss a menu |
| `Tab` / `Shift+Tab` | Switch worktree tabs when a picker or completion menu is closed |
| `Ctrl+R` | Browse prompt history |
| `Ctrl+P` | Open the editor picker |
| `Ctrl+T` | Cycle transcript detail |
| `Ctrl+S` | Cycle the current workspace tag |
| `Ctrl+O` | Open the workspace picker |
| `F2` | Toggle tool approval mode |
| `Ctrl+C` | Dismiss a menu, clear a draft, interrupt work, or quit when idle |
| `Ctrl+D` | Quit immediately |

Shortcuts depend on the active menu or prompt. You can change them in
[`~/.mirror/keybindings.json`](docs/configuration.md#keybindings).

## Noninteractive use

```bash
mirror exec 'Explain this project' > answer.md
mirror exec < prompt.txt
```

## Customize Mirror

### Project instructions

Put repository guidance in `AGENTS.md` or `CLAUDE.md`. Mirror loads instructions
from the repository root through the working directory, preferring `AGENTS.md`
when both files exist. Outside a repository, it reads the launch directory.
Global instructions can live in `~/.mirror/AGENTS.md`.

For example, an `AGENTS.md` might say:

```markdown
Use the existing formatting and naming conventions.
Run `make test-parser` after changing the parser.
Keep generated files unchanged unless their source is updated too.
```

Use these files for commands, conventions, and project context. They become
part of the model's instructions; they do not enforce access restrictions.

### Skills

Skills are reusable task instructions in `SKILL.md` files. Mirror discovers
user skills under `~/.mirror/skills/` and project skills under `.agents/skills/`
or `.claude/skills/`. Project skills override user skills with the same name.
Enabled extensions can also supply skills.

For example, create `.agents/skills/review-api/SKILL.md`:

```markdown
---
name: review-api
description: Review API changes for compatibility and error handling.
---
Check request validation, response compatibility, and failure paths.
Identify the relevant tests and report any missing coverage.
```

Start a new conversation to load the skill, then mention it in a prompt:

```text
Use $review-api to review the changes in this branch.
```

### Memory

Memory is off by default. Use `/settings` → **Memory** to turn it on or off.
Changing it restarts the current session's worker; finish or interrupt an active
turn first. `/memory` shows its status. The memory store lives at
`~/.mirror/memory`, with pinned context in `~/.mirror/PINNED_MEMORIES.md`.

`/memory import` imports memories from Claude Code and Codex on this machine.
Memory files are stored locally, and relevant memories are included in model
context. Remote sessions do not have memory. See the
[memory configuration guide](docs/configuration.md#memory) for source selection
and storage details.

### MCP connections

Use `/connect` to browse services and connect them, `/connections` to check
status, and `/disconnect` to remove a connection. You can also define a custom
MCP server in `~/.mirror/mcp/<name>.json`.

For a trusted, installed stdio server, a definition looks like this:

```json
{
  "command": "/path/to/mcp-server",
  "args": ["--stdio"]
}
```

Replace the executable path and arguments with those your server requires,
then use `/connect <name>`. Each file is one server definition, without an
`mcpServers` wrapper. Project-local MCP configuration is not loaded.

Mirror supports stdio and Streamable HTTP, including OAuth and bearer
credentials. CLI-hosted MCP tools are exposed to local agent sessions; remote
sessions use their server-owned tools. See [Extensions and MCP](docs/extensions.md)
for authentication, tool selection, and timeouts.

### Extensions

Extensions can add terminal widgets, slash commands, keybindings, skills, and
tools. Put local Python extensions under `~/.mirror/extensions/` or a project's
`.mirror/extensions/`, then use `/extensions` to enable them. Use
`/extensions-add` for a GitHub source and `/extensions-update` to update it.

External extensions require consent before import and run with your account's
permissions. Only enable code you trust. See the
[extension guide](docs/extensions.md) for installation, authoring, and examples.

### Configuration files

| Location | Purpose |
| --- | --- |
| `~/.mirror/config.json` | Extension settings and saved approval preferences |
| `~/.mirror/models.json` | Model picker entries and provider routing |
| `~/.mirror/keybindings.json` | Keyboard shortcut overrides |
| `~/.mirror/AGENTS.md` | Global instructions |
| `~/.mirror/tool_approvals.json` | Remembered tool approvals |

See [Configuration](docs/configuration.md) for formats, precedence, and supported
configuration overrides.

## Local data and safety

Mirror runs tools locally with your account's permissions. The working directory
is not an access boundary, and manual approval is not a sandbox. Extensions and
local MCP servers can also access your filesystem, network, and credentials.
Review commands and changes, and use a container or virtual machine when you
need stronger isolation.

Model requests send conversation context, including relevant file contents,
instructions, memories, and tool results, to the selected provider. Connected
MCP servers and web services receive their tool requests. Review access before
using Mirror with sensitive repositories.

| Location | Contents |
| --- | --- |
| `~/.mirror/state.db` | Agent conversations |
| `~/.mirror/state.db.locks/` | Locks for conversations open in a worker |
| `~/.mirror/cli.db` | CLI history and workspace settings |
| `~/.mirror/logs/<conversation-id>/` | Worker logs |
| `~/.mirror/reflection_api_key` | Saved Reflection API key |
| `~/.mirror/mcp/credentials/` | MCP credentials |
| `~/.mirror/memory/` | Local memory store, when enabled |

Conversation history, memories, and logs may contain sensitive information.
Use `--database <path>` to select another agent database. Resuming a conversation
from a different directory asks for confirmation. A conversation can be open
in only one worker at a time.

Report suspected vulnerabilities privately to
[developers@reflection.ai](mailto:developers@reflection.ai). See
[SECURITY.md](SECURITY.md) for report scope and security boundaries.

## Upgrade and troubleshooting

Exit Mirror and install the new supplied release wheel directly:

```bash
uv tool install '/path/to/mirror-<version>-py3-none-<platform>.whl'
mirror --version
```

Use `--force` when replacing a wheel with the same package version. Avoid
registry upgrades for the unrelated PyPI `mirror` package.

Configuration and conversations live outside the tool installation in
`~/.mirror/`. Back up that directory with Mirror stopped before changing
versions if you need a recovery point. Restart with `mirror --continue`.
An older executable may not understand state written by a newer release;
reinstalling it is not a database rollback.

| Problem | Check |
| --- | --- |
| `mirror: command not found` | Run `uv tool update-shell`, then restart your shell. |
| Missing provider credentials | Select the intended `--api`; use `/login` for Reflection or provide that provider's credentials. |
| ChatGPT plan authentication fails | Check the Codex login and the selected credential file. |
| `/worktree` fails | Check Git or Jujutsu, access to `origin`, and the remote `main` branch. |
| Conversation is already open | Close the Mirror process using it before resuming it elsewhere. |
| An MCP tool or extension is missing | Check `/connections` or `/extensions` and the corresponding guide. |
| Installation or upgrade fails | Resolve the reported network or permission error and retry; keep `~/.mirror/` for recovery. |

## Documentation

| Guide | What it covers |
| --- | --- |
| [Reflection API docs](https://developers.reflection.ai/introduction) | API access, authentication, models, and provider guides |
| [Configuration](docs/configuration.md) | Providers, model routing, instructions, approvals, keybindings, and memory |
| [Extensions and MCP](docs/extensions.md) | Service connections, custom servers, extension management, and authoring |
| [Architecture](docs/architecture.md) | CLI, native worker, state, and conversation lifecycle |
| [Terminal themes](docs/themes.md) | Palette roles, `/themes`, and live refresh |
| [Workspace tags](docs/workspace-tags.md) | Tag workspaces and filter the picker |
| [Develop from source](docs/building.md) | Run local changes, test, and build a wheel |

The source includes a Python terminal client and a Rust agent worker. Release
wheels bundle both. The build guide explains how to build your own copy from
the full exported source checkout.

## Feedback and contributions

We welcome bug reports and feature requests through
[GitHub issues](https://github.com/reflection-oss/mirror/issues). Search existing
issues first, then use `/feedback` in the CLI or the
[GitHub issue form](https://github.com/reflection-oss/mirror/issues/new/choose).
`/feedback` opens that form. Include your Mirror version, operating system,
terminal, reproduction steps, and relevant errors. Remove secrets and private
information before posting.

For now, we are not accepting external code or documentation contributions.
We will consider opening up contributions in the future. If you would like to
contribute, express your interest through `/feedback` or the GitHub issue form.
See [CONTRIBUTING.md](CONTRIBUTING.md) for reporting guidance and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for discussion expectations.

Use the private [security reporting channel](SECURITY.md#reporting-a-vulnerability)
for suspected vulnerabilities.

## License

Mirror is licensed under [Apache-2.0](LICENSE). See [NOTICE](NOTICE) for attribution.
Third-party components retain their respective licenses. Native wheels include
`LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.txt` under `mirror/bin/`.
