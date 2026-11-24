# Extensions and MCP

Extend Mirror's terminal UI with trusted Python code, or connect tools through
MCP. The CLI hosts extensions and interactive sign-in; the Rust agent runtime
owns MCP discovery and execution. Extensions and local stdio servers run without
a sandbox.

## Enable trusted extensions

Put an extension at `~/.mirror/extensions/<name>/extension.py`. Mirror also
discovers `.mirror/extensions/<name>/extension.py` from the nearest repository
root through the launch directory (only the launch directory outside a repo).

Use `/extensions` and press Enter to toggle an entry. Choices persist in
`~/.mirror/config.json`; disabling an extension does not delete its files.
New external extensions require consent before import. `--trust-extensions`
remembers new ones as enabled without prompting, but leaves disabled ones off.

**Only enable code you trust.** Extensions execute unsandboxed Python with your
account's filesystem, network, and credential access. Consent is not a code
review or a sandbox; review project-provided extensions before enabling them.

## GitHub sources

- `/extensions-add`: paste URL → download → consent.
- `/extensions-update`: pick one/all → reload enabled extensions now.
- `/extensions-remove`: confirm → unload; keep cache.

Root `extension.py`; ID = unique lowercase repo name. Local extensions win.
GitHub URLs are `extensions` keys in `~/.mirror/config.json`; optional `ref`
selects a branch/tag/SHA (default branch otherwise). Apply ref changes with
`/extensions-update`; failed activation keeps the previous installed revision.

Startup downloads missing sources; installed sources work offline. No automatic
updates, hooks, submodules, or dependency installs. Private repos use Git credential
helpers—never put tokens in URLs or settings.

## Local folders

`/extensions-add ~/src/clock` registers a live directory containing `extension.py`.
Paths are on the CLI host, relative to its launch directory.
Enable with `/extensions`; disable/re-enable after edits.
`/extensions-remove` unregisters it without deleting files or settings.

## Author an extension

Enable `user_extensions` in `/extensions` for authoring tools and
`$mirror-extensions` (off by default).

Start with the widget scaffold in
[`extension.py`](../src/mirror/cli/extensions/user_extensions/extension.py).
The model-facing `user-extensions__extensions` tool supports these JSON arguments
(these are tool calls, not shell or slash commands):

```json
{"action": "create", "name": "clock"}
{"action": "load", "name": "clock"}
{"action": "reload", "extension_id": "clock"}
{"action": "unload", "extension_id": "clock"}
{"action": "list"}
```

Create, edit the generated file, then load. The resolved directory name is its
ID; keep it stable.
Load/unload persist enablement; list reports user extensions and workspace faults.
The tool manages the user directory, not project extension directories.

The mental model is a small state machine:

- Define an `Extension[StateT, EffectT, ContextT]` subclass with an `id()` equal
  to its directory name and a no-argument constructor. The loader rejects
  mismatches before activation. Use the module docstring for its menu description.
- `initial_state()` starts each workspace's state; `reduce()` returns new state
  and effects. `render()` builds UI from state. Keep I/O out of both.
- `handle_effect()` performs async work; `self.dispatch()` routes updates back
  to the extension. Preserve task cancellation when writing long-running effects.
- `on_load()` / `on_unload()` manage shared resources. `slash_commands()`,
  `keybindings()`, and `tools()` contribute interactions.
- Use `context.workspace.run()` or `context.workspace.shell.open()` for workspace
  I/O. The workspace owns execution location; extensions need no location declarations.

Dispatch `RequestSecret(SecretRequest(...))` to ask for a secret such as a
token. The CLI shows a masked prompt with bullets and a character count. On
Enter it clears the prompt and passes the input to `on_submit`, whose result
goes only to the requesting extension's `handle_effect`. The input never enters
UI state, dispatched actions, prompt history, or the transcript.

Dispatch `CreateRemote(title, workspace)` to open a remote tab whose new hosted
session starts in that `mirror_pb2.Workspace`; resuming a session ignores it.

When answering `ApprovalRequested`, an extension can supply
`SendApprovalDecision(call_id, False, reason='…')`. The reason reaches the model
in that call's tool result; blank reasons default to `Tool call rejected.`.
The CLI supplies `Tool call rejected by the user.` for its Reject action.
Reasons are stored in conversation history and sent to the model: do not include
secrets. This does not add a pre-execution review hook or change approval rules.
For runtime-owned checks that do not depend on an attached CLI, use
[pre-tool command hooks](hooks.md).

See the [extension API](../src/mirror/cli/extension.py) and the
[bundled context widget](../src/mirror/cli/extensions/context_usage/extension.py).

## Contribute themes

See the runnable [Tokyo Night extension](../extensions/tokyonight/extension.py):
its `themes()` returns two palettes available through `/themes`.

## Contribute tips

Omit `tips()` unless explicitly asked; tips are mainly for broadly shared
extensions, not personal or one-off widgets:

```python
def tips(self) -> tuple[str, ...]:
    return ('Use /my-command to open this extension',)
```

## Contribute skills

Override `skills()` on your extension:

```python
import pathlib

def skills(self) -> tuple[pathlib.Path, ...]:
    return (pathlib.Path(__file__).with_name('SKILL.md'),)
```

Add `name` and `description` YAML frontmatter. Ship the files as package data.
Reload the extension after editing its skills. Never include secrets.

- Skills require an enabled extension in an applicable local workspace.
- User/project skills override same-named extension skills.
- Reload updates completion immediately and `$name` invocation after the idle
  worker refresh. The model's catalog updates on new conversations or compaction.
- `disable-model-invocation: true` hides a skill from the catalog, not `$name`.

## Connect services and MCP tools

`core/connections` loads MCPs from this machine, independently of the desktop
app's placement preferences. It never discovers or delegates server MCPs.
Local sessions pass endpoint and credential references to their Rust worker;
SSH sessions use the daemon host's configuration, never laptop credentials.

- `/connect` lists the local catalog and configured MCPs.
- `/connect <service>` signs in locally, using its saved OAuth client ID when set.
- `/connections` lists local connections and their status.
- `/disconnect <service>` removes the local definition and credentials.
  `/disconnect` opens a picker.

Explicit remote-session creation still checks `required_remote_connections`
(default `["github"]`) on the configured `--remote-endpoint`. Missing connections
start server-side sign-in; ambiguous accounts require a default selection.
Token setup accepts masked input or the configured `token_commands` command
(default `{"github": "gh auth token"}`). Esc cancels sign-in. A failed readiness
check stops creation. This does not change local MCP routing; hosted sessions
resolve their own tools without receiving local endpoint or credential paths.

### Local servers

Custom servers stay on this machine. Local definitions are individual JSON
objects in `~/.mirror/mcp/<name>.json`, not an `mcpServers` wrapper. Names
allow letters, numbers, underscores, and dashes. `MIRROR_MCP_DIR` overrides that
directory. Project-local MCP config is not loaded. Each definition needs exactly
one of `url` (Streamable HTTP) or `command` (stdio). Connecting or disconnecting
one replaces the local worker so its stable tool set is discovered again.

#### HTTP: OAuth or bearer

For example, connecting Notion on this machine creates `notion.json` with:

```json
{
  "url": "https://mcp.notion.com/mcp",
  "auth": { "type": "oauth", "credential": "notion" }
}
```

OAuth credentials live under `credentials/` in the MCP directory.
`credential` defaults to the server name; `clientId` is optional inside `auth`.
Writing OAuth JSON alone does not sign in; after writing a file,
`/connect <name>` starts it or signs in.

For a custom bearer server (replace the example URL):

```json
{
  "url": "https://mcp.example.com/mcp",
  "auth": { "type": "bearer", "tokenEnvVar": "MY_MCP_TOKEN" }
}
```

Set `MY_MCP_TOKEN` in the CLI environment through your secret-management workflow.
This environment-variable path is the supported bearer compatibility mechanism;
do not put token values in JSON. Omit `auth` for unauthenticated HTTP.
Authenticated HTTP requires HTTPS except on loopback hosts.

#### Stdio

Replace the executable path with a trusted, installed MCP server:

```json
{
  "command": "/path/to/mcp-server",
  "args": ["--stdio"],
  "env": { "API_TOKEN": "MY_MCP_TOKEN" }
}
```

`args` and `env` are optional. `env` maps child variable names to existing CLI
variable names, **not literal values**. The child inherits only `PATH`, `HOME`,
`TMPDIR`, `LANG`, and explicitly mapped variables, not provider credentials.
This is environment filtering, not filesystem isolation. Trust the executable
before granting access.

Both transports accept `enabledTools` (original tool names; omitted means all,
`[]` means none), `startupTimeoutSeconds` (default 15), and `toolTimeoutSeconds`
(default 300). Timeouts must be positive seconds.

The model uses the three-bridge pattern from [Hermes](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-search), with MCP-specific names:

- `mcp_search({"queries":["create issue"],"limit":5})`
- `mcp_describe({"names":["mcp__issues__createIssue"]})`
- `mcp_call({"calls":[{"name":"mcp__issues__createIssue","arguments":{"title":"Example"}}]})`

These are harness tools, not MCP protocol methods. Search returns per-query
`matches` and a shared `tools` map containing descriptions and required argument
names. Describe returns full schemas and a `not_found` list. All three share the
same runtime index and dispatcher; there is no action-based router tool.

Search uses lexical BM25 ranking over tools discovered from the configured
servers. The initial manifest contains only server/tool counts, capped at 256
bytes. Descriptions and schemas are fetched on demand; all are untrusted data.
Router responses are limited to 32 KiB of serialized JSON. Oversized discovery
responses fail explicitly instead of returning incomplete schemas. Call results
retain success status and mark shortened content with `truncated: true`. Calls execute sequentially and request approval under the
original tool name, after argument validation. Manual approval fails closed
without an attached approval handler; automatic approval works detached.
Original `mcp__<server>__<tool>` bindings live behind the bridges, not in the
model registry. Extensions invoke them through `context.call_tool("mcp_call", ...)`,
using the same argument validation and approvals as model calls. Tool allowlists
select the three router names; configured connections and gateway authorization
restrict the underlying tools.

Local HTTP requests and stdio processes run in the worker on the CLI machine;
connected servers receive tool arguments. The CLI performs interactive OAuth
sign-in; the runtime refreshes the same private credential files. Hosted
sessions use their existing credential gateway, not local credentials.
See [definitions/auth](../src/mirror/cli/extensions/connections/configuration.py)
and [local runtime connections](../../mirror-agent/src/mcp/local_mcp_credentials.rs).

## Packaged extensions and precedence

For the same resolved folder ID, nearest project sources win over ancestors,
then user sources, then `mirror/cli/packaged_extensions`. Resolution happens
before import. Disabled, unapproved, or broken local winners do not fall back
to packaged copies. A settings refresh unloads a replaced winner.

Packaged code is trusted as part of the installed application, but still follows
`default_enabled()` and explicit user settings. Settings use stable
`packaged/<folder>` keys. Local overrides keep separate path-based consent;
packaged trust never transfers to them. `MIRROR_EXTENSIONS_ENABLED=false`
suppresses project, user, and packaged discovery, not core built-ins.

External entrypoint IDs must match their resolved folder name: migrate
`user/clock` to `clock`, including routing and tool-name references. Path-based
settings do not change. Core built-ins and programmatic children retain explicit
IDs; directory symlinks retain the target folder's identity.

## Optional dependencies

Ship `pyproject.toml` and `uv.lock` beside your extension. After explicit user
consent, call `extension_dependencies.prepare(project)`, then
`extension_dependencies.run(project, 'worker.py')` to launch a helper.
Requires uv ≥0.10.10; installs locked wheels with the current Python, never uv
or Python itself. Discovery and GitHub updates never install dependencies.
Runs reuse fingerprinted environments under `~/.mirror/extension-environments/`
and inherit no authentication variables. A venv is **not a security sandbox**.

## Workspace execution

Use `await context.workspace.read(path)` and `await context.workspace.write(path,
content)` for UTF-8 text. Newlines are preserved; invalid UTF-8, symlinks and
special files are rejected. Both accept `max_bytes` (default 2 MiB). Writes create
missing parents and replace files atomically, preserving existing permissions.
Pass `expected=original_text` to reject changes since a previous read; this requires
an existing file and checks again before replacement, without locking other writers.
Paths belong to the workspace backend, not necessarily the host filesystem.

Use `context.workspace.run(command)` for bounded stdout/stderr bytes and an exit
status, or `context.workspace.shell.open(argv)` for streaming output. Both accept
`data='text'` as UTF-8 stdin, supplied once and closed automatically.
Daemon execution requires a live workspace attachment. An attachment gap raises
`shell.ProcessError`; it never attaches or starts an agent. Status widgets should
show unavailable and refresh on `WorkerStarted`, discarding pre-disconnect reads.
Missing execution backends fail; they never fall back to the host. Personal
memory, credential commands, microphone capture and the user's editor remain
host-owned. The file picker reads and writes workspace files through the shell;
the host editor receives a temporary copy, with checked write-back.

## View-mode badge

The built-in `view_mode` extension displays the colored mode badge in the dock.
It is disabled by default; enable it in `/extensions` when you want the indicator.
Ctrl+T and `/view` cycle view modes whether the badge is enabled or disabled.

## Agent plugins

Use [`mirror plugins`](plugins.md) for Claude Code and Codex plugins shared by desktop and CLI.
Native Mirror extensions are managed separately in `/extensions`.
