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

## Author an extension
Enable `user_extensions` in `/extensions` for authoring tools and
`$mirror-extensions` (off by default).

Start with the widget scaffold in
[`user_extensions.py`](../src/mirror/cli/user_extensions.py).
The model-facing `extensions` tool supports these JSON arguments (these are
tool calls, not shell or slash commands):

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
UI state, dispatched actions, prompt history, `cli.db`, or the transcript.

Dispatch `CreateRemote(title, workspace)` to open a remote tab whose new hosted
session starts in that `mirror_pb2.Workspace`; resuming a session ignores it.

See the [extension API](../src/mirror/cli/extension.py) and the
[bundled context widget](../src/mirror/cli/extensions/context_usage/extension.py).

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
One extension, `core/connections`, owns every service connection and the MCP
tools they provide. A service can be connected on the Mirror server at
`--remote-endpoint` or on this machine, and each service appears once:

- `/connect` opens a picker of everything connectable: services the Mirror
  server offers (`MirrorRuntime.ListProviders`), the curated local catalog, and
  configured local servers, each marked connected on the Mirror server,
  on this machine only, unavailable on this machine, or available. A service
  the Mirror server offers is marked by its Mirror server state even when a
  local server is configured, because that is where `/connect` signs in. A
  final entry explains how to add a custom server.
- `/connect <service>` signs in on the Mirror server when it offers the service.
  OAuth providers call `BeginOAuthConnection`, show the sign-in URL, open
  it in a browser, and wait on `WaitOAuthConnection`
  while the provider redirects to the Mirror server, which completes the
  sign-in itself; Esc cancels the wait. Token providers send a token through
  `CreateConnection`, either pasted into a masked prompt or read from the
  provider's command in the extension's `token_commands` config (default
  `{"github": "gh auth token"}`; an empty command turns the option off; Esc
  stops a running command). The Mirror server identifies the account. When several accounts are usable, the
  newly connected one becomes the default. When the Mirror server is
  unavailable or does not offer the service, the command signs in on this
  machine instead, except during `/remote` setup, which stops. A custom OAuth
  server signs in with its saved `auth.clientId` when it has one.
- `/connections` lists each connection with where it lives, its account, its
  status, and the default marker.
- `/disconnect <service>` revokes the Mirror server connection or removes the
  local server and its tokens, asking which one when there are several or when
  the Mirror server cannot be reached. `/disconnect` alone picks from every
  connection.

Every sign-in hint points at `/connect <service>` and is shown once per service
until that service connects. Before `/remote` opens a session, the extension
requires a usable connection for each provider in its
`required_remote_connections` config (default `["github"]`) that the Mirror
server offers; a missing one starts `/connect <provider>` and a missing default
opens an account picker, then the original request resumes. When the Mirror
server's connections cannot be listed, `/remote` stops with an error.

Tools are chosen per service. A service connected through the Mirror server's
MCP gateway uses the gateway's tools; otherwise a configured local server's
tools are used; never both, and a local tool whose name a gateway tool already
uses is left out. A gateway server belongs to its provider's service,
or to its own name when it has no provider. Local sessions pass connection declarations to their Rust worker; hosted
sessions discover their own authorized gateway catalog, without a CLI attachment.
Local endpoint and credential paths are never sent to hosted sessions.

### Mirror server gateway
The Mirror server's MCP gateway is off by default. Turn it on with the
extension's `remote_mcp` config, whose `~/.mirror/config.json` entry is
`builtin/connections`:

```json
{"extensions": {"builtin/connections": {"config": {"remote_mcp": true}}}}
```

Changing it takes effect in a running TUI. While it is off the client never
lists or opens gateway servers, so there are no gateway tools, gateway sign-in
hints, or gateway rows in `/connections`; `/connect`, `/disconnect`, and the
`/remote` readiness check still use the Mirror server's connections.

When it is on, the gateway is listed with `McpGatewayService.ListMcpServers`,
and each server is reached with MCP's Streamable HTTP transport at
`/mcp/<server>` on the same host and port, over HTTPS unless `--remote-insecure`
is set. The Mirror server authenticates the caller from its tailnet identity
and uses the caller's own connection for that provider, else the server's app
account; nothing is stored locally. The connection picker checks saved provider
connections before declaring gateway endpoints. An unreachable server is omitted
from runtime discovery. Remote workspaces use the hosted runtime's gateway
instead of opening connections from the CLI.

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
{"url": "https://mcp.notion.com/mcp", "auth": {"type": "oauth", "credential": "notion"}}
```

OAuth credentials live under `credentials/` in the MCP directory.
`credential` defaults to the server name; `clientId` is optional inside `auth`.
Writing OAuth JSON alone does not sign in; after writing a file,
`/connect <name>` starts it or signs in.

For a custom bearer server (replace the example URL):
```json
{"url": "https://mcp.example.com/mcp", "auth": {"type": "bearer", "tokenEnvVar": "MY_MCP_TOKEN"}}
```

Set `MY_MCP_TOKEN` in the CLI environment through your secret-management workflow.
This environment-variable path is the supported bearer compatibility mechanism;
do not put token values in JSON. Omit `auth` for unauthenticated HTTP.
Authenticated HTTP requires HTTPS except on loopback hosts.

#### Stdio
Replace the executable path with a trusted, installed MCP server:
```json
{"command": "/path/to/mcp-server", "args": ["--stdio"], "env": {"API_TOKEN": "MY_MCP_TOKEN"}}
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
status, or `context.workspace.shell.open(argv)` for persistent process streams.
Missing execution backends fail; they never fall back to the host. Personal
memory, credential commands, microphone capture and the user's editor remain
host-owned. The file picker reads and writes workspace files through the shell;
the host editor receives a temporary copy, with checked write-back.
