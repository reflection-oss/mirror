# Architecture

Mirror CLI owns the composer, menus, extensions, workspace lifecycle, and
interactive MCP sign-in in Python. The `mirror-transcript` subprocess owns
fullscreen transcript layout, scrolling, and terminal output. Python sends
ordered sources and rendered chrome over protobuf pipes. Agent workers use
AgentRPC; the model/tool loop never runs in the UI process.

```text
┌──────────────────────────────────────────────┐
│ CLI supervisor                               │
│ workspace tabs · input · rendering            │
│ extensions · connection declarations                  │
└───────────┬─────────────────────┬────────────┘
            │ AgentRPC            │ AgentRPC
┌───────────▼─────────┐ ┌─────────▼───────────┐
│ Worker: main        │ │ Worker: feature     │
│ one conversation    │ │ one conversation    │
└─────────────────────┘ └─────────────────────┘
```

Each started worktree has its own worker and prompt application. Inactive
worktrees keep receiving events while only the selected application renders.
Switching conversations within a worktree closes its worker before starting
the replacement.

## Workspace switching

- Roster/selection: controller. Data/lifecycle: workspace objects.
- Views/effects: CLI root; rendering reads the live roster.
- Exception: extensions retain `context.workspace_controller` for orchestration.
- Creation: immediately selectable; success replaces in place; failures stay.
- Completion preserves identity, order, and selection.
- `/workspaces`: preserve query/highlight; mark removed rows.
- Footer: overflow counts; `Tab` / `Shift+Tab` navigation.
- History: native transcript viewport; scrolling and resizing repaint automatically.
- Shutdown: drain creation/deletion; close agents.

## Source map

| Component | Owns |
| --- | --- |
| [Entrypoint](../src/mirror/cli/main.py) | Arguments, model configuration, and startup |
| [CLI](../src/mirror/cli/cli.py) | Composition/event loop, rendering, input, and AgentRPC sessions |
| [Workspace controller](../src/mirror/cli/workspace_controller.py) | Workspace roster, selection, creation, and persistence |
| [Workspaces](../src/mirror/cli/workspaces.py) | Workspace data and agent-backed lifecycle |
| [Workspace application](../src/mirror/cli/workspace_application.py) | Terminal frame handoff shared by agent and creation views |
| [Local transport](../src/mirror/cli/local_rpc_client.py) | Daemon discovery and workspace attachment |
| [RPC client](../src/mirror/cli/rpc_client.py) | Session requests and incoming events |
| [State](../src/mirror/cli/state.py) | UI state |
| [Actions](../src/mirror/cli/actions.py) | Typed UI actions |
| [Extensions](../src/mirror/cli/extension.py) | Extension state, effects, commands, and lifecycle |
| [Host tools](../src/mirror/cli/host_tools.py) | Tool registration and client-side dispatch |
| [Daemon client](../src/mirror/cli/daemon_client.py) | Workspace, history, and process RPCs |

## Inline animations

[`Animation`](../src/mirror/cli/components/animation.py) owns frame playback:

```python
loading = animation.Animation(frames=[' ●', '••', '● ', '••'], interval=0.1)
fragments = loading.render()
```

Frames can be plain strings or styled prompt_toolkit text. Playback is indexed
by the monotonic clock, not the number of renders; missed frames are skipped.
`Animation` does not own a timer. prompt_toolkit's shared 0.1-second refresh
interval drives the UI, so visible animations update at 10 frames per second.

## A turn

1. The client attaches to the selected daemon workspace. The daemon starts or
   joins its conversation's worker.
2. A user message enters the bidirectional `Session` stream with a session ID
   and message ID.
3. Runtime deltas update streaming output; committed conversation records are
   handled separately from transient previews.
4. Client-owned extension tools arrive as `host_tool_call` events. Python
   executes them and returns `host_tool_result`. MCP tools instead execute in
   the Rust runtime, without a client callback.
5. Turn completion updates the UI. Other worktrees can continue independently.

Model, reasoning, and speed changes are sent as inference controls rather than
forcing a worker restart. Changes during a request apply to the next model request.

## State and recovery

- Agent history: `~/.mirror/state.db`; CLI reads through daemon RPCs, never SQLite.
- Replay/live events: Session stream. Drafts, consent, extensions: client-local.
- One daemon-owned worker per conversation; client exit only detaches.
- Workspace metadata and history survive daemon restart.

`/rewind` forks conversation history; it does not roll back filesystem changes.
`/worktree` creates an independent checkout from `origin/main`; it does not copy
local edits from the current checkout. See the [quickstart](../README.md) for
everyday commands.

## Extension boundary

Extensions are client code, not worker code. They contribute UI state, effects,
commands, host tools, and serializable MCP connection declarations. Local
workers discover and execute those connections; hosted sessions discover their
own authorized gateway catalog. Neither runtime requires a CLI attachment to
execute MCP calls in automatic approval mode.

Extensions and local tools are not sandboxed. Only enable trusted code and
servers. See [Extensions and MCP](extensions.md) for setup and authoring.

### Settings definitions

Settings rows live in `cli/extensions/settings/definitions/`.
`setting_template.py` provides `Setting` for a preference with discrete values,
`Menu` for nested rows, and `Action` for an application action dispatched after
closing the menu. A menu's availability gates its descendants.

Declare a feature's full menu in its definition module and add it to the root
menu in `settings/extension.py`. The settings extension owns generic rendering,
navigation, and applying changes; it does not special-case individual features.
