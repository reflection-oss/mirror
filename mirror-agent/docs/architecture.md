# Architecture

Mirror Agent is a Rust agent runtime with HTTP and AgentRPC adapters. It owns
model calls, tool execution, context compaction, and conversation records.
Clients submit inputs and consume events; the runtime does not depend on their
UI or application framework.

```text
HTTP requests                         AgentRPC messages
      │                                      │
      ▼                                      ▼
PromptRunner                             AgentRpc
      └─────────────────┬────────────────────┘
                        ▼
                 ReactAgentService
                        │
                        ▼
                    ReactAgent
                 ┌──────┼────────┐
                 ▼      ▼        ▼
             Provider  Tools    Store
```

Each service owns one conversation and its execution lifecycle. The adapters
translate external requests into service operations; they do not implement the
model/tool loop themselves.

## Source map

| Component                                                    | Owns                                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------------------ |
| [Agent service](../src/react_agent_service.rs)               | Conversation lifecycle, controls, approvals, and event delivery          |
| [Agent loop](../src/react/agent.rs)                          | Model/tool iteration and turn completion                                 |
| [Providers](../src/providers/model_provider.rs)              | Protocol selection and model clients                                     |
| [Tools](../src/tools/mod.rs)                                 | Native file, shell, search, and other tools                              |
| [Compaction](../src/compaction/compactor.rs)                 | Summarization and context budgets                                        |
| [System prompt](../src/system_prompt/build_system_prompt.rs) | Environment, project instructions, and skills                            |
| [Store](../src/store.rs)                                     | Durable conversation storage contract                                    |
| [Context hooks](../src/react/context_hooks.rs)               | Records that follow what the conversation commits, and the turn-end hook |
| [Workspace](../src/workspace.rs)                             | Repository identity and path facts, supplied by the runtime              |
| [Memory](../src/memory/mod.rs)                               | The user's personal memory; see [Memory](memory.md)                      |
| [HTTP](../src/http/cli.rs)                                   | Standalone server configuration and startup                              |
| [AgentRPC](../src/rpc/agent_rpc.rs)                          | Typed streaming requests, controls, and events                           |

## A turn

1. The service accepts a user message for its conversation.
2. The agent assembles context and calls the configured model provider.
3. Requested tools execute through the tool executor, subject to the service's
   approval mode. Results return to the model for the next iteration.
4. The loop finishes with a response, reaches a limit, fails, or is interrupted.
5. Committed records go to the store; transient previews and runtime events
   are delivered separately.

After the context commits the user's messages or a tool batch's results, its
hooks may add records that follow them in the same append: skills the
user mentioned, memories that apply. When a turn ends cleanly, the service
tells the hooks, handing them the session's tools.

Host tools let an integrating application execute tools outside the runtime.
AgentRPC emits a `host_tool_call`; the client returns `host_tool_result`. Native
tools execute in the worker. See [Interfaces](interfaces.md) for the protocol.

Inference controls can change model settings between requests. Updating them
does not alter a model request already in flight.

## Persistence and lifetime

The runtime depends on the `Store` contract, not a specific database. Available
adapters include [SQLite](../src/local/sqlite_store.rs),
[in-memory storage](../src/react/in_memory_store.rs), and
[JSONL](../src/http/jsonl_store.rs).

The local worker supports durable SQLite conversations. The standalone HTTP
server instead creates one conversation per process and truncates its JSONL
outputs at startup; it has no resume endpoint. Store selection and process
lifecycle belong to the embedding application or adapter.

## Trust boundaries

A worker process is not a security sandbox. Native tools have its permissions.
A `sandbox` configuration moves the workspace tools into a VM; the worker, its
model credentials and its other tools stay on this machine.
The standalone HTTP adapter has no inbound authentication and uses automatic
tool approval. Run it in an isolated environment with a private listener; do not
supply unrelated credentials or host access. AgentRPC integrations must provide
their own access controls rather than exposing a local listener publicly.

## Local and remote tools

`DirectToolExecutor` calls `Tool::run`; `RemoteToolExecutor` validates
`supports_remote()` up front and calls `Tool::run_remote`, with no local fallback.
Subagents inherit the remote executor, never host project context.

`SandboxRoutedTool` wraps one workspace tool and sends its calls to a
`SandboxToolExecutor`, which runs them in a sandbox through `sandboxd`. Hosted
Mirror routes every session's workspace tools this way; a local worker does
when its configuration has a `sandbox` (`LocalSandbox`). Other tools keep
running in the host process.
