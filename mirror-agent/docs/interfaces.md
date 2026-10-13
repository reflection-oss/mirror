# Interfaces

Mirror Agent exposes two interfaces: a simple HTTP server for headless runs and
AgentRPC for typed, bidirectional worker integrations. They are separate
transports; AgentRPC is not served by the HTTP port.

## Standalone HTTP

Start with the [server quickstart](../README.md). The default address
is `http://127.0.0.1:4097`. There is no inbound authentication. Run in an isolated
environment and keep the listener private; tools execute with automatic approval.

| Request | Success | Meaning |
| --- | --- | --- |
| `GET /health` | `204`, empty body | HTTP server is reachable; not a model-provider check |
| `POST /message` | `204`, empty body | The submitted turn finished |

`POST /message` requires `Content-Type: text/plain`. The entire UTF-8 body is one
prompt, limited to 8 MiB. It waits for the turn; it does not stream the response.
To submit a file without changing its contents:

```bash
curl --fail-with-body http://127.0.0.1:4097/message \
  -H 'Content-Type: text/plain' --data-binary @prompt.txt
```

Errors return a non-success status and diagnostic body:

| Status | Typical cause |
| --- | --- |
| `400` | Empty prompt |
| `403` | An `Origin` header was supplied |
| `409` | A turn is already running, or the turn was interrupted |
| `413` | Body exceeds the size limit |
| `415` | Content type is not `text/plain` |
| `502` | Agent turn failed |
| `503` | Agent service unavailable |
| `504` | Turn deadline exceeded |

There is no HTTP idempotency key. After a timeout or lost connection, inspect
the output before resubmitting: tools may already have changed files. The server
owns one conversation, rejects overlapping prompts, and has no resume endpoint.
See [routing](../src/http/server.rs) and
[turn admission](../src/http/prompt_runner.rs).

## JSONL output

`--output` selects the conversation journal. Each committed `AgentMessage` is
written as one protobuf-JSON object per line. These are durable records, not
individual streaming tokens. Use the message schema in
[`agent.proto`](../../protocols/mirror_agent/protos/agent.proto) when parsing them.
A tool result's entry carries, beside its `ToolMessage`, a `metadata` record
(`MetadataRecord`) with the workspace-relative paths the tool touched and, for
a shell, the command's exit code; consumers read those fields rather than
parsing the text.

A sibling file with the extension `.model-calls.jsonl` records model-facing
messages, settings, tools, usage, and errors. Its `context_reset` entries re-emit
the active prompt and history after a reset. It is a diagnostic format, not the
AgentRPC wire protocol.

**Both files are truncated at startup.** Relative output paths are based on the
launch directory, not `--workspace`. Keep a separate output path for each run;
logs can contain sensitive repository and conversation data.

## AgentRPC

The authoritative schema is
[`mirror.agent.v1.AgentRPC`](../../protocols/mirror_agent/protos/agent.proto).

| RPC | Purpose |
| --- | --- |
| `Session` | Bidirectional stream of `ClientMessage` and `SessionMessage` |
| `GetMessages` | Retrieve stored conversation records |
| `CallTool` | Invoke a session tool with normal approval rules, without a model turn |

Every `ClientMessage` carries `session_id`, `message_id`, and one payload.
Payloads include user messages, interruption, task cancellation, approval replies,
inference controls, and host-tool results. Treat message IDs as opaque unique
request identities; do not reuse one for a different request.

The server distinguishes:

- **Input acknowledgement:** `InputAck` / `InputNack` identifies accepted or
  rejected input, not completed model work.
- **Committed history:** `agent_message` carries conversation records.
  `AgentMessage.sequence` is authoritative; `event_id` is deprecated.
- **Transient updates:** `runtime_delta` carries streamed text, reasoning, tool
  previews, usage, task state, and turn completion or errors.
- **Approval and host tools:** `hitl_request` asks the client for a decision;
  `host_tool_call` asks it to execute a client-owned tool and return a result.
- **Replay:** `ReplayDone` marks completion of history replay.

A supervising client starts a worker and waits for `MIRROR_RPC_ADDRESS=...` on stdout
before connecting. Startup may also emit `MIRROR_SESSION_MESSAGE=...` records;
stdout is a control channel, not arbitrary application output. See
[worker startup](../src/main.rs),
[RPC implementation](../src/rpc/agent_rpc.rs).

The worker runs in the workspace the client selected (`MIRROR_WORKSPACE`, or its
working directory). With `"include_environment": true` in its configuration, it
describes that workspace in an `# Environment` prompt section: working
directory, Git status, platform, OS version, shell and date. Without it, the
prompt has no such section.

A `"sandbox"` object in the configuration runs the workspace tools (`read`,
`bash`, `write`, `edit`, `grep`, `glob`, `ls`), including a subagent's, in a
[microsandbox](../../mirror-sandbox/providers/microsandbox/README.md) VM
through `sandboxd`. The VM mounts the workspace at the same path, so tool paths
mean the same files on both sides, and the environment section describes the
VM rather than this machine. Other tools still run on this machine.

```json
"sandbox": {
  "id": "1b4e28ba-2fa1-11d2-883f-0016d3cca427",
  "image": "debian:bookworm-slim",
  "sandboxd": "/path/to/linux/mirror-sandboxd",
  "egress": ["api.github.com", "*.githubusercontent.com"],
  "cpus": 2,
  "memory_mib": 2048
}
```

`id` names the VM: a worker restarted with the same `id` reattaches to it, and
the VM outlives the worker until deleted. `sandboxd` is a Linux build for this
machine's architecture. The VM may reach only the `egress` hosts, over HTTPS.
A sandboxed conversation needs a UUID conversation ID.

Local workers use loopback transport, not a public authenticated service. Remote
clients require a separately configured gateway with its own access controls;
exposing a local worker listener is not a substitute.
