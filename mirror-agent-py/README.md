# Mirror agent Python bindings

This is a Cargo workspace member, not a separately published Python project.
From the repository root, check it with
`cargo check --locked -p mirror-agent-python`. Building the extension uses
`cargo build --locked -p mirror-agent-python` and a compatible Python 3.12+
interpreter (`PYO3_PYTHON` can select it). The resulting cdylib requires an
embedding application's packaging; the CLI does not load it and uses the
standalone worker instead. Keep the sibling agent and protocol crates present.

This package constructs `ReactAgentService` in-process with an explicit model
endpoint and caller-supplied tool implementations. It does not execute native
tools or inspect a host workspace. `Tool.native(name, run)` retains the native
tool's schema and description while replacing only its execution. Session traffic
uses the existing `mirror.agent.v1` protobufs rather than a binding-specific protocol:

- `Session.send()` accepts an encoded `ClientMessage`.
- `Session.recv()` returns an encoded `SessionMessage`.

`Session.close()` half-closes client input while remaining output drains;
`AgentService.close()` stops the in-process service and waits for active work.

The Python `mirror` package provides the generated protobuf classes. Tool
functions receive parsed JSON arguments and return a string; raising an exception
produces a failed tool result. An RL environment can therefore provide tools
that operate directly on its sandbox without changing Mirror's normal tools.

`ScriptedModel` runs encoded assistant messages without an HTTP model endpoint.
Its `request_deltas()` method returns each model request's appended message
suffix so callers can reconstruct long request histories with linear memory.

Pass the desired tools to `AgentService` and use `AgentConfig`'s
`custom_system_prompt` to replace the default prompt, including `''` for no
instructions. The native worker exposes equivalent `tools` and `system_prompt`
construction options. Bash advertises only `command` and defaults to 600 seconds.

`tool_result_max_chars` controls tool-output retention during compaction and
keeps the normal 2,000-character default when omitted.
