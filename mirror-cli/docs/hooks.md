# Pre-tool hooks

Mirror supports a pass/deny subset shared by [Claude Code](https://code.claude.com/docs/en/hooks) and [Codex](https://developers.openai.com/codex/hooks):
synchronous `PreToolUse` commands, not the full plugin or hook ecosystem.

Hooks belong to the agent runtime, not an attached CLI. Native workers load
`~/.mirror/hooks.json` once at startup; restart the worker to reload it.
Multiple clients do not duplicate hooks, and disconnecting a client does not
remove them. A restarted worker reloads the file.

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "^(Bash|Write|Edit)$",
        "hooks": [
          {
            "type": "command",
            "command": "python3 /absolute/path/review.py",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

- `matcher` is a case-sensitive Rust regex. Omit it, use `""`, or use `"*"` to match all tools.
  Use `Bash`, `Read`, `Write`, `Edit`, and `Grep`; other tools retain their Mirror names.
  A matcher that matches only the native lowercase name gets that name on stdin.
- `timeout` is seconds (1–600, default 600). `description` (file) and `statusMessage` (command) are accepted as metadata.
- Other events, handler types, and fields are rejected rather than silently dropping checks.

## Plugin hooks

Register the folder with `mirror plugins add`, review its scripts, then explicitly enable
its hooks by manifest name in `~/.mirror/hooks.json`:

```json
{"plugins": ["my-plugin"]}
```

`plugins` can coexist with user `hooks`. Registration alone never runs a hook. Trust applies to
live source files, not a pinned version. Remove the name to revoke trust; restart the worker
after changing configuration. A missing trusted plugin or invalid hook file prevents startup.

Both formats use `hooks/hooks.json`. A single relative file path in the manifest's `hooks`
field is also supported: Codex replaces the default; Claude Code adds to it. Inline manifests,
arrays, project settings, and Codex TOML are not imported. Plugin commands can use quoted
`"${CLAUDE_PLUGIN_ROOT}"` or `"${PLUGIN_ROOT}"` to locate bundled scripts.

## Command contract

Commands run through `/bin/sh -c` in the configured workspace directory.
JSON on stdin contains `hook_event_name`, `cwd`, `tool_name`, `tool_use_id`, and
`tool_input` (the parsed tool arguments). Arguments are data, never interpolated
into the command. File paths are passed as supplied, not normalized to absolute
paths; resolve relative paths against `cwd` when checking them. Session,
transcript and permission-mode metadata are not supplied.

| Result | Behavior |
| --- | --- |
| Exit 0, empty/plain-text stdout, `{}`, or `{"continue":true}` | Continue ordinary approval processing |
| Exit 2 | Reject using stderr; blank stderr gives `Tool call rejected.` |
| Exit 0, denial JSON below | Reject using the supplied reason |
| Other exit, invalid/unsupported JSON, timeout, or I/O failure | Fail the current operation |

```json
{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"Outside the approved scope."}}
```

For exit 0, stdout beginning with `{` after whitespace must be valid supported
JSON. Other UTF-8 stdout is discarded, not added to the conversation. Boolean
`suppressOutput` is accepted as a no-op. A `hookSpecificOutput` containing only
`{"hookEventName":"PreToolUse"}` also makes no decision.

`continue:false`, `allow`/`ask`, `updatedInput`, `additionalContext`, and other
output fields are unsupported and fail the operation. Errors describe the
supported contract without quoting potentially sensitive output. Only explicit
denial reasons enter the conversation.

For example, `/absolute/path/review.py` can reject direct edits to files named
`.env` (this is not a filesystem sandbox):

```python
import json
import pathlib
import sys

call = json.load(sys.stdin)
if call["tool_name"] in {"Write", "Edit"} and (
    pathlib.Path(call["tool_input"]["file_path"]).name == ".env"
):
    print("Editing .env is not permitted.", file=sys.stderr)
    sys.exit(2)
```

Matching commands run concurrently. All must continue; any can reject.
Continuation never grants ordinary approval. Configuration and each output
stream are limited to 64 KiB. Hook processes are terminated on cancellation.

Hooks cover model-dispatched calls, direct worker calls, nested MCP calls, and
subagent calls. Subagents inherit the parent's configured hooks. They do not
intercept arbitrary Python extension I/O or tools executed outside the worker.
Library/hosted callers supply a hook list when constructing the service or
execution context; an empty list adds no hooks. The hosted server does not
automatically load a local user's configuration.

## Trust

Only install reviewed commands. Hooks run unsandboxed on the worker host,
including when the reviewed tool runs in a sandbox. Project directories are
not searched for hook configuration. Of the parent environment, commands inherit only
`PATH`, not provider secrets or shell startup variables; this is not filesystem
or network isolation.

Rejection reasons enter conversation history and are sent to the model. Do not
include secrets. Other subprocess diagnostics are not forwarded. Unlike Codex's
fail-open default, hook failures stop the current operation.
