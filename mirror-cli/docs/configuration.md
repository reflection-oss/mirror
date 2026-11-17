# Configuration

Remote gateways are disabled unless supplied with `--remote-endpoint` or
`MIRROR_REMOTE_ENDPOINT`. Use your service operator's endpoint. An empty
environment value disables the remote gateway.

Use `mirror --help` for launch options. Configuration belongs to the terminal
client; each workspace selects its own model and approval settings.

## CLI providers

Select a protocol explicitly when setting up a provider; merely exporting an
API key does not select it.

| Provider                       | `--api`                   | Authentication                                 |
| ------------------------------ | ------------------------- | ---------------------------------------------- |
| Reflection                     | `reflection`              | `/login`, a saved key, or `REFLECTION_API_KEY` |
| OpenRouter                     | `openrouter`              | `OPENROUTER_API_KEY`                           |
| OpenAI Responses               | `openai-responses`        | `OPENAI_API_KEY`                               |
| Anthropic Messages             | `anthropic-messages`      | `ANTHROPIC_API_KEY`                            |
| Google Gemini Developer API    | `gemini`                  | `GEMINI_API_KEY` (paid API access)             |
| Vercel AI Gateway              | `vercel`                  | `AI_GATEWAY_API_KEY`                           |
| ChatGPT plan via Codex         | `openai-codex-responses`  | Existing Codex CLI login                       |
| Custom Chat Completions server | `openai-chat-completions` | Endpoint-specific; `--api-key` if required     |

`vercel` sends Chat Completions requests to `https://ai-gateway.vercel.sh/v1`
and defaults to `reflection/beam`; gateway model IDs take the form
`provider/model`. Like `gemini` and `reflection`, it accepts no custom
`--base-url`; use `openai-chat-completions` for another endpoint.

For ChatGPT plan access, log in with the Codex CLI using
`codex login --device-auth`. Mirror reads `~/.codex/auth.json` by default;
`MIRROR_CODEX_AUTH_PATH` can select another credential file. API-key login is
not a substitute for ChatGPT plan authentication.

For a custom endpoint:

```bash
mirror --api openai-chat-completions \
  --base-url http://127.0.0.1:8000/v1 --model your-model-name
```

Without an explicit protocol, saved selection, or model configuration, the CLI
defaults to Reflection, or Chat Completions when a custom endpoint is supplied.

## Model picker

`/model` selects a model; `/effort` selects an effort level. Optional
`~/.mirror/models.json` entries supply model IDs, labels, and routing:

```json
[
  {
    "id": "anthropic/claude-opus-4.8",
    "label": "Opus via OpenRouter",
    "api": "openai-chat-completions"
  },
  {
    "id": "local-model",
    "label": "Local server",
    "api": "openai-chat-completions",
    "base_url": "http://127.0.0.1:8000/v1"
  }
]
```

`id` and `label` are required; `description`, `api`, and `base_url` are optional.
Use `MIRROR_MODELS_CONFIG` to select a different file. Do not put API keys in
model entries or endpoint URLs.

Explicit `--api` and `--base-url` flags override a matching model entry. Without
`--model` or `--api`, startup prefers the workspace's saved model, then the
complete selection in `last_model.json`, including its protocol and endpoint.
That selection need not remain in `models.json`. If no saved selection can be
used, startup falls back to the configured list or the provider's default.
See [model loading](../src/mirror/cli/models.py) and
[startup resolution](../src/mirror/cli/main.py).

## Project instructions and approvals

Put project guidance in `AGENTS.md` or `CLAUDE.md`. Mirror loads instructions
from the repository root through the working directory (only that directory
outside a repository), preferring `AGENTS.md` when both files exist. Global
instructions can live in `~/.mirror/AGENTS.md`. Instructions and available skills
become part of the system prompt.

Manual tool approval is the default unless `auto` was previously saved in the
configuration. Use `mirror --approval-mode approve` or
`mirror --approval-mode auto` to select the mode at launch; explicit mode choices
are remembered across launches. Use `/settings` → **CLI mode** to change the
mode while Mirror is running.
Approval is not filesystem or network isolation.

- Remember: “always allow commands starting with `cat`” → any literal arguments.
  `cat`/`head`/`tail`/`wc` offer command-wide rules; others keep their full prefix.
  Other tools, remote calls and unsupported shell syntax offer exact calls only.
- Match: whole argument words, not text patterns; no shell operators or expansions.
  `cat` does not match `catfish`, `/bin/cat`, or `sudo cat`.
- Scope: local workspace and override, or remote endpoint/transport/session.
- Limits: no file-access restriction or executable identity pinning.
- Storage: `~/.mirror/tool_approvals.json`; exact-call arguments are hashed,
  but command prefixes are stored in plaintext without redaction. Do not remember
  prefixes containing secrets.
- Revoke: stop Mirror → edit file → restart.

## Keybindings

- Approval: Ctrl+C / Escape stop the turn; **Reject** rejects only that call.
- Ctrl+C: dismiss modal → clear draft → interrupt work → exit if idle.
- Escape: dismiss UI or interrupt work. With the `vim` extension, Escape returns
  to Normal mode instead of interrupting; use Ctrl+C, which clears an unsent
  draft first.
- Ctrl+D: quit immediately.
- Ctrl+N: **1** Working directory · **2** New working directory ·
  **3** Worktree · **4** Local sandbox.
  - New working directory: complete `/folder <path>`.
  - Up/Down + Enter: select · Escape: cancel.
  - Working directory/Worktree choice persists.

Create `~/.mirror/keybindings.json` and restart the CLI. Each entry replaces
one action's shortcuts; omitted actions keep their defaults and `[]` disables
an action's bindings. To move cancel-or-quit to Ctrl+G:

```json
[{ "action": { "type": "CancelOrQuitRequested" }, "keys": [["c-g"]] }]
```

To restore immediate quitting on Ctrl+C while retaining Ctrl+D:

```json
[{ "action": { "type": "QuitRequested" }, "keys": [["c-c"], ["c-d"]] }]
```

Or bind interrupt to Ctrl+C:

```json
[{ "action": { "type": "InterruptRequested" }, "keys": [["c-c"]] }]
```

Escape still dismisses menus. Invalid configuration is reported and ignored.
See [action types](../src/mirror/cli/actions.py) and
[keybinding resolution](../src/mirror/cli/keymap.py).

## Sandbox workspaces

`/sandbox` takes its VM from the `sandbox` object in `~/.mirror/config.json`.
Every field is optional:

```json
{
  "sandbox": {
    "egress": ["github.com", "api.github.com", "pypi.org"],
    "cpus": 2,
    "memory_mib": 2048
  }
}
```

- `image` must be an anonymously accessible OCI reference pinned with
  `@sha256:<64 hex digits>`. Omit it to use the pinned default with Git, curl and
  common build tools. Custom images must provide `python3`, `git`, and `rg`.
  Commands must run as root, which lets `sandboxd` see edits made on this machine
  at once.
- `egress` replaces the default allowlist of GitHub, PyPI and npm hosts. Each
  entry is a host name, or `*.domain` for its subdomains, reached over HTTPS
  only.
- `cpus` is from 1 to 255, and `memory_mib` at least 256.
- The clone's `origin` is the current repository's without credentials: no
  query string, and no user name or password in an HTTPS URL.
- Settings apply to sandboxes created afterwards; a sandbox keeps the VM it was
  created with.

`MIRROR_MSB` names the `msb` to use instead of the first on `PATH`.
`MIRROR_SANDBOXD` names a Linux `sandboxd` to mount instead of the one in the
installed wheel, which a source checkout needs.

## Memory

Mirror can remember your preferences, facts about your repositories, and
procedures that worked. Memory is off until you turn it on. The files are stored
on this machine; relevant memories are included in context sent to the selected
model provider.

`/settings` → **Memory** opens a submenu with the on/off switch and
**Import memories**. Changing the switch applies to future local sessions and
restarts the current session's worker. Finish or interrupt an active turn first.
Press Esc to return to Settings.
The row is absent in remote sessions or when the memory extension is disabled.
`/memory` or `/memory status` shows its status. The command belongs to the
bundled memory extension, and the switch is that extension's setting in
`~/.mirror/config.json`:

```json
{ "extensions": { "builtin/memory": { "config": { "enabled": true } } } }
```

The store is `~/.mirror/memory`. Memories that
should always be in front of the model live as bullets in
`~/.mirror/PINNED_MEMORIES.md`, which the agent's `save_memory` tool writes and
you may edit by hand.

**Import memories** (or `/memory import`) brings in the memories Claude Code
and Codex keep on this machine, under your own name. With `/memory import`,
pass a path, `claude-code=<dir>` or
`codex=<dir>` to choose a source instead of the default locations.
Remote sessions have no memory. See the agent's
[memory guide](../../mirror-agent/docs/memory.md) for what is stored and how
it reaches the model.
