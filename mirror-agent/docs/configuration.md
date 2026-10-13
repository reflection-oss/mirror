# Configuration

Configure the standalone HTTP server with `mirror-agent serve` arguments.
Run `mirror-agent serve --help` for all options; see the
[quickstart](../README.md) for a complete launch and request example.

## Model access

`--model` is required. `--api` selects the protocol and defaults to
`openai-chat-completions`; `--base-url` selects a custom endpoint.
Other protocols include `openai-responses`, `anthropic-messages`, and
`openai-codex-responses`.

When set, `OPENAI_API_KEY` is passed as an explicit model credential. Provider
implementations can otherwise resolve their own credentials; see
[provider construction](../src/providers/model_provider.rs). Keep only the
credential intended for the selected endpoint in the server environment.
This environment-variable input is a compatibility interface: use a dedicated,
narrowly scoped credential, never host-wide access.

Codex plan access uses an existing Codex CLI login, created with
`codex login --device-auth`. The default file is `~/.codex/auth.json`;
`MIRROR_CODEX_AUTH_PATH` overrides it. API-key login is not equivalent to
ChatGPT plan authentication.

Credentials authenticate outgoing model requests, not incoming HTTP requests.
The server has no inbound authentication. Keep it isolated and on loopback.

## Workspace, instructions, and tools

By default, model requests have no system message. Eval launches should omit
`--system-prompt`, `--include-project-instructions`, and `--include-skills`.
No flag adds an environment section, so the prompt never describes the host.

`--workspace` chooses the working directory, not a sandbox. With
`--include-project-instructions`, project instructions are loaded from `AGENTS.md` or `CLAUDE.md` along the repository ancestor chain;
`AGENTS.md` takes precedence within a directory. Outside a repository, only the
working directory is searched. Global instructions can live in
`~/.mirror/AGENTS.md`. When a local tool later touches a file in a directory
off that chain, that directory's `AGENTS.md` or `CLAUDE.md` joins the
conversation once, after the tool's result.

`--include-skills` adds the available skills catalog to the system prompt.
Both sources are opt-in, including their global locations. `--system-prompt`
adds explicit instructions before either opted-in source.
These server defaults do not change interactive CLI instruction loading.

`--tools` selects a comma-separated list from `bash`, `read`, `write`, `edit`,
`grep`, `find` and `ls`; the default is `bash` alone, the tool set models are
trained with. The HTTP server executes tools without
approval prompts. Limiting tools does not create filesystem or network isolation.

`--memory-dir` gives the agent the personal memory stored under that directory,
registers `search_memory` and `save_memory`, and loads
`~/.mirror/PINNED_MEMORIES.md` into the system prompt; see [Memory](memory.md).
It needs a local workspace, and the store's contents reach the model endpoint.
Without it the server has no memory, which is what evaluations expect.

`--remote <send-command-url>` runs workspace tools remotely. The endpoint needs
`/bin/bash`, Python 3, and `rg` (for glob/grep). Detached Bash is unsupported.
Host credentials and project context are not forwarded. The endpoint must provide
isolation, disk limits, and temporary-file cleanup: cancellation may leave commands
running or temporary files behind. Mutations are never replayed. Remote file
snapshots, directory listings, and search streams are limited to 64 MiB; edits
reject files changed since their snapshot.

## Inference and limits

| Options                              | Purpose                                                            |
| ------------------------------------ | ------------------------------------------------------------------ |
| `--reasoning-effort`                 | Named level or non-negative integer budget                         |
| `--temperature`, `--top-p`, `--seed` | Provider-dependent sampling controls                               |
| `--extra-body`                       | JSON object merged into model requests                             |
| `--context-window`, `--max-tokens`   | Context and output-token budgets                                   |
| `--compaction-threshold`             | When to summarize history                                          |
| `--max-iterations`                   | Maximum model/tool iterations per turn                             |
| `--turn-timeout-seconds`             | Turn deadline                                                      |
| `--request-timeout-seconds`          | Limit on one model request, including generation; unset by default |

Sampling and reasoning support depend on the provider. Context-window must
exceed max-tokens; limits such as max-iterations must be positive. See
[startup validation](../src/http/cli.rs) for the exact contract.

## Output

`--output` defaults to `mirror.jsonl`; a sibling `.model-calls.jsonl` file
records model-facing diagnostics. Relative paths use the launch directory,
not `--workspace`. **Both files are truncated at startup.** Use a fresh path
for every run you want to retain. See [Interfaces](interfaces.md) for formats.
