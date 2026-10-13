# Mirror Agent

Mirror Agent runs a coding agent over a local HTTP API. Send prompts, let it
read and edit files or run commands, and collect the conversation as JSONL.
Use it for automation, evaluations, and embedding in other applications.

## Install

The standalone distribution is a native executable, not a Python package.
Given a release archive for your platform, extract it into a new directory:

```bash
mkdir mirror-agent-release
tar -xzf mirror-agent-linux-x86_64.tar.gz -C mirror-agent-release
cd mirror-agent-release
./mirror-agent --version
```

Use `mirror-agent-linux-aarch64.tar.gz` for ARM64. Keep the accompanying license
and attribution files with the executable. Public download instructions will be
added when the release channel is available.

### Requirements

- **Linux x86-64 or ARM64**, matching the release archive.
- **Model access:** a supported provider or an OpenAI-compatible endpoint.
- **A writable workspace and output directory.**

Release binaries bundle ripgrep; Python, uv, and a Rust toolchain are not
required. Install any compilers, test runners, or other tools your project needs
separately.

## First conversation

**Run inside an isolated environment. The HTTP API has no authentication, and
all tools execute without approval prompts.** Keep the default loopback bind;
do not expose the server publicly. `--workspace` chooses a working directory,
not a sandbox. Do not give the environment unrelated credentials or host access.

Start with an OpenAI-compatible Chat Completions endpoint:

```bash
# Only needed if your model endpoint requires authentication.
export OPENAI_API_KEY="your-model-api-key"

./mirror-agent serve \
  --api openai-chat-completions \
  --base-url https://model-endpoint.example/v1 \
  --model your-model-name \
  --workspace /path/to/project \
  --output ./mirror.jsonl
```

The key authenticates requests to the model, **not** requests to Mirror Agent.
Use a dedicated, narrowly scoped credential if one is required.

From another terminal in the same environment:

```bash
curl --fail-with-body -i http://127.0.0.1:4097/health

curl --fail-with-body -i http://127.0.0.1:4097/message \
  -H 'Content-Type: text/plain' \
  --data-binary 'Explain how this project is organized.'

cat mirror.jsonl
```

`/health` returns `204`. `/message` accepts a plain-text UTF-8 prompt, waits for
the turn to finish, and returns `204` with no body. Read replies and tool results
from the output file. Successive prompts share one conversation; concurrent
prompts return `409`. Prompt bodies are limited to 8 MiB; requests to `/message`
with an `Origin` header are rejected.

## Configure

Run `./mirror-agent serve --help` for all options.

| Setting | Option |
| --- | --- |
| Provider protocol, endpoint, and model | `--api`, `--base-url`, `--model` |
| Working directory | `--workspace` |
| Available tools | `--tools bash` (default); also `read,write,edit,grep,find,ls` |
| Set the system prompt | `--system-prompt` |
| Opt in to project instructions | `--include-project-instructions` |
| Opt in to the skills catalog | `--include-skills` |
| Reasoning level or token budget | `--reasoning-effort` |
| Sampling | `--temperature`, `--top-p`, `--seed` |
| Context and output token limits | `--context-window`, `--max-tokens` |
| Per-turn limits | `--max-iterations`, `--turn-timeout-seconds`, `--request-timeout-seconds` |
| Provider-specific request fields | `--extra-body` (JSON object) |
| Personal memory | `--memory-dir` (see [Memory](docs/memory.md)) |

By default, model requests have no system message. Project instructions
(`AGENTS.md` / `CLAUDE.md`, including global instructions) and skills are disabled
unless explicitly enabled with the flags above. `--system-prompt` adds instructions before any
opted-in context. The server never adds an environment section. Reasoning and
sampling support vary by provider.

## Output and lifecycle

**Starting the server truncates existing output files. There is no resume
support.** Choose a new `--output` path for each run you want to preserve.
Relative paths resolve from the launch directory, not `--workspace`; missing
parent directories are created.

| File (with `--output mirror.jsonl`) | Contents |
| --- | --- |
| `mirror.jsonl` | Committed `AgentMessage` records, one protobuf-JSON object per line |
| `mirror.model-calls.jsonl` | Model-facing messages, tools, settings, usage, and errors |

Conversation records are flushed as they are committed, not streamed token by
token. Model-call traces include `context_reset` records that re-emit the active
context; interrupted calls may have no assistant response. Model-call trace
write failures warn on stderr without stopping inference.

These files can contain source code, prompts, and tool output. Treat them as
sensitive. Model requests send context to the configured provider.

Stop with Ctrl-C or SIGTERM to cancel active work and shut down. To upgrade,
stop the server, extract the new release into a separate directory, check its
version, and start it with a fresh output path. Keep previous binaries and
outputs if you need to reproduce a run.

## Documentation

See the [documentation index](docs/README.md) for architecture, configuration,
and HTTP/AgentRPC interface guides.

## License

Mirror Agent is licensed under [Apache-2.0](LICENSE). See [NOTICE](NOTICE) for
attribution. Third-party components retain their respective licenses.
