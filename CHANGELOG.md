# Changelog

CLI releases use `reflection-mirror-vX.Y.Z`; agent releases use `mirror-agent-vX.Y.Z`.

## [reflection-mirror-v0.1.2]

### Added

- Vercel AI Gateway as a model provider.
- `/speed` to select normal or fast processing on supported OpenAI and Anthropic
  models.
- `/ssh` to attach to a devserver, plus remote workspace setup pickers.
- Model and reasoning-effort selection from ACP clients using `mirror acp`.
- Left and right sidebars for terminal extensions.
- Custom tool-rejection reasons supplied by extensions and sent to the model.
- An explicit `bypass` tool-approval mode in launch options and settings.

### Changed

- Local workspaces and conversations now use a shared native daemon. Quitting
  the CLI detaches without stopping its workers; `mirror daemon` commands manage
  the daemon's lifecycle.
- A workspace landing screen and indexed conversation history let you browse
  without starting agents or replaying transcripts.
- Fullscreen transcripts now use a native renderer.
- Transcript detail preferences apply across workspaces.
- Workspace status shows directory paths and handles non-repository folders.
- Agent prompts include repository startup context and the local timezone.
- Compaction entries show context token counts before and after compaction.
- Anthropic sessions default to Opus 5.5.

### Fixed

- Concurrent Git worktree creation and worktree deletion handling.
- Startup with broken Jujutsu metadata.
- Repository status stuck loading after restoring tabs or resuming conversations.
- Recovery after dropped agent connections, daemon replacement, and remote server
  restarts.
- Lagging conversation updates resynchronize without blocking terminal prompts.
- MCP initialization compatibility and visibility of discovery failures.
- MCP sign-in accepts a pasted redirect URL.
- OpenAI Responses WebSocket handshakes use HTTP/1.1 to preserve upgrade headers.
- Provider failures no longer block startup or prevent choosing another model.
- Models missing from provider catalogs can be selected, with clearer warnings
  when their context-window size is unknown.
- Smoother fast mouse-wheel and trackpad scrolling.
- Down-arrow prompt history navigation takes priority over background tasks.
- The sticky user header stays hidden while the original message is visible.
- Help text and quit-shortcut hints match the configured behavior.
- Clearer database migration and model response-stream errors.
- Full conversation snapshot queues no longer crash the worker.

### Security

- Keep model API keys out of debug output.
- Guard against recursive `grep` and `rg` searches over the filesystem root,
  using Bash-aware command parsing.

### Contributors

- @ChetanTekur
- @adaminsky
- @adriansa-rai
- @akseljoonas
- @germtb
- @jnations-reflection
- @nschank
- @sndrws
- @trrwilson
- @yonigozlan

## [reflection-mirror-v0.1.1]

### Added

- `/feedback` for sharing feedback and reporting issues.
- `/fork [name]` to copy an idle local conversation and its working files into a
  new workspace.
- In-app inspection of sub-agent conversations.
- `/web-provider` to choose You.com, Tavily, or Firecrawl for search and browsing,
  with the choice saved across launches. You.com is the default.
- Display of recorded session usage.
- Support for loading extensions directly from local source directories.
- Batch extension approvals and visibility into failed or unapproved extensions
  in `/extensions`.
- Tools can require confirmation even in automatic approval mode.
- Support for additional Gemini models.
- Optional HTTP endpoints for in-memory agent traces.

### Changed

- Smoother transcript scrolling and streaming, with fewer redraws and fewer
  stalls in long conversations.
- Mouse-wheel events now scroll one row at a time.
- When scrolled up, Escape returns to the latest output before interrupting
  active work.
- Automatic terminal background detection and improved light-theme syntax colors.
- Improved highlighting for tool actions, quoted strings, and Bash commands.
- Terminal tabs show the active workspace name.
- Clearer workspace choices in `Ctrl+N`.
- More compact elapsed-time labels, single-row running summaries, and clearer
  tool progress groups.
- More consistent web-search and browsing output.
- Concurrency-safe tool calls can run in parallel.
- Persistent WebSocket connections for supported OpenAI Responses endpoints.
- macOS wheels now support macOS 14 and later.
- Expanded usage analytics and documented collection and opt-out controls.

### Fixed

- Clipboard copying in local Apple Terminal sessions.
- Activity-layout warnings triggered by scrolling.
- Name broken configuration files in error messages.
- Include failure reasons in `mirror exec` errors.
- Initialize HTTP MCP clients for each run.
- Avoid repeated launch screens and gaps around the scrolling status area.
- Limit repeated malformed tool turns using the consecutive-error budget.
- Coerce tool arguments to their declared types.
- Correct Anthropic usage accounting, prompt caching, and thinking-history replay.

### Security

- Updated TLS trust stores and certifi certificate data.

### Contributors

- @akseljoonas
- @ChetanTekur
- @drubinstein
- @germtb
- @jnations-reflection
- @m1k3s0 — reference implementation for the Apple Terminal clipboard fix.
- @nschank
- @sndrws
