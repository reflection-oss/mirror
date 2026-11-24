# Agent plugins

Register a reviewed plugin folder from Claude Code or Codex, or a portable folder with a root manifest:

```bash
mirror plugins add ~/src/my-plugin
mirror plugins list
mirror plugins remove my-plugin
```

Mirror reads `.claude-plugin/plugin.json`, `.codex-plugin/plugin.json`, or a
portable root `plugin.json`. The portable manifest takes precedence, followed
by Codex and Claude Code.

Registrations live in `~/.mirror/plugins` on the execution host and are shared
by desktop and CLI. Adding a folder creates a live link; removing it preserves
the source files. Review changes to the source as you would any installed code.

## Skills and commands

New local desktop and CLI conversations load registered plugin skills. On SSH,
register the folder on that host. List the available names with:

```bash
mirror plugins skills
```

Invoke a skill or Markdown command with `$plugin-name:skill-name`. Commands are
explicit invocations; start the message with `$plugin-name:command arguments`
to substitute the remaining text for `$ARGUMENTS`. Arguments stay text.
`${CLAUDE_PLUGIN_ROOT}` and `${PLUGIN_ROOT}` expand to the plugin directory in
explicitly invoked bodies. Skills take precedence over same-named commands.

Mirror reads `skills/` and `commands/`, plus manifest-declared skill directories
and command files or directories. Plugin names keep identically named skills
from different plugins separate. Existing conversations pick up the catalog
when their worker restarts.

## MCP connections

List a plugin's MCP servers, then register the ones you want:

```bash
mirror plugins connections my-plugin
mirror plugins connect my-plugin docs
mirror plugins connect my-plugin private-docs --oauth
```

Connections appear in desktop **Connections** and CLI `/connections`. Use the
usual connection controls to sign in or remove them; they have independent
settings after registration. Existing settings are preserved. Start a new
worker to use newly registered connections.

Mirror imports stdio commands and Streamable HTTP endpoints from `.mcp.json`,
manifest `mcpServers`, and portable `mcp.json`. `${CLAUDE_PLUGIN_ROOT}` and
`${PLUGIN_ROOT}` expand in commands and arguments. Environment entries such as
`"TOKEN": "${MY_TOKEN}"` and `"Authorization": "Bearer ${MY_TOKEN}"` become
credential references. Registration requires `--allow-env MY_TOKEN` for each host variable that the selected server may receive. Names are not grants; the runtime resolves values only when connecting. `--oauth`
configures browser sign-in for an HTTP endpoint.

Review a stdio server before registering it: it runs with your account's
permissions when Mirror connects. Registration itself only writes configuration.

## Hooks

Registration does not enable shell hooks. To opt in to a plugin’s `PreToolUse` commands,
see [pre-tool hooks](hooks.md#plugin-hooks).
