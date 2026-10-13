---
name: mirror-extensions
description: >
  Build or modify trusted Python extensions for the Mirror CLI terminal UI:
  widgets, slash commands, keybindings, tools, and skills. Explains the extension
  API and the create, load, reload, and unload workflow.
---

# Mirror CLI extensions

Mirror can host Python UI extensions, including live widgets such as a clock.
These extend the Mirror terminal, not a browser or VS Code. Confirm the target
when the user's intended platform is unclear.

## Authoring workflow

The `user-extensions__extensions` tool manages
`~/.mirror/extensions/<name>/extension.py`:

1. Call it with `{"action": "create", "name": "clock"}` for a working scaffold.
2. Edit the generated file. Keep its extension ID equal to its folder name.
3. Call `{"action": "load", "name": "clock"}` to enable it.
4. After further edits, call `{"action": "reload", "extension_id": "clock"}`.

`list` reports loaded user extensions and workspace faults. `unload` disables
an extension without deleting it. Load/unload persist the user's choice;
`/extensions` provides the interactive toggle. These operations execute trusted
code locally, without a sandbox; do not enable unreviewed external code.

Project extensions can also live at `.mirror/extensions/<name>/extension.py`.
They are discovered from the nearest repository root through the launch
directory, or just the launch directory outside a repository. The authoring tool
manages the user directory only; project extensions use `/extensions`.

## Extension API

Read [the API](../../extension.py) for signatures and
[the scaffold](extension.py) for a complete widget example.

- Import `extension`, `actions`, and `state` from `mirror.cli`.
- Subclass `Extension[StateT, EffectT, ContextT]` with a no-argument constructor
  and an `id()` matching its folder name. The module docstring describes it in `/extensions`.
- Extensions execute on the CLI machine. Use `context.workspace.run(command)`
  for workspace I/O and `context.workspace.shell.open(argv)` for process streams;
  no location-support declarations are needed.
- `initial_state()` creates per-workspace state. Keep `reduce()` and `render()`
  free of I/O. Use `handle_effect()` for async work and `self.dispatch()` for
  routed updates. Preserve task cancellation.
- `on_load()` / `on_unload()` own shared resources. `on_config()` receives
  configuration changes. `on_foreground()` / `on_background()` observe tab
  visibility, not extension lifetime.
- `slash_commands()`, `keybindings()`, and `tools(context)` expose interactions.
  Tool names are namespaced by the extension ID.
- Leave `tips()` out unless the user explicitly asks for a tip. Most extensions
  should omit it; tips are intended for broadly shared extensions, not personal
  or one-off widgets. When asked, return short, plain single-line strings without
  a `Tip:` prefix. Contributions are read on load/reload and shown only where
  the extension is enabled, healthy, and supported.
- `skills()` returns paths to `SKILL.md` files. Enabled extensions contribute
  these to local workspaces' skill catalogs and `$skill` completion. Include a
  `name` and `description` in YAML frontmatter, and resolve paths relative to
  `__file__`, not the user's working directory.

Validate behavior through the extension's public interactions and inspect load
faults before claiming that a widget works. Do not put authentication secrets in
skill files, tool descriptions, UI state, or transcripts.
