# Workspace tags

- `/tag`: assign, clear, create, or edit a tag. Delete removes it immediately.
- `Ctrl+S`: cycle the current workspace’s tag, then untagged; never switch workspaces.
- One icon per workspace; optional color and custom symbols/emoji. Names stay in the picker.
- Defaults: red and blue squares. Assignment changes are silent.

## Storage

- Definitions: ordered `workspace_tags` in `~/.mirror/config.json`; `[]` disables defaults.
- Assignments: optional `WorkspacePreferences.tag_id` in the existing preferences blob; no migration.
- Rename preserves assignments; delete clears them, including closed workspaces.
- Unknown tag IDs survive loading and clear on explicit cycling.
- Jujutsu IDs survive relocation. **Git/plain-folder IDs remain path-based.**

## Integration

Use `context.workspace_controller` on the CLI event loop; do not write SQLite directly.

- Definitions: `list_workspace_tags()`, `save_workspace_tag(WorkspaceTag(id, name, color=None, icon='■'))`, `delete_workspace_tag(tag_id)`.
- Assignment: `set_workspace_tag(workspace_id, tag_id | None)`, `cycle_workspace_tag()`.
- `list_workspaces(source)` adds `tag_id` and `tag` (`{id, name, color, icon}`). Both are null when untagged; unknown IDs have `tag: null`.
- Creation: save the definition, then pass `WorkspacePreferences(tag_id=...)` to `spawn_workspace(..., preferences=...)`.

Use `cycle_workspace_tag(workspace_id=...)` to target an exact `AgentWorkspace.id`
without changing selection. Omitting `workspace_id` cycles the active tab.

### cctl / agent-tool follow-up

- [ ] Add tag inputs to cctl’s spawn schema and tag fields to its serialized output.
- [ ] Expose definition and assignment operations through explicit agent tools.
- [ ] Validate requested tag IDs before spawning; the preferences path currently accepts unknown IDs.

The controller API is available to extensions now; cctl and agent-tool wiring are not included.
