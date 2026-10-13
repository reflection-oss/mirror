"""Edit the current draft in your external editor with Ctrl+G."""

from typing import override

from mirror.cli import actions, extension, keymap


class ExternalEditorExtension(extension.Extension[None, None, None]):
  """A default-on shortcut, independent of the composer's editing mode."""

  @classmethod
  @override
  def id(cls) -> str:
    return 'core/external_editor'

  @override
  def tips(self) -> tuple[str, ...]:
    return ('Ctrl+G edits your draft prompt in an external editor',)

  @override
  def keybindings(self) -> tuple[keymap.Binding, ...]:
    return (keymap.Binding(('c-g',), actions.ComposerEditExternally()),)
