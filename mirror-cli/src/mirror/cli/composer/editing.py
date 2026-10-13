"""Keep the requested native editor mode scoped to the prompt."""

import dataclasses

from prompt_toolkit import application, buffer, cursor_shapes, enums, styles


@dataclasses.dataclass(frozen=True)
class Editor:
  """An extension's native prompt editor and appearance."""

  mode: enums.EditingMode
  style: styles.BaseStyle | None = None
  cursor: cursor_shapes.CursorShapeConfig | None = None


def sync(
  app: application.Application[str | None],
  composer: buffer.Buffer,
  mode: enums.EditingMode,
  *,
  suspended: bool,
) -> None:
  """Keep menus and secret inputs non-modal; resume prompts in Insert mode."""
  desired = enums.EditingMode.EMACS if suspended else mode
  if app.editing_mode != desired:
    app.editing_mode = desired
    app.vi_state.reset()
    composer.exit_selection()
