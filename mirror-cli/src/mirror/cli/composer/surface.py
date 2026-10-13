"""Background and padding around the composer input."""

from collections.abc import Callable

from prompt_toolkit import filters
from prompt_toolkit.layout import containers

from mirror.cli import themes


def wrap_input(
  input_window: containers.Window,
  menu_visible: Callable[[], bool],
  status_bar: containers.AnyContainer,
  *,
  theme_name: Callable[[], str] = lambda: 'dark',
) -> containers.HSplit:
  """Frame the input and status bar without duplicating an open menu border."""
  show_menu = filters.Condition(menu_visible)
  input_surface = containers.HSplit(
    [
      containers.ConditionalContainer(
        containers.Window(height=1), filter=~show_menu
      ),
      containers.ConditionalContainer(
        containers.VSplit([
          containers.Window(width=1),
          input_window,
          containers.Window(width=1),
        ]),
        # VSplit's padding otherwise expands when its input has zero height.
        filter=filters.Condition(
          lambda: input_window.preferred_height(1, 1).max > 0
        ),
      ),
      containers.Window(height=1),
      status_bar,
    ],
    style='class:composer',
  )
  return containers.HSplit([
    input_surface,
    containers.Window(
      height=1,
      # Half-blocks only soften an explicit fill, not the terminal background.
      char=lambda: (
        '▀'
        if themes.rich_theme(theme_name()).styles['ui.input'].bgcolor
        else ' '
      ),
      style='class:composer-bottom-divider',
    ),
  ])
