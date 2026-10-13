"""Single-line, default-background cards for terminal information."""

from collections.abc import Sequence

from rich import box, console, panel, table, text


def frame(
  content: console.RenderableType,
  *,
  title: str = '',
  expand: bool = True,
  width: int | None = None,
) -> panel.Panel:
  """Frame content without imposing a background or a minimum terminal width."""
  return panel.Panel(
    content,
    title=text.Text(title, style='bold'),
    title_align='left',
    box=box.ROUNDED,
    border_style='ui.muted',
    padding=(0, 1),
    expand=expand,
    width=width,
  )


def sections(
  title: str, groups: Sequence[Sequence[tuple[str, str]]]
) -> panel.Panel:
  """Align facts within a group, with quiet space between separate subjects."""
  content = table.Table.grid(padding=(0, 2), expand=True)
  content.add_column(style='ui.muted', ratio=1)
  content.add_column(ratio=3, overflow='fold')
  for index, group in enumerate(groups):
    if index:
      content.add_row('', '')
    for label, value in group:
      content.add_row(text.Text(label), text.Text(value))
  return frame(content, title=title)
