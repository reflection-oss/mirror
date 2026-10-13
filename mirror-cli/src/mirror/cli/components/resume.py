"""Cross-directory resume confirmation; choices use the generic selector."""

from typing import override
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout import containers, controls, dimension
from mirror import workspace
from mirror.cli import state
from mirror.cli.components import component, menu


def scope_tabs(
  active: workspace.SessionScope, width: int
) -> StyleAndTextTuples:
  """Show scope in keyboard cycling order, with a color-independent active cue."""
  scopes = tuple(workspace.SessionScope) if width >= 24 else (active,)
  fragments: StyleAndTextTuples = []
  for scope in scopes:
    fragments.append((
      'class:resume-scope-active' if scope is active else 'class:resume-scope',
      f'[{scope.value}] ' if scope is active else f' {scope.value}  ',
    ))
  return menu.fit_line(fragments, width)


class Resume(component.Component):
  @override
  def render(self, state: state.State) -> containers.AnyContainer:
    info = state.resume.confirmation
    if info is None:
      return containers.Window(height=0)
    panel = menu.ConfirmPanel(
      title='Resume from another directory',
      lines=(
        f'{info.title or "(untitled)"}  {info.id[:8]} was started in:',
        info.cwd,
        f'Resume it from {state.workspace}?',
      ),
      hint='y resume · n/Esc cancel',
    )
    return containers.Window(
      controls.FormattedTextControl(panel.render()),
      height=dimension.Dimension.exact(panel.height),
    )
