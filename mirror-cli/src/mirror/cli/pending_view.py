"""Creation state presented through the ordinary workspace composer."""

from collections.abc import Callable
from typing import override

from prompt_toolkit import buffer
from prompt_toolkit.application import current
from prompt_toolkit.layout import containers, controls, layout

from mirror.cli import approval_modes
from mirror.cli import extension, selection, state, themes, workspaces
from mirror.cli.components import component
from mirror.cli.composer import frame, pane, surface
from mirror.cli.sandbox import view as sandbox_view


def layout_for(
  workspace: workspaces.PendingWorkspace,
  get_menu: Callable[[], state.MenuState],
  navigation: containers.AnyContainer,
  draft: buffer.Buffer,
  theme_name: Callable[[], str],
  extensions: extension.ExtensionView,
) -> layout.Layout:
  """Supply creation state, not a separate composer or widget layout."""
  sandbox = (
    workspace if isinstance(workspace, workspaces.SandboxWorkspace) else None
  )

  def view_state() -> state.State:
    choices = sandbox_view.choices(sandbox) if sandbox is not None else ()
    label = 'worktree' if workspace.label == 'creating…' else workspace.label
    preferences = workspace.preferences
    definition = preferences.definition
    source = (
      sandbox.source.agent
      if sandbox is not None and sandbox.source is not None
      else None
    )
    return state.State(
      session=state.SessionState(conversation_id=''),
      lifecycle=state.LifecycleState(active=True, failure=workspace.error),
      selector=state.SelectorState(
        menu=get_menu(),
        selector=selection.Selector(
          'sandbox', 'Sandbox', choices, str, numbered=True
        )
        if choices
        else None,
      ),
      composer=workspace.draft,
      layout=state.LayoutState(
        columns=current.get_app().output.get_size().columns
      ),
      extensions=extensions.initial_states.configured(
        themes.EXTENSION_ID, {'name': theme_name()}
      ),
      activity=state.ActivityState(
        busy=workspace.error is None and not choices,
        status=f'{sandbox.phase} · Esc to cancel'
        if sandbox is not None
        else workspace.notice or f'Creating {label}…',
      ),
      status_bar=state.StatusBarState(
        source.model if source is not None else definition.model or '…',
        source.reasoning_effort
        if source is not None
        else definition.reasoning_effort or 'none',
        source.approval_mode
        if source is not None
        else approval_modes.from_protocol(preferences.approval_mode),
      ),
    )

  workspace_frame = frame.Frame(
    extensions,
    view_state,
    navigation,
    get_size=current.get_app_session().output.get_size,
  )
  prompt = surface.input_window(
    draft,
    hidden=lambda: view_state().selector.numbered,
    max_height=lambda: max(1, workspace_frame.content_size().rows // 3),
  )
  widgets = pane.components(extensions, lambda: composer.selector_max_height())
  widgets['failure'] = _CreationNotice(workspace)
  composer = pane.Pane(
    workspace_frame.content_state,
    component.ComponentHost(workspace_frame.content_state),
    component.ComponentCollection(widgets),
    prompt,
    navigation,
    fullscreen=True,
    get_size=workspace_frame.content_size,
  )
  return layout.Layout(
    workspace_frame.wrap(
      containers.HSplit([containers.Window(), composer.dock])
    ),
    focused_element=prompt,
  )


class _CreationNotice(component.Component):
  def __init__(self, workspace: workspaces.PendingWorkspace) -> None:
    self._workspace = workspace

  @override
  def render(self, state: state.State) -> containers.AnyContainer:
    del state
    workspace = self._workspace
    message = workspace.error or ''
    if workspace.error and not isinstance(
      workspace, workspaces.SandboxWorkspace
    ):
      message += ' · Enter to dismiss'
    return containers.Window(
      controls.FormattedTextControl([('class:worktree-failed', message)]),
      height=int(bool(message)),
    )
