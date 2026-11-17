"""The shared workspace composer, status bar, widgets, and bounded dock."""

from collections.abc import Callable

from prompt_toolkit import data_structures, filters
from prompt_toolkit.application import current
from prompt_toolkit.layout import (
  containers,
  controls,
  dimension,
)

from mirror.cli import actions, extension, retained, state, themes
from mirror.cli.components import (
  activity,
  completion,
  component,
  failure,
  layout,
  pending,
  queued,
  resume,
  rewind,
  secret_prompt,
  selector,
  status_bar,
  tasks,
)
from mirror.cli.composer import surface


def components(
  extensions: extension.ExtensionView,
  selector_max_height: Callable[[], int] | None = None,
  *,
  skip_first_queued: Callable[[], bool] = lambda: False,
  shortcut_hint: Callable[[actions.Action | None], str] = lambda _: '',
) -> dict[str, component.Component]:
  """Use the same components before and after the workspace agent is ready."""
  return {
    'activity': activity.Activity(),
    'completion': completion.Completion(shortcut_hint),
    'failure': failure.Failure(),
    'pending': pending.Pending(shortcut_hint),
    'queued': queued.Queued(skip_first=skip_first_queued),
    'resume': resume.Resume(),
    'selector': selector.Selector(get_max_height=selector_max_height),
    'secret': secret_prompt.SecretPrompt(),
    'rewind_banner': rewind.RewindBanner(),
    'rewind_picker': rewind.RewindPicker(),
    'tasks': tasks.Tasks(),
    'status_bar': status_bar.StatusBar(
      extensions.component(extension.RenderRegion.STATUS_LEFT),
      extensions.component(extension.RenderRegion.STATUS_RIGHT),
    ),
    'extensions_below_activity': extensions.component(
      extension.RenderRegion.WIDGET_BELOW_ACTIVITY
    ),
    'extensions_above_prompt': extensions.component(
      extension.RenderRegion.WIDGET_ABOVE_PROMPT
    ),
    'extensions_below_prompt': extensions.component(
      extension.RenderRegion.WIDGET_BELOW_PROMPT
    ),
  }


class Pane:
  """Compose workspace UI independently of agent execution and terminal ownership."""

  def __init__(
    self,
    get_state: Callable[[], state.State],
    host: component.ComponentHost,
    components: component.ComponentCollection,
    input_window: containers.Window,
    navigation: containers.AnyContainer,
    *,
    fullscreen: bool,
    above_input: containers.Container | None = None,
    secret_active: Callable[[], bool] = lambda: False,
    show_activity: Callable[[], bool] = lambda: True,
    queued_input_pending: Callable[[], bool] | None = None,
    get_size: Callable[[], data_structures.Size] = lambda: (
      current.get_app().output.get_size()
    ),
  ) -> None:
    self._get_state = get_state
    self._get_size = get_size
    self._paint_components = tuple(
      components[name]
      for name in (
        'completion',
        'selector',
        'status_bar',
        'extensions_below_activity',
        'extensions_above_prompt',
        'extensions_below_prompt',
      )
    )
    self._paint_keys = layout.FrameCache[tuple[object, ...]]()
    above_input = above_input or containers.Window(height=0)
    self._above_input = above_input
    queued_input_pending = queued_input_pending or (
      lambda: bool(get_state().turn.pending_instructions)
    )
    pending_window = containers.ConditionalContainer(
      host.mount(components['pending']),
      filter=filters.Condition(lambda: not fullscreen),
    )
    status_bar = containers.to_container(host.mount(components['status_bar']))
    footer = containers.ConditionalContainer(
      layout.Rows(
        [status_bar],
        height=lambda: dimension.Dimension.exact(
          status_bar.preferred_height(get_state().layout.columns, 1).preferred
        ),
      ),
      filter=filters.Condition(lambda: not secret_active()),
    )
    setup_footer = containers.ConditionalContainer(
      containers.Window(
        controls.FormattedTextControl([
          ('class:ui.accent bold', ' Enter'),
          ('class:ui.muted', ' to connect  ·  '),
          ('bold', 'Esc'),
          ('class:ui.muted', ' to choose another model  ·  '),
          ('bold', 'Ctrl+C'),
          ('class:ui.muted', ' to cancel'),
        ]),
        height=1,
      ),
      filter=filters.Condition(secret_active),
    )
    composer_visible = filters.Condition(
      lambda: (
        get_state().approval.request is None
        or get_state().selector.active
        or get_state().secret.active
      )
    )
    composer: containers.Container = containers.ConditionalContainer(
      layout.Rows([
        host.mount(components['completion']),
        surface.wrap_input(
          input_window,
          lambda: (
            get_state().selector.active or completion.visible(get_state())
          ),
          layout.Rows([footer, setup_footer]),
          theme_name=lambda: themes.current(
            get_state().extensions, get_state().extension_themes
          ),
          bash_mode=lambda: (
            not secret_active()
            and not get_state().secret.active
            and not get_state().selector.active
            and get_state().composer.text.lstrip().startswith('!')
          ),
        ),
      ]),
      filter=composer_visible,
    )
    if fullscreen:
      composer = retained.Layer(composer, self.paint_key)
    activity_window = containers.ConditionalContainer(
      host.mount(components['activity']),
      filter=filters.Condition(lambda: not secret_active() and show_activity()),
    )
    above_prompt = containers.to_container(
      host.mount(components['extensions_above_prompt'])
    )
    below_prompt = containers.to_container(
      host.mount(components['extensions_below_prompt'])
    )
    if fullscreen:
      above_prompt = _bounded_widget(above_prompt, get_size)
      below_prompt = _bounded_widget(below_prompt, get_size)

    def occupied(child: containers.Container) -> bool:
      return bool(
        child.preferred_height(get_state().layout.columns, 1).preferred
      )

    queued_window = containers.to_container(host.mount(components['queued']))
    if fullscreen:
      queued_window = retained.Layer(queued_window, animated=False)
    self._activity_prefix = layout.Rows([
      pending_window,
      containers.Window(
        height=lambda: int(
          bool(
            pending_window.filter()
            and pending.rows(get_state())
            and queued_input_pending()
          )
        )
      ),
      queued_window,
      containers.Window(
        height=lambda: int(
          bool(
            (pending_window.filter() and pending.rows(get_state()))
            or queued_input_pending()
          )
        )
      ),
      activity_window,
      containers.ConditionalContainer(
        host.mount(components['extensions_below_activity']),
        filter=filters.Condition(lambda: occupied(activity_window)),
      ),
      host.mount(components['failure']),
      containers.Window(
        height=lambda: int(occupied(activity_window) and occupied(above_prompt))
      ),
      above_prompt,
      containers.ConditionalContainer(
        containers.Window(
          # Above-input content supplies separation without another blank row.
          height=lambda: int(
            not secret_active()
            and not occupied(above_input)
            and bool(
              occupied(activity_window)
              or get_state().lifecycle.failure
              or occupied(above_prompt)
            )
          )
        ),
        filter=composer_visible,
      ),
      host.mount(components['resume']),
    ])
    body: containers.Container = layout.Rows([
      self._activity_prefix,
      host.mount(components['selector']),
      containers.ConditionalContainer(
        layout.Rows([
          host.mount(components['secret']),
          host.mount(components['rewind_picker']),
          host.mount(components['rewind_banner']),
        ]),
        filter=filters.Condition(
          lambda: not fullscreen or not get_state().selector.active
        ),
      ),
    ])
    self.tasks = host.mount(components['tasks'])
    below = layout.Rows([
      self.tasks,
      containers.Window(
        height=lambda: dimension.Dimension.exact(
          int(
            bool(
              get_state().tasks.tasks
              and get_state().status_bar.view_mode is not actions.ViewMode.FOCUS
            )
          )
        )
      ),
      containers.ConditionalContainer(footer, filter=~composer_visible),
      below_prompt,
    ])
    self._fixed = layout.Rows(
      [above_input, composer, below]
      if fullscreen
      else [above_input, composer, below, navigation]
    )
    visible_body: containers.Container = body
    if fullscreen:
      body = retained.Layer(
        body,
        # Above-input visibility changes the separator, even at the same budget.
        lambda: (
          self._body_height(),
          occupied(above_input),
          self.paint_key(),
        ),
      )
      body_pane = layout.ScrollableRows(
        body,
        show_scrollbar=False,
        keep_cursor_visible=False,
        keep_focused_window_visible=False,
      )

      def activity_height() -> dimension.Dimension:
        size = self._get_size()
        preferred = body.preferred_height(
          size.columns, body_pane.max_available_height
        ).preferred
        visible = min(preferred, self._body_height())
        body_pane.vertical_scroll = max(0, preferred - visible)
        return dimension.Dimension.exact(visible)

      body_pane.height = activity_height
      visible_body = body_pane
    self.prompt = layout.Rows([visible_body, above_input, composer, below])
    self.bottom = layout.Rows([self.prompt, navigation])
    self._dock_content = self.prompt if fullscreen else self.bottom
    self.dock = layout.Rows(
      [self._dock_content],
      height=lambda: dimension.Dimension.exact(self.height()),
    )

  def paint_key(self) -> tuple[object, ...]:
    """Keep unkeyed menu and extension callbacks live on transcript-only draws."""
    return self._paint_keys.get((), self._component_key)

  def _component_key(self) -> tuple[object, ...]:
    snapshot = self._get_state()
    keys = tuple(
      component.cache_key(snapshot) for component in self._paint_components
    )
    return (
      (current.get_app().render_counter,)
      if any(key is None for key in keys)
      else keys
    )

  def _budget(self) -> int:
    size = self._get_size()
    # On small terminals, reserve the footer before allocating activity rows.
    minimum = self._fixed.preferred_height(size.columns, size.rows).min
    # Approval inspection takes priority over history, leaving a little context.
    budget = (
      size.rows - 3
      if self._get_state().approval.request
      else size.rows * 2 // 3
    )
    # Above-input rows borrow transcript space, not activity capacity.
    above_input = self._above_input.preferred_height(
      size.columns, size.rows
    ).preferred
    return min(size.rows, max(minimum, budget + above_input))

  def _body_height(self) -> int:
    size = self._get_size()
    return max(
      0,
      self._budget()
      - self._fixed.preferred_height(size.columns, size.rows).preferred,
    )

  def selector_max_height(self) -> int:
    """Prioritize the active selector over concurrent activity."""
    size = self._get_size()
    limit = self._body_height()
    prefix = self._activity_prefix.preferred_height(
      size.columns, size.rows
    ).preferred
    snapshot = self._get_state()
    request = snapshot.selector.selector
    minimum = (
      len(snapshot.selector.filtered_choices()) + 5
      if request is not None and request.header
      else 4
    )
    return max(4, limit - min(prefix, max(0, limit - minimum)))

  def height(self) -> int:
    """Bound the dock without letting activity displace the composer."""
    size = self._get_size()
    preferred = self._dock_content.preferred_height(
      size.columns, size.rows
    ).preferred
    return max(1, min(preferred, self._budget()))


def _bounded_widget(
  child: containers.Container, get_size: Callable[[], data_structures.Size]
) -> containers.Container:
  def height() -> dimension.Dimension:
    size = get_size()
    limit = max(1, size.rows // 6)
    return dimension.Dimension.exact(
      min(child.preferred_height(size.columns, size.rows).preferred, limit)
    )

  return layout.Rows(
    [layout.ScrollableRows(child, show_scrollbar=False)],
    height=height,
  )
