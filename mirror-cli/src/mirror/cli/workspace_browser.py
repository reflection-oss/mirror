"""The workspace landing screen uses the shared selector and search composer."""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import pathlib
from collections.abc import Callable, Mapping

import grpc
from mirror.protocol import daemon_pb2
from prompt_toolkit import (
  buffer,
  filters,
  formatted_text,
  styles,
)
from prompt_toolkit.application import current
from prompt_toolkit.key_binding import key_processor
from prompt_toolkit.layout import containers, controls, layout, processors
from rich import align, text

from mirror.cli import (
  actions,
  daemon_client,
  extension_state,
  frontend,
  rendering,
  selection,
  state,
  themes,
  workspace_application,
  workspace_controller,
  workspaces,
)
from mirror.cli.components import banner, selector, worktree_tabs
from mirror.cli.composer import surface


def _single_line(value: str) -> str:
  return ' '.join(rendering.strip_ansi(value).split())


def _directory_label(directory: str) -> str:
  if not directory:
    return 'Remote'
  path = pathlib.Path(directory)
  home = pathlib.Path.home()
  if path == home:
    return '~'
  if path.is_relative_to(home):
    return _single_line(f'~/{path.relative_to(home)}')
  return _single_line(directory)


def _title(
  record: daemon_pb2.Workspace, info: daemon_client.ConversationInfo | None
) -> str:
  # Explicit names take precedence over automatically generated prompt summaries.
  if record.name and record.name != pathlib.Path(record.directory).name:
    return record.name
  if info is not None and info.title:
    return info.title
  return record.name or f'Conversation {record.session_id[:8]}'


def _status(
  record: daemon_pb2.Workspace,
  view: workspaces.AgentWorkspace | None,
) -> tuple[str, str]:
  if view is not None:
    activity = view.state
  elif record.state in (
    daemon_pb2.WORKSPACE_STATE_AVAILABLE,
    daemon_pb2.WORKSPACE_STATE_CLOSED,
  ):
    # Not open in this CLI; a worker elsewhere may still be running.
    activity = actions.WorktreeAgentState.SAVED
  elif record.state == daemon_pb2.WORKSPACE_STATE_PROVISIONING:
    activity = actions.WorktreeAgentState.CREATING
  else:
    activity = actions.WorktreeAgentState.FAILED
  return worktree_tabs.status_indicator(activity)


def _updated_at(
  record: daemon_pb2.Workspace, info: daemon_client.ConversationInfo | None
) -> datetime.datetime | None:
  if info is not None and info.updated_at is not None:
    return info.updated_at
  if record.HasField('created_at'):
    return record.created_at.ToDatetime(tzinfo=datetime.UTC)
  return None


def workspace_choices(
  controller: workspace_controller.WorkspaceController,
  summaries: Mapping[str, daemon_client.ConversationInfo],
) -> tuple[selection.Choice, ...]:
  """Join catalog and history metadata without constructing or attaching agents."""
  views = {
    view.id: view
    for view in controller.workspaces
    if isinstance(view, workspaces.AgentWorkspace)
  }
  records = sorted(
    controller.workspace_catalog(),
    key=lambda record: (
      record.directory != str(controller.launch_workspace),
      record.directory,
      -updated_at.timestamp()
      if (updated_at := _updated_at(record, summaries.get(record.session_id)))
      is not None
      else 0,
      record.session_id,
      record.id,
    ),
  )
  choices: list[selection.Choice] = []
  for record in records:
    info = summaries.get(record.session_id)
    icon, style = _status(record, views.get(record.id))
    label = _title(record, info)
    model = (
      info.model if info is not None else record.preferences.definition.model
    )
    if record.HasField('inference'):
      model = record.inference.model
    kind = workspace_controller.workspace_kind(record)
    details: list[str] = []
    updated_at = _updated_at(record, info)
    if updated_at is not None:
      details.append(f'{updated_at.astimezone():%Y-%m-%d %H:%M}')
    if model:
      details.append(model)
    if record.name not in ('', label, pathlib.Path(record.directory).name):
      details.append(record.name)
    choices.append(
      selection.Choice(
        f'resume:{record.session_id}',
        _single_line(label),
        details[0] if updated_at is not None else '',
        prefix=icon,
        prefix_style=style,
        status=f'{kind:8}',
        section=_directory_label(record.directory),
        search_terms=tuple(
          _single_line(detail)
          for detail in (
            record.directory,
            kind,
            info.title or '' if info else '',
            *details,
          )
        ),
      )
    )
  return tuple(choices)


def recent_choices(
  summaries: Mapping[str, daemon_client.ConversationInfo],
  catalog: tuple[selection.Choice, ...],
) -> tuple[selection.Choice, ...]:
  """Show the ten latest conversations still in the workspace catalog."""
  current = {choice.value: choice for choice in catalog}
  recent = sorted(
    (info for info in summaries.values() if f'resume:{info.id}' in current),
    key=lambda info: (
      info.updated_at.timestamp() if info.updated_at is not None else 0,
      info.id,
    ),
    reverse=True,
  )[:10]
  return tuple(
    dataclasses.replace(
      current[f'resume:{info.id}'],
      value=f'recent:{info.id}',
      detail=(
        f'{info.updated_at.astimezone():%Y-%m-%d %H:%M} · '
        if info.updated_at is not None
        else ''
      )
      + _directory_label(info.cwd),
      section='Recent',
      section_label='',
      search_terms=current[f'resume:{info.id}'].search_terms,
    )
    for info in recent
  )


def deletion_choices(
  record: daemon_pb2.Workspace,
) -> tuple[selection.Choice, ...]:
  """Separate catalog removal from explicitly destructive managed cleanup."""
  remote = record.kind == daemon_pb2.WORKSPACE_KIND_REMOTE
  choices = (
    selection.Choice('cancel', 'Cancel', 'Nothing will be changed.'),
    selection.Choice(
      'delete',
      'Delete conversation' if remote else 'Remove from list',
      'Deletes the remote conversation and its history.'
      if remote
      else 'Stops its agent; keeps local files and conversation history.',
    ),
  )
  if workspace_controller.can_delete_workspace_files(record):
    kind = workspace_controller.workspace_kind(record)
    choices += (
      selection.Choice(
        'delete-files',
        f'Delete {kind} and files',
        'Deletes files, including uncommitted changes; keeps conversation history.',
      ),
    )
  return choices


async def activate(
  controller: workspace_controller.WorkspaceController,
  screen: workspace_application.WorkspaceApplication | None = None,
  *,
  theme_name: Callable[[], str] = lambda: themes.effective('dark'),
  frontend_context: frontend.Context | None = None,
) -> str | None:
  """Browse metadata; execution and deletion require explicit user actions."""
  view = _Browser(controller, screen, theme_name, frontend_context)
  controller.refresh()
  view.load_summaries()
  try:
    if screen is not None:
      return await screen.show_view(
        view.app, accept_buffered_input=lambda: False
      )
    view.app.prepare_for_run()
    return await view.app.run_async()
  finally:
    if view.details_task is not None:
      view.details_task.cancel()
      await asyncio.gather(view.details_task, return_exceptions=True)
    if view.opening_task is not None:
      view.opening_task.cancel()
      await asyncio.gather(view.opening_task, return_exceptions=True)
    if view.deletion_task is not None:
      await asyncio.shield(view.deletion_task)
    view.app.finish_handoff()


class _Browser:
  def __init__(
    self,
    controller: workspace_controller.WorkspaceController,
    screen: workspace_application.WorkspaceApplication | None,
    theme_name: Callable[[], str],
    frontend_context: frontend.Context | None = None,
  ) -> None:
    self.frontend = frontend_context
    self.frontend_selector: selection.Selector[actions.Action] | None = None
    self.controller = controller
    self.screen = screen
    self.theme_name = theme_name
    self.selected = controller.selected
    self.menu = state.MenuState(active=True, max_rows=8)
    self.collapsed = controller.collapsed_workspace_sections
    self.highlight: str | None = (
      'section:New' if 'New' in self.collapsed else 'create:normal'
    )
    self.summaries: dict[str, daemon_client.ConversationInfo] = {}
    self.details_loaded = False
    self.details_task: asyncio.Task[None] | None = None
    self.deletion_task: asyncio.Task[None] | None = None
    self.confirmation: daemon_pb2.Workspace | None = None
    self.confirm_menu = state.MenuState(active=True, max_rows=3)
    self.notice = ''
    self.entering_folder = False
    self.folder_trust: pathlib.Path | None = None
    self.directory_exists = True
    self.opening_task: asyncio.Task[None] | None = None
    self.trust_session: str | None = None
    self.search = buffer.Buffer(
      multiline=False,
      on_text_changed=self.search_changed,
      read_only=filters.Condition(
        lambda: (
          self.confirmation is not None
          or self.folder_trust is not None
          or self.deleting
        )
      ),
    )
    self.prompt = surface.input_window(
      self.search,
      hidden=lambda: (
        self.confirmation is not None
        or self.folder_trust is not None
        or self.deleting
      ),
      max_height=lambda: 1,
      input_processors=[
        processors.AfterInput(lambda: [('class:ui.muted', self.input_hint())])
      ],
    )
    self.selector = selector.Selector(
      self.menu_height, inset=False, minimal_chrome=True
    )
    footer = containers.VSplit(
      [
        containers.Window(controls.FormattedTextControl(self.hint)),
        containers.Window(
          controls.FormattedTextControl(
            lambda: (
              [('class:footer', f' {status} ')]
              if self.frontend and (status := self.frontend.status())
              else []
            )
          ),
          dont_extend_width=True,
        ),
      ],
      height=lambda: int(
        current.get_app().output.get_size().rows >= 12
        or bool(self.frontend and self.frontend.status())
      ),
    )
    composer = surface.wrap_input(
      self.prompt, lambda: False, footer, theme_name=theme_name
    )
    compact_composer = containers.HSplit(
      [
        containers.VSplit(
          [containers.Window(width=1), self.prompt, containers.Window(width=1)],
          height=lambda: self.prompt.preferred_height(1, 1).preferred,
        ),
        footer,
      ],
      style='class:composer',
    )
    self.composer = containers.DynamicContainer(
      lambda: (
        composer
        if current.get_app().output.get_size().rows >= 12
        else compact_composer
      )
    )
    bindings = workspace_application.terminal_bindings()
    for key in ('up', 'down', 'c-j', 'c-k'):
      bindings.add(key)(self.move)
    for key in ('left', 'right'):
      bindings.add(
        key,
        filter=filters.Condition(
          lambda: (
            not self.search.text
            and self.frontend_selector is None
            and self.confirmation is None
            and self.folder_trust is None
            and not self.entering_folder
            and not self.deleting
          )
        ),
      )(self.fold)
    for key in '123456789':
      bindings.add(
        key, filter=filters.Condition(lambda: self.selector_state().numbered)
      )(self.accept_number)
    bindings.add('enter')(self.accept)
    bindings.add('delete')(self.confirm_delete)
    bindings.add(
      'backspace',
      filter=filters.Condition(
        lambda: not self.search.text and not self.entering_folder
      ),
    )(self.confirm_delete)
    bindings.add('escape', eager=True)(self.back)
    for key in ('c-s', 'c-r', 'c-c', 'c-d'):
      bindings.add(key)(self.shortcut)
    self.app = workspace_application.WorkspaceApplication(
      layout=layout.Layout(
        containers.DynamicContainer(self.content), focused_element=self.prompt
      ),
      key_bindings=bindings,
      full_screen=True,
      style=styles.DynamicStyle(lambda: themes.prompt_style(theme_name())),
      color_depth=themes.prompt_color_depth(),
      refresh_interval=0.1,
      before_render=lambda _: self.refresh(),
    )
    self.app.ttimeoutlen = 0.05

  @property
  def deleting(self) -> bool:
    return self.deletion_task is not None and not self.deletion_task.done()

  def load_summaries(self) -> None:
    if self.details_task is None or self.details_task.done():
      self.details_task = asyncio.create_task(self.read_summaries())

  async def read_summaries(self) -> None:
    try:
      summaries = await self.controller.conversation_summaries()
      self.summaries = {info.id: info for info in summaries}
    except (grpc.RpcError, ValueError) as error:
      self.notice = f'Could not load conversation details: {error}'
    self.details_loaded = True
    (self.screen or self.app).invalidate()

  def input_hint(self) -> str:
    if self.frontend and self.search.text.startswith('/'):
      name, separator, arguments = self.search.text[1:].partition(' ')
      command = next(
        (item for item in self.frontend.commands() if item.name == name), None
      )
      if separator and not arguments and command and command.arguments:
        return command.arguments.hint
    if self.search.text:
      return ''
    if self.frontend_selector is not None:
      return 'Search…'
    return 'Directory path…' if self.entering_folder else 'Search workspaces…'

  def dispatch(self, action: actions.Action) -> None:
    match action:
      case actions.OpenSelector(request):
        self.frontend_selector = request
        self.search.reset()
        self.highlight = request.selected or request.current
      case actions.ClearSelector():
        self.frontend_selector = None
        self.search.reset()
      case actions.RestorePrompt(value, cursor):
        self.search.text = value
        self.search.cursor_position = len(value) if cursor is None else cursor
      case actions.ShowText(message):
        self.notice = rendering.renderable_to_ansi(
          message,
          current.get_app().output.get_size().columns,
          theme_name=self.theme_name(),
        )
      case actions.RunFrontendCommand(name, arguments) if self.frontend:
        self.frontend.start(name, arguments, dispatch=self.dispatch)
      case _:
        self.notice = f'{type(action).__name__} requires an open workspace.'
    (self.screen or self.app).invalidate()

  def picker(self) -> selection.Selector[object]:
    if self.frontend_selector is not None:
      return self.frontend_selector
    size = current.get_app().output.get_size()
    if self.deleting:
      return selection.Selector(
        'deleting-workspace',
        'Updating list…',
        (),
        str,
        empty='Please wait.',
        footer='',
      )
    if self.confirmation is not None:
      remote = self.confirmation.kind == daemon_pb2.WORKSPACE_KIND_REMOTE
      info = self.summaries.get(self.confirmation.session_id)
      title = _title(self.confirmation, info)
      return selection.Selector(
        'delete-workspace',
        'Delete remote conversation?' if remote else 'Remove from list?',
        deletion_choices(self.confirmation),
        str,
        header=text.Text.assemble(
          (_single_line(title), 'bold'),
          ('\n' + _directory_label(self.confirmation.directory), 'ui.muted')
          if size.rows >= 12
          else '',
        ),
        stacked=size.rows >= 18,
        numbered=True,
        footer='',
      )
    if self.folder_trust is not None:
      return selection.Selector(
        'folder-trust',
        'Trust directory?' if self.directory_exists else 'Create directory?',
        (
          selection.Choice('cancel', 'Cancel'),
          selection.Choice(
            'trust',
            'Trust and open' if self.directory_exists else 'Create and open',
          ),
        ),
        str,
        numbered=True,
        header=text.Text(
          f'{self.folder_trust}\n\n'
          'Mirror can read/edit files and run commands here.\n'
          'Only open directories whose files and instructions you trust.'
        ),
        footer='',
      )
    if self.entering_folder:
      return selection.Selector(
        'new-folder',
        'Open directory',
        (),
        str,
        empty='Enter a directory path below.',
        footer='',
      )
    choices = tuple(
      dataclasses.replace(
        choice, value=f'create:{choice.value}', prefix='+', section='New'
      )
      for choice in self.controller.new_workspace_selector().choices
    )
    if self.details_loaded:
      catalog = workspace_choices(self.controller, self.summaries)
      choices += recent_choices(self.summaries, catalog) + catalog
    choices = self.grouped_choices(choices)
    current_value = None
    if isinstance(self.selected, workspaces.AgentWorkspace):
      session_id = self.selected.agent.conversation_id
      current_value = f'resume:{session_id}'
      if any(choice.value == f'recent:{session_id}' for choice in choices):
        current_value = f'recent:{session_id}'
    return selection.Selector(
      'workspace-browser',
      'Workspaces',
      choices,
      str,
      empty='Loading conversations…'
      if self.details_task is not None and not self.details_task.done()
      else 'No conversations.',
      current=current_value,
      footer='',
    )

  def selector_state(self) -> state.SelectorState:
    return state.SelectorState(
      selector=self.picker(),
      menu=self.confirm_menu if self.confirmation is not None else self.menu,
      query=''
      if self.confirmation is not None
      or self.folder_trust is not None
      or self.entering_folder
      or self.deleting
      else self.search.text,
    )

  def grouped_choices(
    self, choices: tuple[selection.Choice, ...]
  ) -> tuple[selection.Choice, ...]:
    if self.search.text:
      return choices
    groups: dict[str, list[selection.Choice]] = {}
    for choice in choices:
      groups.setdefault(choice.section, []).append(choice)
    result: list[selection.Choice] = []
    for section, children in groups.items():
      result.append(
        selection.Choice(
          f'section:{section}',
          children[0].section_label or section,
          str(len(children)),
          prefix='▸' if section in self.collapsed else '▾',
          section=section,
          is_heading=True,
        )
      )
      if section not in self.collapsed:
        result.extend(children)
    return tuple(result)

  def fold(self, event: key_processor.KeyPressEvent) -> None:
    choice = self.selected_choice()
    if choice is None:
      return
    if event.key_sequence[-1].key == 'left':
      self.collapsed.add(choice.section)
      self.highlight = f'section:{choice.section}'
    elif choice.section in self.collapsed:
      self.collapsed.remove(choice.section)
    elif choice.is_heading:
      choices = self.picker().choices
      self.highlight = next(
        (
          child.value
          for child in choices
          if child.section == choice.section and not child.is_heading
        ),
        choice.value,
      )

  def search_changed(self, value: buffer.Buffer) -> None:
    del value
    self.highlight = None
    self.menu = self.menu.open()

  def refresh(self) -> None:
    self.menu = dataclasses.replace(self.menu, max_rows=self.menu_height())
    if self.deleting:
      return
    if self.controller.selected is not self.selected:
      self.finish('replaced')
      return
    if self.confirmation is not None:
      return
    choices = self.selector_state().filtered_choices()
    selected = next(
      (
        index
        for index, choice in enumerate(choices)
        if choice.value == self.highlight
      ),
      min(self.menu.selected, max(0, len(choices) - 1)),
    )
    self.menu = dataclasses.replace(self.menu, selected=selected)
    self.highlight = choices[selected].value if choices else None

  def move(self, event: key_processor.KeyPressEvent) -> None:
    if self.deleting:
      return
    direction = -1 if event.key_sequence[-1].key in ('up', 'c-k') else 1
    if self.confirmation is not None:
      self.confirm_menu = self.confirm_menu.move(
        direction, len(deletion_choices(self.confirmation))
      )
      return
    choices = self.selector_state().filtered_choices()
    self.menu = self.menu.move(direction, len(choices))
    self.highlight = choices[self.menu.selected].value if choices else None

  def selected_choice(self) -> selection.Choice | None:
    choices = self.selector_state().filtered_choices()
    if self.highlight is None:
      return self.menu.selected_item(choices)
    return next(
      (choice for choice in choices if choice.value == self.highlight), None
    )

  def accept_number(self, event: key_processor.KeyPressEvent) -> None:
    current = self.selector_state()
    request = current.selector
    if request is None or not current.numbered:
      return
    index = int(event.data) - 1 + request.numbered_from
    choices = current.filtered_choices()
    if index >= len(choices):
      return
    if self.confirmation is not None:
      self.confirm_menu = dataclasses.replace(self.confirm_menu, selected=index)
    else:
      self.menu = dataclasses.replace(self.menu, selected=index)
      self.highlight = choices[index].value
    self.accept(event)

  def accept(self, _: key_processor.KeyPressEvent) -> None:
    if self.deleting:
      return
    if self.frontend_selector is not None:
      request = self.frontend_selector
      choice = self.selected_choice()
      if choice is not None:
        if not request.keep_open:
          self.dispatch(actions.ClearSelector())
        self.dispatch(request.on_select(choice.value))
      return
    if self.confirmation is not None:
      choice = self.confirm_menu.selected_item(
        deletion_choices(self.confirmation)
      )
      if choice is not None and choice.value != 'cancel':
        self.deletion_task = asyncio.create_task(
          self.delete(
            self.confirmation.id, delete_files=choice.value == 'delete-files'
          )
        )
      else:
        self.confirmation = None
      return
    if self.folder_trust is not None:
      choice = self.selected_choice()
      if choice is not None and choice.value == 'trust':
        if self.trust_session is not None:
          self.open_conversation(self.trust_session, confirmed=True)
        else:
          self.open_folder(confirmed=True)
      else:
        self.folder_trust = None
      return
    if self.entering_folder:
      self.open_folder()
      return
    command = self.search.text.strip()
    name, _separator, arguments = command[1:].partition(' ')
    if (
      self.frontend
      and command.startswith('/')
      and any(item.name == name for item in self.frontend.commands())
    ):
      self.search.reset()
      self.dispatch(actions.RunFrontendCommand(name, arguments))
    elif command in ('/open', '/new'):
      self.search.reset()
      self.collapsed.discard('New' if command == '/new' else 'Recent')
      self.menu = self.menu.open()
      self.highlight = (
        'create:normal'
        if command == '/new'
        else next(
          (
            choice.value
            for choice in self.picker().choices
            if choice.value.startswith(('recent:', 'resume:'))
          ),
          None,
        )
      )
    elif command == '/quit':
      self.finish(None)
    else:
      choice = self.selected_choice()
      if choice is not None and choice.is_heading:
        self.collapsed.symmetric_difference_update((choice.section,))
      elif (
        choice is not None
        and choice.value == 'create:normal'
        and not self.controller.is_trusted_directory(
          self.controller.launch_workspace
        )
      ):
        self.entering_folder = True
        self.search.text = str(self.controller.launch_workspace)
        self.open_folder()
      elif choice is not None and choice.value == 'create:folder':
        self.entering_folder = True
        self.notice = ''
        self.search.reset()
      elif choice is not None and choice.value.startswith((
        'recent:',
        'resume:',
      )):
        self.open_conversation(choice.value.split(':', 1)[1])
      elif choice is not None:
        self.finish(choice.value)

  def open_conversation(
    self, session_id: str, *, confirmed: bool = False
  ) -> None:
    record = next(
      (
        record
        for record in self.controller.workspace_catalog()
        if session_id in (record.session_id, *record.previous_session_ids)
      ),
      None,
    )
    if record is None:
      self.folder_trust = None
      self.notice = 'This workspace is no longer available. Refresh the list.'
      return
    if record.kind != daemon_pb2.WORKSPACE_KIND_DIRECTORY:
      self.finish(f'resume:{session_id}')
      return
    self.resolve_open(
      record.directory, confirmed=confirmed, session_id=session_id
    )

  def open_folder(self, *, confirmed: bool = False) -> None:
    entered = self.search.text.strip()
    if entered:
      self.resolve_open(entered, confirmed=confirmed)

  def resolve_open(
    self, entered: str, *, confirmed: bool, session_id: str | None = None
  ) -> None:
    if self.opening_task is None or self.opening_task.done():
      self.opening_task = asyncio.create_task(
        self.open_directory(entered, confirmed=confirmed, session_id=session_id)
      )

  async def open_directory(
    self, entered: str, *, confirmed: bool, session_id: str | None
  ) -> None:
    try:
      resolved = await asyncio.to_thread(
        self.controller.resolve_directory,
        entered,
        base=self.selected.path or self.controller.launch_workspace,
      )
      directory = pathlib.Path(resolved.directory)
      if session_id is not None and not resolved.exists:
        raise ValueError(f'directory is unavailable: {directory}')
      if not resolved.exists or not self.controller.is_trusted_directory(
        directory
      ):
        if (
          not confirmed
          or directory != self.folder_trust
          or resolved.exists != self.directory_exists
        ):
          self.folder_trust = directory
          self.directory_exists = resolved.exists
          self.trust_session = session_id
          self.highlight = None
          self.menu = self.menu.open()
          return
        self.controller.trust_directory(directory)
      if session_id is not None:
        self.finish(f'resume:{session_id}')
      else:
        operation = 'folder' if resolved.exists else 'create-folder'
        self.finish(f'{operation}:{directory}')
    except (OSError, ValueError, grpc.RpcError) as error:
      self.folder_trust = None
      self.notice = str(error)

  def confirm_delete(self, _: key_processor.KeyPressEvent) -> None:
    if self.frontend_selector is not None:
      return
    if (
      self.deleting
      or self.confirmation is not None
      or self.folder_trust is not None
      or self.entering_folder
    ):
      return
    choice = self.selected_choice()
    if choice is None or not choice.value.startswith(('recent:', 'resume:')):
      return
    self.confirmation = next(
      (
        record
        for record in self.controller.workspace_catalog()
        if choice.value.split(':', 1)[1] == record.session_id
      ),
      None,
    )
    self.confirm_menu = self.confirm_menu.open()

  async def delete(
    self, workspace_id: str, *, delete_files: bool = False
  ) -> None:
    try:
      await self.controller.delete_workspace(
        workspace_id, delete_files=delete_files
      )
      if delete_files:
        self.notice = 'Workspace and files deleted.'
      elif (
        self.confirmation is not None
        and self.confirmation.kind == daemon_pb2.WORKSPACE_KIND_REMOTE
      ):
        self.notice = 'Conversation deleted.'
      else:
        self.notice = 'Workspace removed from list.'
    except (grpc.RpcError, ValueError, OSError, RuntimeError) as error:
      self.notice = f'Could not remove entry: {error}'
      self.controller.refresh()
    finally:
      self.selected = self.controller.selected
      self.confirmation = None
      (self.screen or self.app).invalidate()

  def shortcut(self, event: key_processor.KeyPressEvent) -> None:
    if self.deleting:
      return
    key = event.key_sequence[-1].key
    if self.frontend_selector is not None:
      if key in ('c-c', 'c-d'):
        self.back(event)
      return
    if self.confirmation is not None:
      if key in ('c-c', 'c-d'):
        self.confirmation = None
      return
    if self.folder_trust is not None:
      if key in ('c-c', 'c-d'):
        self.folder_trust = None
      return
    if key == 'c-s':
      if self.frontend is not None:
        self.dispatch(actions.RunFrontendCommand('ssh'))
    elif key == 'c-r':
      self.controller.refresh()
      self.load_summaries()
    else:
      self.finish(None)

  def back(self, _: key_processor.KeyPressEvent) -> None:
    if self.deleting:
      return
    if self.frontend_selector is not None:
      cancel = self.frontend_selector.cancel_action
      self.dispatch(actions.ClearSelector())
      if cancel is not None:
        self.dispatch(cancel)
    elif self.confirmation is not None:
      self.confirmation = None
    elif self.folder_trust is not None:
      self.folder_trust = None
    elif self.entering_folder:
      self.entering_folder = False
      self.search.reset()
      self.highlight = None
      self.menu = self.menu.open()
    elif self.search.text:
      self.search.reset()
    elif self.controller.visible_workspaces():
      self.finish('back')

  def finish(self, result: str | None) -> None:
    application = self.screen or self.app
    if result is None:
      application.exit()
    else:
      application.exit_for_handoff(result)

  def hint(self) -> formatted_text.StyleAndTextTuples:
    if self.frontend_selector is not None:
      return [
        ('class:ui.muted', ' ↑↓ move · Enter select · Esc cancel'),
      ]
    if self.deleting:
      return [('class:ui.muted', ' Please wait…')]
    width = current.get_app().output.get_size().columns
    hints = [('↑↓', 'move')]
    if self.selector_state().numbered:
      hints.append((f'1–{len(self.picker().choices)}', 'choose'))
    if self.confirmation is not None:
      action = (
        'delete'
        if self.confirmation.kind == daemon_pb2.WORKSPACE_KIND_REMOTE
        or self.confirm_menu.selected == 2
        else 'remove'
      )
      hints.extend((
        ('Enter', action if self.confirm_menu.selected else 'cancel'),
        ('Esc', 'cancel'),
      ))
    elif self.folder_trust is not None:
      choice = self.selected_choice()
      hints.extend((
        ('Enter', 'trust' if choice and choice.value == 'trust' else 'cancel'),
        ('Esc', 'back'),
      ))
    elif self.entering_folder:
      hints = [('Enter', 'open directory'), ('Esc', 'back')]
    else:
      choice = self.selected_choice()
      if choice is not None:
        creating = choice.value.startswith('create:')
        action = 'open'
        if choice.is_heading:
          action = 'expand' if choice.section in self.collapsed else 'collapse'
        elif creating:
          action = 'create' if choice.value == 'create:normal' else 'choose'
        hints.append(('Enter', action))
        if choice.value.startswith(('recent:', 'resume:')):
          hints.append((
            'Del' if self.search.text or width < 80 else 'Backspace/Del',
            'remove',
          ))
      else:
        hints = []
      if choice is not None and not self.search.text:
        hints.append(('←/→', 'fold'))
      if self.search.text:
        hints.append(('Esc', 'clear'))
      elif self.controller.visible_workspaces():
        hints.append(('Esc', 'back'))
      else:
        hints.append(('Ctrl-D', 'quit'))
    if width < 60:
      hints = [(key, label) for key, label in hints if key != '↑↓']
    if width >= 100:
      hints = [
        ('↑↓ / Ctrl-J/K' if key == '↑↓' else key, label) for key, label in hints
      ]
    fragments: formatted_text.StyleAndTextTuples = [('', ' ')]
    for index, (key, label) in enumerate(hints):
      if index:
        fragments.append(('', '   '))
      fragments.extend((
        ('class:ui.text', key),
        ('class:ui.muted', f' {label}'),
      ))
    return fragments

  def logo(self) -> text.Text:
    output = current.get_app().output
    size = output.get_size()
    if (
      size.rows < 18
      or self.frontend_selector is not None
      or self.confirmation is not None
      or self.folder_trust is not None
      or self.deleting
    ):
      return text.Text()
    return banner.artwork(
      size.columns - 1,
      size.rows,
      output.encoding(),
    )

  def notice_text(self) -> str:
    return (
      self.notice
      or (
        self.selected.notice
        if isinstance(self.selected, workspaces.EmptyWorkspace)
        else ''
      )
      or ('Loading conversations…' if not self.details_loaded else '')
    )

  def menu_height(self) -> int:
    size = current.get_app().output.get_size()
    occupied = self.composer.preferred_height(size.columns, size.rows).preferred
    image = self.logo()
    occupied += len(image.split()) if image else 0
    return max(3, size.rows - occupied - int(bool(self.notice_text())))

  def content(self) -> containers.AnyContainer:
    output = current.get_app().output
    size = output.get_size()
    width = size.columns
    snapshot = state.State(
      activity=state.ActivityState(),
      status_bar=state.StatusBarState('', '', ''),
      session=state.SessionState(''),
      lifecycle=state.LifecycleState(active=True),
      selector=self.selector_state(),
      layout=state.LayoutState(columns=width),
      extensions=extension_state.ExtensionStates().configured(
        themes.EXTENSION_ID, {'name': self.theme_name()}
      ),
    )
    menu = containers.to_container(self.selector.render(snapshot))
    notice = self.notice_text()
    image = self.logo()
    logo = formatted_text.ANSI(
      rendering.renderable_to_ansi(
        align.Align.left(image), max(1, width - 1), theme_name=self.theme_name()
      ).rstrip()
    )
    return containers.HSplit(
      [
        containers.VSplit(
          [
            containers.Window(width=1),
            containers.Window(controls.FormattedTextControl(logo)),
          ],
          height=len(image.split()) if image else 0,
        ),
        menu,
        containers.Window(),
        containers.Window(
          controls.FormattedTextControl([
            ('class:ui.muted', _single_line(notice))
          ]),
          height=int(bool(notice)),
        ),
        self.composer,
      ],
      style='class:theme-root class:workspace-browser',
    )
