from __future__ import annotations

import asyncio
import pathlib
import webbrowser
from typing import cast
from unittest import mock

import pytest
from prompt_toolkit import data_structures as point
from prompt_toolkit import mouse_events
from prompt_toolkit.layout import containers, mouse_handlers, screen

from mirror.cli import actions, commands, extension, extension_state, state
from mirror.cli.extensions.session_link import extension as session_link

_SESSION = '0f9e8d7c-6b5a-4321-8765-43210fedcba9'


def _context(
  dispatched: list[actions.Action], endpoint: str | None, *, secure: bool = True
) -> extension.Context:
  return extension.Context(
    extension.Workspace(pathlib.Path('/workspace'), remote=True),
    dispatch=dispatched.append,
    get_state=lambda: pytest.fail('Unexpected state read'),
    enqueue=lambda _message: None,
    refresh_tools=lambda _name: None,
    call_tool=lambda *_: pytest.fail('Unexpected tool call'),
    transcript=tuple,
    builtins=tuple,
    extension_tools=tuple,
    workspace_controller=mock.Mock(),
    create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
    remote_endpoint=endpoint,
    remote_secure=secure,
  )


def _render_text(container: containers.AnyContainer) -> str:
  content = containers.to_container(container)
  width = content.preferred_width(80).preferred
  painted = screen.Screen()
  content.write_to_screen(
    painted,
    mouse_handlers.MouseHandlers(),
    screen.WritePosition(0, 0, width, 1),
    '',
    True,
    None,
  )
  return ''.join(painted.data_buffer[0][x].char for x in range(width))


def _application(session: state.SessionState) -> state.State:
  return state.State(
    activity=state.ActivityState(),
    status_bar=state.StatusBarState('model', 'medium', 'auto'),
    session=session,
    lifecycle=state.LifecycleState(active=True),
    workspace='/workspace',
    extensions=extension_state.ExtensionStates().updated(
      'core/session-link', session_link.SessionLinkExtension.initial_state()
    ),
  )


def _click(container: containers.AnyContainer) -> object:
  content = containers.to_container(container)
  handlers = mouse_handlers.MouseHandlers()
  content.write_to_screen(
    screen.Screen(),
    handlers,
    screen.WritePosition(0, 0, content.preferred_width(80).preferred, 1),
    '',
    True,
    None,
  )
  return handlers.mouse_handlers[0][0](
    mouse_events.MouseEvent(
      point.Point(0, 0),
      mouse_events.MouseEventType.MOUSE_UP,
      mouse_events.MouseButton.LEFT,
      frozenset(),
    )
  )


@pytest.mark.parametrize(
  ('endpoint', 'secure', 'expected'),
  [
    (
      'mirror.example.com:443',
      True,
      f'https://mirror.example.com/sessions/{_SESSION}',
    ),
    ('127.0.0.1:50051', False, f'http://127.0.0.1:50051/sessions/{_SESSION}'),
  ],
)
def test_web_url_uses_the_gateway_origin(
  endpoint: str, secure: bool, expected: str
) -> None:
  assert session_link.web_url(endpoint, _SESSION, secure=secure) == expected


def test_web_command_routes_hosted_sessions_to_the_extension() -> None:
  plugin = session_link.SessionLinkExtension()
  context = commands.CommandContext(
    'model', 'medium', _SESSION, False, '/workspace', remote=True
  )

  action = plugin.slash_commands()[0].handler(context, '')

  assert action == actions.RoutedExtensionAction(
    'core/session-link', session_link.OpenWeb(_SESSION)
  )


def test_web_command_names_the_local_conversation_instead() -> None:
  plugin = session_link.SessionLinkExtension()
  context = commands.CommandContext(
    'model', 'medium', 'local-conversation', False, '/workspace'
  )

  action = plugin.slash_commands()[0].handler(context, '')

  assert action == actions.ShowText(
    'Only hosted sessions have a Mirror web page. '
    'Conversation ID: local-conversation',
    style='ui.warning',
  )


@pytest.mark.asyncio
async def test_open_web_prints_and_opens_the_session_page() -> None:
  opened: list[str] = []

  async def open_url(url: str) -> None:
    opened.append(url)

  dispatched: list[actions.Action] = []
  plugin = session_link.SessionLinkExtension(open_url)

  await plugin.handle_effect(
    _context(dispatched, 'mirror.example.com:443'),
    session_link.OpenWeb(_SESSION),
  )

  url = f'https://mirror.example.com/sessions/{_SESSION}'
  assert opened == [url]
  assert dispatched == [actions.ShowText(f'Session {_SESSION}\n{url}')]


@pytest.mark.asyncio
async def test_open_web_still_prints_the_link_without_a_browser() -> None:
  async def open_url(url: str) -> None:
    raise webbrowser.Error(url)

  dispatched: list[actions.Action] = []
  plugin = session_link.SessionLinkExtension(open_url)

  await plugin.handle_effect(
    _context(dispatched, '127.0.0.1:50051', secure=False),
    session_link.OpenWeb(_SESSION),
  )

  url = f'http://127.0.0.1:50051/sessions/{_SESSION}'
  assert dispatched == [
    actions.ShowText(f'Session {_SESSION}\n{url}'),
    actions.ShowText(
      f'Could not open a browser. Open {url} manually.', style='ui.warning'
    ),
  ]


@pytest.mark.asyncio
async def test_status_bar_shows_the_short_session_id() -> None:
  registry = extension.ExtensionRegistry()
  entrypoint = pathlib.Path(cast(str, session_link.__file__)).parent
  assert await registry.load(entrypoint) == 'core/session-link'
  host = registry.create_host(_context([], None))
  current = state.State(
    activity=state.ActivityState(),
    status_bar=state.StatusBarState('model', 'medium', 'auto'),
    session=state.SessionState(_SESSION),
    lifecycle=state.LifecycleState(active=True),
    workspace='/workspace',
    extensions=host.initial_states,
  )
  await host.start()

  rendered = host.render(current, extension.RenderRegion.STATUS_RIGHT)

  assert _render_text(rendered).strip() == 'session 0f9e8d7c'


@pytest.mark.asyncio
async def test_configured_server_is_recorded_for_the_status_bar() -> None:
  dispatched: list[actions.Action] = []
  registry = extension.ExtensionRegistry()
  entrypoint = pathlib.Path(cast(str, session_link.__file__)).parent
  assert await registry.load(entrypoint) == 'core/session-link'
  host = registry.create_host(_context(dispatched, 'mirror.example.com:443'))

  await host.start()

  assert (
    actions.RoutedExtensionAction(
      'core/session-link',
      session_link.ServerConfigured('https://mirror.example.com'),
    )
    in dispatched
  )


@pytest.mark.asyncio
async def test_clicking_a_hosted_session_id_opens_its_page() -> None:
  opened: list[str] = []

  async def open_url(url: str) -> None:
    opened.append(url)

  plugin = session_link.SessionLinkExtension(open_url)
  current = _application(state.SessionState(_SESSION, remote=True))
  configured, _ = plugin.reduce(
    current,
    actions.RoutedExtensionAction(
      'core/session-link',
      session_link.ServerConfigured('https://mirror.example.com'),
    ),
  )
  rendered = plugin.render(current, configured)
  assert rendered is not None

  text = _render_text(rendered)
  assert _click(rendered) is None
  await asyncio.sleep(0)

  assert text == 'session 0f9e8d7c'
  assert opened == [f'https://mirror.example.com/sessions/{_SESSION}']


def test_a_local_session_id_is_not_a_link() -> None:
  plugin = session_link.SessionLinkExtension()
  current = _application(state.SessionState(_SESSION))
  configured, _ = plugin.reduce(
    current,
    actions.RoutedExtensionAction(
      'core/session-link',
      session_link.ServerConfigured('https://mirror.example.com'),
    ),
  )
  rendered = plugin.render(current, configured)
  assert rendered is not None

  assert _click(rendered) is NotImplemented
