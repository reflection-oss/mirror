"""Show the session ID and open hosted sessions in Mirror web."""

from __future__ import annotations

import dataclasses
import urllib.parse
import webbrowser
from collections.abc import Awaitable, Callable
from typing import override

from prompt_toolkit import application, mouse_events
from prompt_toolkit.key_binding import key_bindings
from prompt_toolkit.layout import containers, controls, dimension
from prompt_toolkit.layout import mouse_handlers as _mouse_handlers
from prompt_toolkit.layout import screen as _screen

from mirror.cli import actions, commands, extension, state, terminal

_SHORT_ID_LENGTH = 8


@dataclasses.dataclass(frozen=True)
class _State:
  web_origin: str | None = None


@dataclasses.dataclass(frozen=True)
class OpenWeb(actions.ExtensionAction):
  """Open a hosted session's page on the server that hosts it."""

  session_id: str


@dataclasses.dataclass(frozen=True)
class ServerConfigured(actions.ExtensionAction):
  """Record the Mirror web origin of this workspace's configured server."""

  web_origin: str


def web_origin(endpoint: str, *, secure: bool = True) -> str:
  """Return the Mirror web origin for a gRPC `host:port`.

  Mirror web is served from the same origin as the AgentRPC gateway.

  Args:
    endpoint: The gRPC target, such as `mirror.example.com:443`.
    secure: Whether the endpoint uses TLS.

  Returns:
    The origin, omitting the scheme's default port.
  """
  scheme = 'https' if secure else 'http'
  parts = urllib.parse.urlsplit(f'{scheme}://{endpoint}')
  netloc = parts.netloc
  if parts.port == (443 if secure else 80):
    netloc = netloc.rsplit(':', 1)[0]
  return urllib.parse.urlunsplit((scheme, netloc, '', '', ''))


def session_url(origin: str, session_id: str) -> str:
  """Return the Mirror web page for `session_id` under `origin`."""
  return f'{origin}/sessions/{urllib.parse.quote(session_id, safe="")}'


def web_url(endpoint: str, session_id: str, *, secure: bool = True) -> str:
  """Return the Mirror web page for `session_id` on a gRPC `host:port`.

  Mirror web is served from the same origin as the AgentRPC gateway.

  Args:
    endpoint: The gRPC target, such as `mirror.example.com:443`.
    session_id: The hosted session's ID.
    secure: Whether the endpoint uses TLS.

  Returns:
    The page URL, omitting the scheme's default port.
  """
  return session_url(web_origin(endpoint, secure=secure), session_id)


def _web_command(context: commands.CommandContext, args: str) -> actions.Action:
  del args
  if not context.remote:
    return actions.ShowText(
      'Only hosted sessions have a Mirror web page. '
      f'Conversation ID: {context.conversation_id}',
      style='ui.warning',
    )
  return actions.RoutedExtensionAction(
    SessionLinkExtension.id(), OpenWeb(context.conversation_id)
  )


def _text(style: str, label: str) -> containers.Window:
  return containers.Window(
    controls.FormattedTextControl([(style, label)]),
    dont_extend_height=True,
    dont_extend_width=True,
  )


class _Clickable(containers.Container):
  """Route clicks on a window's cells to `handler`.

  Extension output is hidden from layout traversal, and a `Window` ignores
  mouse events for windows outside it, so the handler is installed directly.
  """

  def __init__(
    self, content: containers.Window, handler: _mouse_handlers.MouseHandler
  ) -> None:
    self._content = content
    self._handler = handler

  @override
  def reset(self) -> None:
    self._content.reset()

  @override
  def preferred_width(self, max_available_width: int) -> dimension.Dimension:
    return self._content.preferred_width(max_available_width)

  @override
  def preferred_height(
    self, width: int, max_available_height: int
  ) -> dimension.Dimension:
    return self._content.preferred_height(width, max_available_height)

  @override
  def write_to_screen(
    self,
    screen: _screen.Screen,
    mouse_handlers: _mouse_handlers.MouseHandlers,
    write_position: _screen.WritePosition,
    parent_style: str,
    erase_bg: bool,
    z_index: int | None,
  ) -> None:
    self._content.write_to_screen(
      screen, mouse_handlers, write_position, parent_style, erase_bg, z_index
    )
    mouse_handlers.set_mouse_handler_for_range(
      write_position.xpos,
      write_position.xpos + write_position.width,
      write_position.ypos,
      write_position.ypos + write_position.height,
      self._handler,
    )

  @override
  def get_children(self) -> list[containers.Container]:
    return [self._content]


class SessionLinkExtension(extension.Extension[_State, OpenWeb, None]):
  """Show the session ID in the status bar and link hosted sessions to web."""

  def __init__(
    self, open_url: Callable[[str], Awaitable[None]] = terminal.open_url
  ) -> None:
    self._open_url = open_url

  @classmethod
  @override
  def id(cls) -> str:
    return 'core/session-link'

  @staticmethod
  @override
  def initial_state() -> _State:
    return _State()

  @override
  def slash_commands(self) -> tuple[commands.CommandDef, ...]:
    return (
      commands.CommandDef(
        'web',
        'Open this hosted session in Mirror web',
        _web_command,
        available_when_busy=True,
      ),
    )

  @override
  def reduce(
    self, global_state: state.State, action: actions.Action
  ) -> tuple[_State, tuple[extension.PrivateEffect[OpenWeb], ...]]:
    current = self.require_state(global_state)
    if (
      not isinstance(action, actions.RoutedExtensionAction)
      or action.extension_id != self.id()
    ):
      return current, ()
    match action.action:
      case OpenWeb():
        return current, (self.effect(action.action),)
      case ServerConfigured(origin):
        return _State(web_origin=origin), ()
      case _:
        return current, ()

  @override
  async def on_config(self, context: extension.Context) -> None:
    if context.remote_endpoint is not None:
      self.dispatch(
        context,
        ServerConfigured(
          web_origin(context.remote_endpoint, secure=context.remote_secure)
        ),
      )

  @override
  async def handle_effect(
    self, context: extension.Context, effect: OpenWeb
  ) -> None:
    if context.remote_endpoint is None:
      context.dispatch(
        actions.ShowText(
          'No Mirror server is configured for this session. '
          f'Session ID: {effect.session_id}',
          style='ui.warning',
        )
      )
      return
    url = web_url(
      context.remote_endpoint, effect.session_id, secure=context.remote_secure
    )
    context.dispatch(actions.ShowText(f'Session {effect.session_id}\n{url}'))
    try:
      await self._open_url(url)
    except (OSError, webbrowser.Error):
      context.dispatch(
        actions.ShowText(
          f'Could not open a browser. Open {url} manually.',
          style='ui.warning',
        )
      )

  @override
  def render_region(self) -> extension.RenderRegion:
    return extension.RenderRegion.STATUS_RIGHT

  @override
  def render_order(self) -> int:
    return 90

  @override
  def cache_key(
    self, global_state: state.State, extension_state: _State
  ) -> object:
    return (
      global_state.session.conversation_id,
      global_state.session.remote,
      global_state.session.pending_switch is None,
      extension_state.web_origin,
    )

  @override
  def render(
    self, global_state: state.State, extension_state: _State
  ) -> containers.AnyContainer | None:
    session = global_state.session
    if session.pending_switch is not None or not session.conversation_id:
      return None
    label = f'session {session.conversation_id[:_SHORT_ID_LENGTH]}'
    origin = extension_state.web_origin
    if not session.remote or origin is None:
      return _text('class:footer', label)
    return _Clickable(
      _text('class:footer underline', label),
      self._opener(session_url(origin, session.conversation_id)),
    )

  def _opener(self, url: str) -> _mouse_handlers.MouseHandler:
    def click(
      event: mouse_events.MouseEvent,
    ) -> key_bindings.NotImplementedOrNone:
      if (
        event.event_type is not mouse_events.MouseEventType.MOUSE_UP
        or event.button is not mouse_events.MouseButton.LEFT
      ):
        return NotImplemented
      application.get_app().create_background_task(self._open_or_bell(url))
      return None

    return click

  async def _open_or_bell(self, url: str) -> None:
    try:
      await self._open_url(url)
    except (OSError, webbrowser.Error):
      application.get_app().output.bell()
