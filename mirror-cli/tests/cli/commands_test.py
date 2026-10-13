"""Unit tests for the built-in slash-command registry."""

from __future__ import annotations

from mirror.cli import actions, commands, rendering


def _context(*, busy: bool = False) -> commands.CommandContext:
  return commands.CommandContext(
    model='gpt-test',
    reasoning_effort='medium',
    conversation_id='conv-abc',
    busy=busy,
    cwd='/tmp/project',
  )


def test_path_guard() -> None:
  """`/Users/…` is not a command; bare `/name` is."""
  assert commands.looks_like_slash_command('/help')
  assert commands.looks_like_slash_command('/')
  assert not commands.looks_like_slash_command('/Users/yoni/code')
  assert not commands.looks_like_slash_command('help')
  assert commands.dispatch('/Users/yoni/code', _context()) is None


def test_menu_commands() -> None:
  assert [c.name for c in commands.menu_commands('/') or ()] == [
    c.name for c in commands.COMMANDS
  ]
  assert [c.name for c in commands.menu_commands('/he') or ()] == ['help']
  assert {c.name for c in commands.menu_commands('/ex') or ()} == {
    'quit',
    'extensions',
    'extensions-add',
    'extensions-update',
    'extensions-remove',
  }
  assert (
    commands.dispatch('/extensions', _context(busy=True))
    == actions.ShowExtensions()
  )
  assert commands.menu_commands('/help ') is None  # args started
  assert commands.menu_commands('/Users/x') is None  # path-like
  assert commands.menu_commands('/he\nlp') is None  # multiline


def test_alias_override_matches_dispatch_completion_and_help() -> None:
  original = commands.command_named('status')
  assert original is not None
  inspect = commands.CommandDef(
    'inspect',
    'Extension inspector',
    lambda _, __: actions.ShowText('extension'),
    aliases=('status',),
    extension_id='test/inspector',
  )
  available = commands.resolve_commands((*commands.COMMANDS, inspect))

  assert commands.dispatch(
    '/status', _context(), available
  ) == actions.ShowText('extension')
  assert [
    command.name
    for command in commands.menu_commands('/status', available) or ()
  ] == ['inspect']
  help_text = commands.dispatch('/help', _context(), available)
  assert isinstance(help_text, actions.ShowText)
  assert original.description not in help_text.text
  assert '/inspect' in help_text.text
  assert 'aliases: /status' in help_text.text


def test_namespaced_commands_retain_unshadowed_aliases() -> None:
  first = commands.CommandDef(
    'status',
    'First status',
    lambda _, __: actions.ShowText('first'),
    aliases=('first-status',),
    extension_id='test/first',
  )
  second = commands.CommandDef(
    'status',
    'Second status',
    lambda _, __: actions.ShowText('second'),
    extension_id='test/second',
  )
  available = commands.resolve_commands((first, second))

  assert [(command.extension_id, command.names()) for command in available] == [
    ('test/first', ('first-status',)),
    ('test/second', ('status',)),
  ]
  assert commands.dispatch(
    '/status', _context(), available
  ) == actions.ShowText('second')
  assert commands.dispatch(
    '/first-status', _context(), available
  ) == actions.ShowText('first')
  assert first.name == 'status'
  assert commands.resolve_commands((first,)) == (first,)


def test_command_metadata() -> None:
  quit_cmd = next(c for c in commands.COMMANDS if c.name == 'quit')
  assert quit_cmd.matching_name('ex') == 'exit'
  assert quit_cmd.matching_name('nope') is None
  assert quit_cmd.accepts_arguments is False
  assert quit_cmd.is_available(busy=False) is True
  assert quit_cmd.is_available(busy=True) is True
  clear = next(c for c in commands.COMMANDS if c.name == 'clear')
  assert clear.is_available(busy=True) is False


def test_command_may_return_no_action() -> None:
  def handle(_: commands.CommandContext, args: str) -> None:
    del args

  command = commands.CommandDef('noop', 'Do nothing', handle)

  assert commands.dispatch('/noop', _context(), (command,)) is None


def test_argument_command_dispatch(monkeypatch) -> None:
  received: list[str] = []

  def handle(_: commands.CommandContext, args: str) -> actions.Action:
    received.append(args)
    return actions.ShowText(args)

  rename = commands.CommandDef(
    'rename',
    'Rename the conversation',
    handle,
    arguments=commands.RequiredArguments('<title>'),
  )
  monkeypatch.setattr(commands, 'COMMANDS', (rename,))
  outcome = commands.dispatch('/rename Release prep', _context())
  assert isinstance(outcome, actions.ShowText)
  assert received == ['Release prep']
  assert rename.accepts_arguments is True


def test_dispatch_outcomes() -> None:
  help_text = commands.dispatch('/help', _context())
  assert isinstance(help_text, actions.ShowText)
  assert '/status' in help_text.text
  assert '/permissions' not in help_text.text

  status = commands.dispatch('/status', _context())
  assert isinstance(status, actions.ShowText)
  status_text = rendering.strip_ansi(
    rendering.renderable_to_ansi(status.text, 80)
  )
  assert 'gpt-test' in status_text and 'conv-abc' in status_text

  assert isinstance(commands.dispatch('/quit', _context()), actions.QuitApp)
  assert isinstance(commands.dispatch('/exit', _context()), actions.QuitApp)
  assert isinstance(
    commands.dispatch('/clear', _context()), actions.ClearScreen
  )

  unknown = commands.dispatch('/nope', _context())
  assert isinstance(unknown, actions.ShowText) and unknown.style == 'ui.error'

  args = commands.dispatch('/status extra', _context())
  assert isinstance(args, actions.ShowText)
  assert 'does not accept arguments' in args.text


def test_session_command_outcomes() -> None:
  resume = next(c for c in commands.COMMANDS if c.name == 'resume')
  assert resume.accepts_arguments is True
  assert resume.prompt_hint == '[reference]'
  assert resume.arguments_are_optional
  assert resume.description == 'Choose a conversation'
  assert commands.dispatch('/resume abc123', _context()) == (
    actions.ResumeSession('abc123')
  )
  assert commands.dispatch('/resume', _context()) == actions.ResumeSession()
  assert commands.dispatch('/new', _context()) == actions.NewSession()
  assert commands.dispatch('/worktree', _context()) == actions.CreateWorktree()
  assert commands.dispatch('/worktree release-prep', _context()) == (
    actions.CreateWorktree('release-prep')
  )
  assert commands.command_named('remote') is None
  assert commands.dispatch('/drop', _context()) == actions.Drop()
  assert 'bug' not in {command.name for command in commands.COMMANDS}
  assert commands.dispatch('/rename Useful title', _context()) == (
    actions.RenameWorktree('Useful title')
  )
  assert commands.command_named('permissions') is None
  assert commands.menu_commands('/permissions') == ()
  assert commands.dispatch('/permissions', _context()) == actions.ShowText(
    'Unknown command: /permissions. Type /help.', style='ui.error'
  )
  for text, expected in (('/rename', 'Usage: /rename <alias>'),):
    outcome = commands.dispatch(text, _context())
    assert isinstance(outcome, actions.ShowText)
    assert outcome.style == 'ui.warning'
    assert expected in outcome.text


def test_busy_gates_commands() -> None:
  blocked = commands.dispatch('/clear', _context(busy=True))
  assert isinstance(blocked, actions.ShowText)
  assert 'unavailable' in blocked.text
  assert isinstance(
    commands.dispatch('/quit', _context(busy=True)), actions.QuitApp
  )
  assert commands.dispatch('/permissions', _context(busy=True)) == (
    actions.ShowText(
      'Unknown command: /permissions. Type /help.', style='ui.error'
    )
  )
  assert (
    commands.dispatch('/worktree', _context(busy=True))
    == actions.CreateWorktree()
  )
  for text in (
    '/resume c1',
    '/new',
    '/drop',
  ):
    outcome = commands.dispatch(text, _context(busy=True))
    assert isinstance(outcome, actions.ShowText)
    assert 'unavailable' in outcome.text
  assert commands.dispatch('/rename title', _context(busy=True)) == (
    actions.RenameWorktree('title')
  )


def test_compact_command_is_idle_only_and_takes_no_arguments() -> None:
  assert (
    commands.dispatch('/compact', _context()) == actions.CompactConversation()
  )
  assert commands.dispatch('/compact', _context(busy=True)) == actions.ShowText(
    '/compact is unavailable while a turn is in progress.', style='ui.warning'
  )
  assert commands.dispatch('/compact extra', _context()) == actions.ShowText(
    '/compact does not accept arguments.', style='ui.warning'
  )
