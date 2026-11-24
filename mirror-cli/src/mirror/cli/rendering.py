"""Rich and ANSI rendering for conversation entries and tool calls."""

from __future__ import annotations

import copy
import dataclasses
import difflib
import functools
import io
import itertools
import json
import os
import re
import time
from collections.abc import Collection, Mapping, Sequence
from typing import Any, ClassVar, cast, override

import pydantic
from markdown_it import token as _markdown_token
from mirror.protocol import agent_pb2
from rich import box as _box
from rich import cells as _cells
from rich import console as _console
from rich import markdown as _markdown
from rich import measure as _measure
from rich import padding as _padding
from rich import segment as _segment
from rich import style as _style
from rich import syntax as _syntax
from rich import table as _table
from rich import text as _text

from mirror import distribution
from mirror.cli import code_rendering, prose_preview, themes
from mirror.cli import quotes as _quotes
from mirror.cli.composer import skill_completion as _skill_completion

# The glyph the input line uses, reused as the user-message prefix.
PROMPT = '›'
# Gutter glyph marking an assistant turn.
ASSISTANT_ICON = '●'
_RESULT_PREVIEW_MAX_LINES = 3
_PREVIEW_MAX_CHARS = 1024
_COMMAND_PREVIEW_MAX_CHARS = 240
_TOOL_PREVIEW_MAX_CHARS = 16_384
_TOOL_PREVIEW_MAX_LINES = 200
_PARTIAL_ARGUMENTS = pydantic.TypeAdapter(dict[str, Any])
# Reserve two icon cells (including Crossfire) and one separator cell.
GUTTER_WIDTH = 3
_HUNK_RE = re.compile(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@')
# Rich's repr highlighter also scans for non-string tokens and mishandles
# closing quotes after escaped backslashes. Limit label highlighting to quotes;
# skip escaped characters, and don't treat apostrophes inside words as quotes.
_TOOL_STRINGS_RE = re.compile(
  r"""\\.|(?P<accent>"(?:\\.|[^"\\])*"|\$'(?:\\.|[^'\\])*'|(?<!\w)'[^']*')""",
  re.DOTALL,
)

# Full ECMA-48 escape coverage. Conversation history can contain terminal
# output captured by tools; replaying it must never execute those controls in
# the user's terminal.
_ANSI_ESCAPE_RE = re.compile(
  r'\x1b'
  r'(?:'
  r'\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]'
  r'|\][\s\S]*?(?:\x07|\x1b\\)'
  r'|[PX^_][\s\S]*?(?:\x1b\\)'
  r'|[\x20-\x2f]+[\x30-\x7e]'
  r'|[\x30-\x7e]'
  r')'
  r'|\x9b[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]'
  r'|\x9d[\s\S]*?(?:\x07|\x9c)'
  r'|[\x80-\x9f]',
  re.DOTALL,
)
_HAS_TERMINAL_CONTROL = re.compile(r'[\x00-\x08\x0b-\x1f\x7f-\x9f]')
_SAFE_SGR_RE = re.compile(r'\x1b\[[0-9:;]*m')
_SAFE_HYPERLINK_RE = re.compile(
  r'\x1b\]8;[^\x00-\x1f\x7f-\x9f]*(?:\x07|\x1b\\)', re.DOTALL
)

# Local bash ends every result with its wall time, and only a timed-out call
# fails. Other bash results start with `Exit code:`, after `Timed out.` for a
# timeout.
_BASH_COMPLETED_LINE_RE = re.compile(r'(?:\A|\n)\[Completed in [\d.]+s\]\Z')
_BASH_TIMEOUT_RE = re.compile(r'\ACommand timed out after \d+ms(?:\n|$)')
_BASH_EXIT_STATUS_RE = re.compile(
  r'\A(Timed out\.\n)?Exit code: (-?\d+)(?:\n|$)'
)

# Background tints for changed lines; syntax highlighting supplies the fg.
_DIFF_ADD_BG = 'ui.addition'
_DIFF_DEL_BG = 'ui.deletion'


def icon_prefix(icon: str) -> str:
  """Pad a transcript icon to the shared terminal-cell gutter."""
  return _cells.set_cell_size(icon, GUTTER_WIDTH)


def activity_kind(name: str) -> str:
  return 'agent' if name == 'subagent' else 'tool'


def _one_line(value: object) -> str:
  return ' '.join(str(value).split())


def _argument_summary(arguments: Mapping[str, object]) -> str:
  return ', '.join(_one_line(value) for value in arguments.values())


def _bounded_preview(value: str, limit: int = _PREVIEW_MAX_CHARS) -> str:
  """Bound Rich's measurement work; terminal width controls final clipping."""
  if len(value) <= limit:
    return value
  return value[: limit - 1].rstrip() + '…'


def command_preview(command: str) -> str:
  """A shell command on one line, short enough to wrap onto a few rows."""
  return _bounded_preview(_one_line(command), _COMMAND_PREVIEW_MAX_CHARS)


def _result_preview(result: str) -> str:
  text = _bounded_preview(result).strip()
  if not text:
    return '(no output)'
  lines = text.splitlines()
  shown = lines[:_RESULT_PREVIEW_MAX_LINES]
  preview = '\n'.join(shown)
  truncated = len(lines) > len(shown)
  if truncated:
    remaining = len(lines) - len(shown)
    preview += f'\n… (+{remaining} more lines)' if remaining else '\n…'
  return _bounded_preview(preview)


def compact_args(arguments_json: str, *, tool_name: str = '') -> str:
  """Summarize tool arguments; the renderer clips to available terminal cells."""
  try:
    # Preview long commands and paths before their closing quote arrives.
    arguments = _PARTIAL_ARGUMENTS.validate_json(
      arguments_json[:_TOOL_PREVIEW_MAX_CHARS] or '{}',
      experimental_allow_partial='trailing-strings',
    )
  except pydantic.ValidationError:
    return ''
  arguments.pop('content', None)
  if tool_name in {'read', 'edit', 'write'}:
    arguments = {'path': arguments.get('file_path', '')}
  elif tool_name in {'grep', 'find', 'glob'}:
    pattern = arguments.get('pattern', '')
    path = arguments.get('path', '.')
    arguments = {'search': f'{pattern} in {path}'}
  elif tool_name == 'ls':
    arguments = {'path': arguments.get('path', '.')}
  elif tool_name == 'bash':
    arguments = {'command': arguments.get('command', '')}
  elif tool_name == 'mcp_call' and isinstance(
    calls := arguments.get('calls'), list
  ):
    arguments = {
      'calls': ', '.join(
        name.removeprefix('mcp__').replace('__', '.', 1)
        for call in cast(list[pydantic.JsonValue], calls)
        if isinstance(call, dict) and isinstance(name := call.get('name'), str)
      )
    }
  if 'prompt' in arguments and len(arguments) == 1:
    summary = _one_line(arguments['prompt'])
  else:
    summary = _argument_summary(arguments)
  return _bounded_preview(summary)


def _readable_argument(value: str) -> bool:
  return all(char.isprintable() or char in '\n\t' for char in value)


def _approval_file(
  arguments: Mapping[str, object], *, tool_name: str
) -> _console.RenderableType | None:
  path = arguments.get('file_path')
  old = arguments.get('old_string') if tool_name == 'edit' else ''
  new = arguments.get('new_string' if tool_name == 'edit' else 'content')
  if (
    tool_name not in {'edit', 'write'}
    or not isinstance(path, str)
    or not path.isprintable()
    or not isinstance(old, str)
    or not isinstance(new, str)
    or not _readable_argument(old)
    or not _readable_argument(new)
  ):
    return None
  before, after = old.splitlines(keepends=True), new.splitlines(keepends=True)
  diff = ''.join(
    line if line.endswith('\n') else f'{line}\n'
    for line in difflib.unified_diff(
      before, after, fromfile=path, tofile=path, n=max(len(before), len(after))
    )
  )
  if not diff and tool_name == 'edit':
    return None
  consumed = (
    {'file_path', 'old_string', 'new_string'}
    if tool_name == 'edit'
    else {'file_path', 'content'}
  )
  remaining = {
    key: value for key, value in arguments.items() if key not in consumed
  }
  body: _console.RenderableType = (
    DiffPanel(diff, show_file_headers=False)
    if diff
    else _text.Text('(empty file)', style='ui.muted')
  )
  if remaining:
    body = _console.Group(body, approval_arguments(remaining))
  return tool_block(f'{tool_name} {path}', body, wrap=True)


def _approval_excerpt(
  name: str, arguments_json: str
) -> _console.RenderableType:
  """Avoid parsing, diffing or highlighting oversized argument payloads."""
  excerpt = '\n'.join(arguments_json[:4096].split('\n')[:40])
  body = _text.Text(
    f'Preview limited · raw excerpt: {len(excerpt):,}/{len(arguments_json):,} characters. '
    'Approval uses the full call.\n',
    style='ui.warning',
  )
  body.append(
    excerpt if _readable_argument(excerpt) else json.dumps(excerpt),
    style='ui.text',
  )
  return tool_block(name, body, wrap=True)


def approval_body(
  arguments_json: str, *, tool_name: str
) -> _console.RenderableType:
  """Show proposed arguments with bounded work and explicit overflow notices."""
  name = strip_ansi(tool_name).removeprefix('mcp__').replace('__', ' · ')
  if (
    len(arguments_json) > _TOOL_PREVIEW_MAX_CHARS
    or arguments_json.count('\n') > _TOOL_PREVIEW_MAX_LINES
  ):
    return _approval_excerpt(name, arguments_json)
  try:
    arguments = _PARTIAL_ARGUMENTS.validate_json(arguments_json or '{}')
  except pydantic.ValidationError:
    body = _text.Text(json.dumps(arguments_json))
  else:
    formatted = json.dumps(arguments, indent=2)
    if (
      len(formatted) > _TOOL_PREVIEW_MAX_CHARS
      or formatted.count('\n') + formatted.count('\\n')
      > _TOOL_PREVIEW_MAX_LINES
    ):
      return _approval_excerpt(name, arguments_json)
    if (
      file_preview := _approval_file(arguments, tool_name=tool_name)
    ) is not None:
      return file_preview
    command = arguments.get('command')
    if (
      tool_name == 'bash'
      and isinstance(command, str)
      and command
      and _readable_argument(command)
    ):
      arguments.pop('command')
      return CommandApproval(name, command, arguments)
    body = approval_arguments(arguments)
  return tool_block(name, body, wrap=True)


@dataclasses.dataclass(frozen=True)
class CommandApproval:
  """A proposed shell command, highlighted for the palette it renders in."""

  name: str
  command: str
  arguments: Mapping[str, object]

  def __rich_console__(
    self, console: _console.Console, options: _console.ConsoleOptions
  ) -> _console.RenderResult:
    del options  # Unused.
    code = _syntax.Syntax(
      self.command,
      'bash',
      theme=themes.syntax_theme(console, 'code'),
      background_color='default',
      word_wrap=True,
    )
    arguments = approval_arguments(self.arguments) if self.arguments else None
    if self.command.isprintable():
      header = _tool_title(self.name)
      header.append(' ')
      header.append_text(code.highlight(self.command)[: len(self.command)])
      yield tool_block(header, arguments, wrap=True)
      return
    body = _console.Group(code, arguments) if arguments else code
    yield tool_block(self.name, body, wrap=True)


def approval_arguments(arguments: Mapping[str, object]) -> _text.Text:
  """Keep named arguments readable without interpreting markup or terminal controls."""
  text = _text.Text()
  for key, value in arguments.items():
    if text:
      text.append('\n')
    label = key if key.isprintable() else json.dumps(key)
    text.append(f'{label}: ', style='ui.muted')
    display = json.dumps(value, indent=2)
    if isinstance(value, str) and value and _readable_argument(value):
      if '\n' in value:
        ending = '' if value.endswith('\n') else '(no final newline)'
        display = f'{ending}\n{value}'
      elif (
        value.isprintable()
        and value == value.strip()
        and value not in {'true', 'false', 'null', 'NaN', 'Infinity'}
        and value[0] not in '-0123456789[{"'
      ):
        display = value
    text.append(display, style='ui.text')
  return text


def tool_operation(arguments_json: str) -> str:
  """The structured action discriminator of a multiplexed tool, if present."""
  if len(arguments_json) > _TOOL_PREVIEW_MAX_CHARS:
    return ''
  try:
    arguments = _PARTIAL_ARGUMENTS.validate_json(arguments_json or '{}')
  except pydantic.ValidationError:
    return ''
  action = arguments.get('action')
  return _one_line(strip_ansi(action)) if isinstance(action, str) else ''


def format_window(window: int) -> str:
  """A compact token-window size, e.g. 128K, 1M, 1.5M."""
  if window >= 1_000_000:
    megs = window / 1_000_000
    return f'{megs:.1f}M'.replace('.0M', 'M')
  return f'{window / 1_000:.0f}K'


def next_status_label(message: agent_pb2.ChatRecord) -> str:
  """The activity label for the wait that follows `message`."""
  if (
    message.WhichOneof('message') == 'assistant'
    and message.assistant.tool_calls
  ):
    return 'running tool…'
  # An assistant message can end without ending the turn.
  return 'thinking…'


class _InteractiveConsole(_console.Console):
  """Opt in to controls that only the owned transcript can activate."""


def ansi_console(
  buf: io.StringIO,
  width: int,
  *,
  theme_name: str | themes.Theme = 'dark',
  interactive: bool = False,
) -> _console.Console:
  """A console that renders to ANSI in `buf`, whatever the real output is.

  Escapes are always written: the live pane parses them itself, and scrollback
  is printed through the same pre-render so both keep one notion of spacing.
  """
  console_type = _InteractiveConsole if interactive else _console.Console
  return console_type(
    file=buf,
    force_terminal=True,
    no_color=theme_name == 'mono',
    color_system='standard' if theme_name == 'terminal' else 'truecolor',
    theme=themes.rich_theme(theme_name),
    style='ui.text',
    width=width,
    # Rich otherwise ignores an explicit width when TERM=dumb.
    height=25,
  )


def strip_ansi(text: str) -> str:
  """Remove terminal escape and C0/C1 control sequences from plain text."""
  if not text or not _HAS_TERMINAL_CONTROL.search(text):
    return text
  without_escapes = _ANSI_ESCAPE_RE.sub('', text)
  return _HAS_TERMINAL_CONTROL.sub('', without_escapes)


def sanitize_ansi(text: str) -> str:
  """Keep Rich's formatting escapes while removing terminal commands."""
  if not text or not _HAS_TERMINAL_CONTROL.search(text):
    return text
  parts: list[str] = []
  position = 0
  for match in _ANSI_ESCAPE_RE.finditer(text):
    parts.append(_HAS_TERMINAL_CONTROL.sub('', text[position : match.start()]))
    sequence = match.group()
    if _SAFE_SGR_RE.fullmatch(sequence) or _SAFE_HYPERLINK_RE.fullmatch(
      sequence
    ):
      parts.append(sequence)
    position = match.end()
  parts.append(_HAS_TERMINAL_CONTROL.sub('', text[position:]))
  return ''.join(parts)


def renderable_to_ansi(
  renderable: _console.RenderableType,
  width: int,
  *,
  theme_name: str | themes.Theme = 'dark',
) -> str:
  """Render Rich output as terminal ANSI at a fixed width."""
  buf = io.StringIO()
  ansi_console(buf, width, theme_name=theme_name).print(renderable, end='')
  return sanitize_ansi(buf.getvalue())


def render_blocks(
  renderables: Sequence[_console.RenderableType],
  width: int,
  *,
  theme_name: str | themes.Theme = 'dark',
) -> str:
  """Render blocks to ANSI, each separated by one blank line.

  Pre-rendering to ANSI and joining manually is the single source of truth
  for block spacing.  Rich's `print(*args, sep='\\n\\n')` does not apply
  `sep` consistently across renderable types (Table.grid absorbs it into
  its trailing newline), so every rendering path goes through here instead.
  """
  return '\n\n'.join(
    renderable_to_ansi(r, width, theme_name=theme_name).rstrip()
    for r in renderables
  )


def render_banner(
  model: str,
  cwd: str | os.PathLike[str] | None = None,
) -> _console.RenderableType:
  """Quiet welcome facts shared by the launch presentation."""
  home = os.path.expanduser('~')
  cwd = os.fspath(cwd) if cwd is not None else os.getcwd()
  # Guard the separator so `/home/foobar` under `$HOME=/home/foo` isn't
  # mangled into `~bar`.
  in_home = cwd == home or cwd.startswith(home + os.sep)
  directory = strip_ansi(f'~{cwd[len(home) :]}' if in_home else cwd)
  model = strip_ansi(model)

  info = _text.Text(
    f'Reflection mirror v{distribution.installed().version}', style='ui.muted'
  )
  facts = _table.Table.grid(padding=(0, 2))
  facts.add_column(style='ui.muted', no_wrap=True)
  facts.add_column(style='ui.muted', overflow='fold')
  facts.add_row('model', _text.Text(model))
  facts.add_row('directory', _text.Text(directory))

  return _console.Group(info, facts)


def render_key_setup(provider: str, url: str) -> _text.Text:
  """Actionable sign-in steps, without competing launch branding or metadata."""
  welcome = _text.Text()
  welcome.append(f'Connect to {provider}\n', style='ui.info.bold')
  welcome.append(
    'Add an API key to start coding with Mirror.\n\n', style='default'
  )
  welcome.append('1  ', style='ui.info.bold')
  welcome.append('Create a key\n   ', style='bold')
  link = _text.Text(url, style='ui.info')
  link.stylize(f'underline link {url}')
  welcome.append_text(link)
  welcome.append('\n\n2  ', style='ui.info.bold')
  welcome.append('Paste your key below', style='bold')
  welcome.append(' — input is hidden.', style='ui.muted')
  return welcome


@dataclasses.dataclass(frozen=True)
class UserText:
  """One source for the full user message and its viewport summary."""

  message: agent_pb2.UserMessage
  skill_names: Collection[str] = ()

  @property
  def text(self) -> str:
    """The message as the composer showed it, with quotes as chips."""
    return _quotes.display(self.message)[0]

  def body(self) -> _text.Text:
    """The message text, with quotes styled as the composer's chips."""
    text = _text.Text(overflow='fold')
    shown: tuple[_quotes.Quote, ...] = ()
    parts = self.message.parts or [
      agent_pb2.UserMessagePart(text=self.message.content.strip())
    ]
    for part in parts:
      if part.WhichOneof('part') == 'quoted_selection':
        # Match the composer chip the user saw before sending.
        quote, shown = _quotes.add(part.quoted_selection, shown)
        rail, preview, detail = _quotes.split_chip(quote.label)
        text.append(rail, 'ui.quote-chip-rail')
        text.append(preview, 'ui.quote.italic')
        text.append(detail, 'ui.quote-chip-detail')
      else:
        own = _text.Text(strip_ansi(part.text))
        _skill_completion.highlight_mentions(own, self.skill_names)
        text.append_text(own)
    return text

  def __rich__(self) -> _padding.Padding:
    grid = _table.Table.grid(padding=(0, 1), expand=True)
    grid.add_column(width=GUTTER_WIDTH - 1)
    grid.add_column(ratio=1, overflow='fold')
    grid.add_row(_text.Text(PROMPT, style='bold dim'), self.body())
    # Set the foreground too so the dark fill stays readable on light terminals.
    return _padding.Padding(grid, (1, 1, 1, 0), style='ui.message')


def user_text(
  message: agent_pb2.UserMessage, skill_names: Collection[str] = ()
) -> UserText:
  """A full-width shaded prompt, with wrapped lines aligned to its text."""
  return UserText(message, skill_names)


class _TranscriptTable(_markdown.TableElement):
  """Box each cell so wrapped table rows remain visually distinct."""

  @override
  def __rich_console__(
    self, console: _console.Console, options: _console.ConsoleOptions
  ) -> _console.RenderResult:
    for renderable in super().__rich_console__(console, options):
      if isinstance(renderable, _table.Table):
        renderable.box = _box.SQUARE
        renderable.show_edge = True
        renderable.show_lines = True
        renderable.padding = (0, 1)
        renderable.pad_edge = True
        renderable.collapse_padding = False
        renderable.border_style = 'markdown.table.border'
      yield renderable


class _TranscriptHeading(_markdown.Heading):
  """Keep headings aligned with the answer rather than centering them."""

  LEVEL_ALIGN: ClassVar[dict[str, _console.JustifyMethod]] = {
    f'h{level}': 'left' for level in range(1, 7)
  }


class _CodePanel:
  """A shaded code surface with a full-height rail, shared by code and edits."""

  def __init__(
    self, body: _console.RenderableType, *, copy_text: str = ''
  ) -> None:
    self.body = body
    self.copy_text = copy_text

  def __rich_console__(
    self, console: _console.Console, options: _console.ConsoleOptions
  ) -> _console.RenderResult:
    if options.max_width < 2:
      yield self.body
      return
    if isinstance(self.body, code_rendering.Syntax) and self.body.retained:
      background = console.get_style('ui.code')
      blank = [_segment.Segment(' ' * (options.max_width - 1), background)]
      lines = itertools.chain(
        itertools.repeat(blank, 2 if self.copy_text else 1),
        _segment.Segment.split_lines(
          console.render(self.body, options.update(width=options.max_width - 1))
        ),
        (blank,),
      )
    else:
      lines = console.render_lines(
        _padding.Padding(
          self.body,
          (2 if self.copy_text else 1, 0, 1, 0),
          style='ui.code',
          expand=True,
        ),
        options.update(width=options.max_width - 1),
      )
    rail = console.get_style('ui.code-rail')
    interactive = (
      isinstance(console, _InteractiveConsole)
      and self.copy_text
      and options.max_width >= 3
    )
    metadata = _style.Style(meta={'code_block': True}) if interactive else None
    for index, line in enumerate(lines):
      rail_style = rail + metadata
      if interactive and index == 0:
        rail_style += _style.Style(meta={'copy_code': self.copy_text})
      yield _segment.Segment('┃', rail_style)
      if interactive and index == 1:
        # Inset the control without ever covering source characters.
        width = options.max_width - 2
        button_width = min(3, width)
        background = console.get_style('ui.code')
        yield _segment.Segment(
          ' ' * (width - button_width),
          background + _style.Style(meta={'code_hint': True}),
        )
        yield _segment.Segment(
          ' ' * button_width,
          background + _style.Style(meta={'code_action': True}),
        )
        yield _segment.Segment(' ', background + metadata)
      else:
        yield from _segment.Segment.apply_style(line, metadata)
      yield _segment.Segment.line()


class _TranscriptCodeBlock(_markdown.CodeBlock):
  """Set code on the shared panel without changing its syntax foregrounds."""

  rows: code_rendering.RowCache | None = None

  @classmethod
  @override
  def create(
    cls, markdown: _markdown.Markdown, token: _markdown_token.Token
  ) -> _TranscriptCodeBlock:
    block = cls(token.info.partition(' ')[0] or 'text', markdown.code_theme)
    if isinstance(markdown, _TranscriptMarkdown):
      block.rows = markdown.code_rows
    return block

  @override
  def __rich_console__(
    self, console: _console.Console, options: _console.ConsoleOptions
  ) -> _console.RenderResult:
    background = console.get_style('ui.code').bgcolor
    code = code_rendering.Syntax(
      str(self.text).rstrip(),
      self.lexer_name,
      rows=self.rows,
      theme=themes.syntax_theme(console, 'code'),
      background_color=background.name if background else 'default',
      padding=(0, 1) if options.max_width >= 6 else 0,
    )
    yield _CodePanel(code, copy_text=self.text.plain)


class _TranscriptQuote(_markdown.BlockQuote):
  """Keep quoted words intact beside an accent rail."""

  @override
  def __rich_console__(
    self, console: _console.Console, options: _console.ConsoleOptions
  ) -> _console.RenderResult:
    inset = min(2, max(0, options.max_width - 1))
    lines = console.render_lines(
      self.elements,
      options.update(width=max(1, options.max_width - inset)),
      style=self.style,
    )
    for line in lines:
      yield _segment.Segment('▌ '[:inset], console.get_style('ui.quote-rail'))
      yield from line
      yield _segment.Segment.line()


class _TranscriptListItem(_markdown.ListItem):
  """Include the source's period after ordered-list numbers."""

  @override
  def render_number(
    self,
    console: _console.Console,
    options: _console.ConsoleOptions,
    number: int,
    last_number: int,
  ) -> _console.RenderResult:
    width = min(len(str(last_number)) + 3, max(0, options.max_width - 1))
    lines = console.render_lines(
      self.elements,
      options.update(width=max(1, options.max_width - width)),
      style=self.style,
    )
    for index, line in enumerate(lines):
      label = f'{number}.'.rjust(width - 1) + ' ' if index == 0 else ' ' * width
      yield _segment.Segment(
        label[:width], console.get_style('markdown.item.number')
      )
      yield from line
      yield _segment.Segment.line()


class _StartTrimmed:
  """Align the answer glyph with content, not a list's leading blank row."""

  def __init__(self, body: _console.RenderableType) -> None:
    self.body = body

  def __rich_console__(
    self, console: _console.Console, options: _console.ConsoleOptions
  ) -> _console.RenderResult:
    started = False
    for line in console.render_lines(self.body, options, pad=False):
      if not started and all(
        not segment.text.strip()
        and not (segment.style and segment.style.bgcolor)
        for segment in line
      ):
        continue
      started = True
      yield from line
      yield _segment.Segment.line()


class _TranscriptMarkdown(_markdown.Markdown):
  """Shared Markdown presentation for streaming and settled transcript text."""

  code_rows: code_rendering.RowCache | None = None

  elements: ClassVar[dict[str, type[_markdown.MarkdownElement]]] = {
    **_markdown.Markdown.elements,
    'table_open': _TranscriptTable,
    'heading_open': _TranscriptHeading,
    'fence': _TranscriptCodeBlock,
    'code_block': _TranscriptCodeBlock,
    'blockquote_open': _TranscriptQuote,
    'list_item_open': _TranscriptListItem,
  }

  @override
  def __rich_console__(
    self, console: _console.Console, options: _console.ConsoleOptions
  ) -> _console.RenderResult:
    # markdown-it's maxNesting=20 silently discards the remaining parse input.
    if any(
      token.level >= 19 and token.type in ('list_item_open', 'blockquote_open')
      for token in self.parsed
    ):
      yield _text.Text(self.markup, style=self.style)
      return
    yield from super().__rich_console__(console, options)


@dataclasses.dataclass(frozen=True)
class MarkdownSource:
  """Parse only when Rich paints, not when a background record arrives."""

  content: str
  style: str = 'none'
  hyperlinks: bool = True
  justify: _console.JustifyMethod | None = None

  @functools.cached_property
  def parsed(self) -> _TranscriptMarkdown:
    return _TranscriptMarkdown(
      self.content,
      style=self.style,
      hyperlinks=self.hyperlinks,
      justify=self.justify,
    )

  def __rich_console__(
    self, console: _console.Console, options: _console.ConsoleOptions
  ) -> _console.RenderResult:
    yield self.parsed


class StreamingAnswer:
  """Retain completed Markdown blocks and parse only the changing final block."""

  def __init__(
    self, skill_names: Collection[str], *, hyperlinks: bool = True
  ) -> None:
    self._skill_names = frozenset(skill_names)
    self._hyperlinks = hyperlinks
    self._code_rows = code_rendering.RowCache()
    self._preview = prose_preview.ProsePreview()
    self._source = ''
    self._source_offset = 0
    self._truncated = False
    self._stable_source_len = 0
    self._stable: list[_console.RenderableType] = []
    self._parts: tuple[_console.RenderableType, ...] = ()

  def _append_fence(self, source: str) -> bool:
    if not self._parts or not isinstance(self._parts[-1], Answer):
      return False
    previous = self._parts[-1]
    markdown = previous.markdown
    if markdown is None or len(markdown.parsed) != 1:
      return False
    token = markdown.parsed[0]
    if (
      token.type != 'fence'
      or not token.info
      or not source.startswith(self._source)
    ):
      return False
    tail = source[self._stable_source_len :]
    opening, newline, code = previous.source.partition('\n')
    # Match only canonical, unindented, still-open fences. Closing markers,
    # normalization and edits go back through the Markdown parser.
    if (
      not newline
      or not opening.startswith(token.markup)
      or token.content != code
    ):
      return False
    added = tail[len(previous.source) :]
    last_line = previous.source.rfind('\n') + 1
    if (
      '\r' in added
      or '\0' in added
      or re.search(r'(?m)^ {0,3}' + re.escape(token.markup), tail[last_line:])
    ):
      return False
    updated = copy.copy(markdown)
    updated.markup = tail
    updated_token = copy.copy(token)
    updated_token.content = tail.partition('\n')[2]
    updated.parsed = [updated_token]
    answer = dataclasses.replace(previous, source=tail, markdown=updated)
    self._source = source
    self._parts = (*self._stable, answer)
    return True

  def update(
    self, source: str, *, source_offset: int = 0, preview: bool = False
  ) -> tuple[_console.RenderableType, ...]:
    """Render live text, optionally bounding prose while following the stream."""
    content, omitted = self._preview.update(source) if preview else (source, 0)
    if not source:
      self._preview.update('')
    return self._update(
      content, source_offset=source_offset + omitted, truncated=bool(omitted)
    )

  def _update(
    self, source: str, *, source_offset: int, truncated: bool
  ) -> tuple[_console.RenderableType, ...]:
    source = strip_ansi(source)
    if source_offset:
      source_offset += len(source) - len(source.lstrip())
    source = source.strip()
    if source_offset != self._source_offset or truncated != self._truncated:
      self._source_offset = source_offset
      self._truncated = truncated
      self._source = ''
      self._stable_source_len = 0
      self._stable.clear()
      self._parts = ()
    if source == self._source or self._append_fence(source):
      return self._parts
    previous_source = self._source[self._stable_source_len :]
    previous_tail = self._parts[-1] if self._parts else None
    previous_continuation = bool(self._stable)
    if not source.startswith(self._source[: self._stable_source_len]):
      self._stable_source_len = 0
      self._stable.clear()
      previous_tail = None
    self._source = source
    if not source:
      self._parts = ()
      return ()
    # Definitions can change links in any preceding block, including unresolved links.
    if re.search(r'(?m)^ {0,3}\[[^\n]+\]:', source):
      self._stable_source_len = 0
      self._stable.clear()
      previous_tail = None
    suffix = source[self._stable_source_len :]
    markdown = _TranscriptMarkdown(suffix, hyperlinks=self._hyperlinks)
    markdown.code_rows = self._code_rows
    boundaries = [
      (index, token.map[0])
      for index, token in enumerate(markdown.parsed)
      if token.level == 0 and token.nesting != -1 and token.map is not None
    ]
    # Keep invisible blocks and rule spacing with their neighboring content.
    boundaries = [
      (index, line)
      for position, (index, line) in enumerate(boundaries)
      if position == 0
      or (
        markdown.parsed[index].type != 'html_block'
        and markdown.parsed[boundaries[position - 1][0]].type
        not in ('html_block', 'hr')
      )
    ]
    if not boundaries:
      self._parts = tuple(self._stable)
      return self._parts
    lines = suffix.splitlines(keepends=True)
    offsets = list(itertools.accumulate(map(len, lines), initial=0))
    last_index, last_line = boundaries[-1]
    for position, (token_index, line) in enumerate(boundaries[:-1]):
      next_index, next_line = boundaries[position + 1]
      block_source = ''.join(lines[line:next_line])
      if (
        position == 0
        and previous_tail is not None
        and previous_continuation == bool(self._stable)
        and block_source.rstrip() == previous_source.rstrip()
      ):
        self._stable.append(previous_tail)
        continue
      block = copy.copy(markdown)
      block.markup = block_source
      block.parsed = markdown.parsed[token_index:next_index]
      answer = assistant_render(
        block_source,
        hyperlinks=self._hyperlinks,
        skill_names=self._skill_names,
        continuation=bool(self._stable),
        markdown=block,
        source_offset=source_offset + self._stable_source_len + offsets[line],
      )
      self._stable.append(
        dataclasses.replace(answer, truncated=True) if truncated else answer
      )
    self._stable_source_len += offsets[last_line]
    markdown.parsed = markdown.parsed[last_index:]
    markdown.markup = ''.join(lines[last_line:])
    tail = assistant_render(
      markdown.markup,
      hyperlinks=self._hyperlinks,
      skill_names=self._skill_names,
      continuation=bool(self._stable),
      markdown=markdown,
      source_offset=source_offset + self._stable_source_len,
    )
    if truncated:
      tail = dataclasses.replace(tail, truncated=True)
    self._parts = (*self._stable, tail)
    return self._parts


@dataclasses.dataclass(frozen=True)
class Answer:
  """An answer with independently wrapped Markdown parts and source anchors."""

  source: str
  hyperlinks: bool = True
  skill_names: frozenset[str] = frozenset()
  continuation: bool = False
  markdown: _TranscriptMarkdown | None = None
  source_offset: int = 0
  truncated: bool = False

  @functools.cached_property
  def parts(self) -> tuple[_console.RenderableType, ...]:
    """Split only at Markdown boundaries, never inside a list, table, or fence."""
    if self.markdown is not None:
      return (self,)
    return StreamingAnswer(self.skill_names, hyperlinks=self.hyperlinks).update(
      self.source, source_offset=self.source_offset
    ) or (self,)

  @functools.cached_property
  def body(self) -> _console.RenderableType:
    """Keep the canonical Rich presentation for non-virtualized output."""
    grid = _table.Table.grid(padding=(0, 1))
    grid.add_column(width=GUTTER_WIDTH - 1)
    # Rich's code-block wrapping inherits the column's overflow policy.
    grid.add_column(overflow='fold')
    content: _console.RenderableType = self.markdown or MarkdownSource(
      self.source, hyperlinks=self.hyperlinks
    )
    if self.truncated:
      content = _console.Group(_text.Text('…', style='ui.muted'), content)
    grid.add_row(
      _text.Text(
        '' if self.continuation else ASSISTANT_ICON,
        style='' if self.continuation else 'ui.accent',
      ),
      _StartTrimmed(content),
    )
    return _skill_completion.highlight_renderable(grid, self.skill_names)

  def __rich_console__(
    self, console: _console.Console, options: _console.ConsoleOptions
  ) -> _console.RenderResult:
    markdown = self.markdown
    # A top-level fence has fixed gutters. Avoid measuring it through a table
    # and rendering its entire contents a second time just to add that gutter.
    if (
      markdown is not None
      and markdown.code_rows is not None
      and len(markdown.parsed) == 1
      and markdown.parsed[0].type == 'fence'
      and options.max_width >= GUTTER_WIDTH + 8
      and not self.skill_names
    ):
      block = _TranscriptCodeBlock.create(markdown, markdown.parsed[0])
      block.text = _text.Text(markdown.parsed[0].content)
      rows = _segment.Segment.split_lines(
        console.render(
          block, options.update(width=options.max_width - GUTTER_WIDTH)
        )
      )
      for index, row in enumerate(rows):
        if index == 0 and not self.continuation:
          yield _segment.Segment(
            ASSISTANT_ICON + ' ', console.get_style('ui.accent')
          )
          yield _segment.Segment(' ', console.get_style('none'))
        else:
          yield _segment.Segment(' ' * GUTTER_WIDTH, console.get_style('none'))
        yield from row
        yield _segment.Segment.line()
      return
    yield from console.render(self.body, options)


def assistant_render(
  content: str,
  *,
  hyperlinks: bool = True,
  skill_names: Collection[str] = (),
  continuation: bool = False,
  markdown: _TranscriptMarkdown | None = None,
  source_offset: int = 0,
) -> Answer:
  """An assistant turn aligned with the user prompt's shared icon gutter."""
  return Answer(
    strip_ansi(content).strip(),
    hyperlinks,
    frozenset(skill_names),
    continuation,
    markdown,
    source_offset,
  )


def assistant_turn_renderable(
  reasoning: str | None,
  content: str | None,
  refusal: str | None = None,
  *,
  hyperlinks: bool = True,
  skill_names: Collection[str] = (),
) -> _console.RenderableType | None:
  """Render the visible reasoning and answer portions of an assistant turn."""
  parts: list[_console.RenderableType] = []
  if isinstance(reasoning, str) and reasoning.strip():
    parts.append(reasoning_text(reasoning))
  answer = '\n\n'.join(
    value.strip()
    for value in (content, refusal)
    if isinstance(value, str) and value.strip()
  )
  if answer:
    parts.append(
      assistant_render(answer, hyperlinks=hyperlinks, skill_names=skill_names)
    )
  if not parts:
    return None
  if len(parts) == 1:
    return parts[0]
  # Text('') renders as a blank line, separating reasoning from answer
  # so spacing matches the blank line between separate scrollback entries.
  spaced: list[_console.RenderableType] = [parts[0]]
  for part in parts[1:]:
    spaced.append(_text.Text(''))
    spaced.append(part)
  return _console.Group(*spaced)


def reasoning_text(content: str) -> MarkdownSource:
  """Distinguish provider-supplied reasoning with muted italic styling."""
  return MarkdownSource(
    strip_ansi(content).strip(),
    style='ui.muted.italic',
    # Left justification pads paragraphs with theme-colored spaces.
    justify='default',
  )


def log_text(level: str, text: str) -> _text.Text:
  return _text.Text(
    f'{strip_ansi(level)}: {strip_ansi(text)}', style='ui.warning'
  )


def error_text(message: str) -> _text.Text:
  return _text.Text(
    f'{icon_prefix("✗")}{strip_ansi(message)}', style='ui.error'
  )


def interrupt_text() -> _text.Text:
  return _text.Text(f'{icon_prefix("⊘")}interrupted by user', style='ui.muted')


def skill_status_text(display: str, status: str) -> _text.Text:
  text = _text.Text(
    f'{icon_prefix("◆")}{strip_ansi(status)} ', style='ui.muted'
  )
  text.append(strip_ansi(display), style='ui.info')
  return text


def skill_loaded_text(display: str) -> _text.Text:
  return skill_status_text(display, 'Loaded')


def context_display_text(display: str) -> _text.Text:
  """Render context summaries verbatim; bare skill mentions need a status."""
  if display.startswith('$'):
    return skill_loaded_text(display)
  text = _text.Text(icon_prefix('◆'), style='ui.muted')
  text.append(strip_ansi(display), style='ui.info')
  return text


def skill_display_for_tool(name: str, args: str) -> str | None:
  """Return the skill mention represented by rendered tool arguments."""
  if name != 'read':
    return None
  match = re.search(r'(?:^|[/\\])([^/\\]+)[/\\]SKILL\.md(?:,|$)', args)
  return f'${match.group(1)}' if match else None


def compaction_text(
  compaction: agent_pb2.Compaction,
  elapsed: str | None = None,
  *,
  show_detail: bool = True,
) -> _console.RenderableType:
  """A compaction marker styled like the other gutter rows."""
  detail = 'older history replaced by a summary'
  if compaction.HasField('usage') and compaction.usage.completion_tokens:
    detail = (
      f'{compaction.usage.prompt_tokens:,} tokens summarized into '
      f'{compaction.usage.completion_tokens:,}'
    )
  header = _text.Text('context compacted', style='ui.info')
  if compaction.HasField('context_tokens_before') and compaction.HasField(
    'context_tokens_after'
  ):
    header.append(
      f' {compaction.context_tokens_before:,} → '
      f'{compaction.context_tokens_after:,} tokens',
      style='ui.muted',
    )
  if elapsed is not None:
    header.append(f' ({elapsed})', style='ui.muted')
  return tool_block(header, result_summary(detail) if show_detail else None)


def result_summary(
  preview: str, style: str = 'ui.muted', *, compact: bool = False
) -> _text.Text:
  """The bounded result body shown under a tool call.

  In compact mode only the first line is shown, truncated with an ellipsis.
  """
  lines = strip_ansi(_bounded_preview(preview)).splitlines() or ['']
  first = lines[0]
  if compact and len(lines) > 1:
    first = first.rstrip() + '…'
  text = _text.Text(first, style=style, overflow='ellipsis', no_wrap=True)
  if not compact:
    for line in lines[1:]:
      text.append(f'\n{line}', style=style)
  return text


def looks_like_diff(text: str) -> bool:
  """Whether `text` is a unified diff (edit/write output, or `git diff`)."""
  return any(
    line.startswith('@@ ')
    for line in text[:_TOOL_PREVIEW_MAX_CHARS].splitlines()
  )


def _diff_highlighter(
  diff_lines: list[str],
  background: str = 'default',
  theme: str = 'ansi_dark',
) -> _syntax.Syntax:
  """A syntax highlighter keyed off the diff's target file path."""
  path = next(
    (ln[4:].strip() for ln in diff_lines if ln.startswith('+++ ')), 'file.txt'
  )
  lexer = _syntax.Syntax.guess_lexer(path)
  return _syntax.Syntax('', lexer, theme=theme, background_color=background)


def _diff_line(
  highlighter: _syntax.Syntax, lineno: int, sign: str, code: str, bg: str
) -> _text.Text:
  """One diff line: a number/sign gutter, syntax-highlighted code, a bg tint."""
  line = _text.Text(f'{lineno:>4} {sign} ')
  # Per-line highlighting has no cross-line state; multi-line strings/comments
  # aren't handled yet.
  highlighted = highlighter.highlight(code)
  highlighted.rstrip()  # Mutates in place; drops the newline `highlight` adds.
  line.append_text(highlighted)
  if bg:
    line.stylize(bg)  # Tint the whole line; fg stays syntax-coloured.
  return line


type _DiffRow = tuple[str, int, str]


def _diff_rows(diff: str, *, show_file_headers: bool) -> list[_DiffRow]:
  """Share hunk numbering between the text and panel presentations."""
  rows: list[_DiffRow] = []
  old_ln = new_ln = 0
  in_hunk = False
  for line in strip_ansi(diff).splitlines():
    if hunk := _HUNK_RE.match(line):
      old_ln, new_ln = int(hunk.group(1)), int(hunk.group(2))
      in_hunk = True
      continue
    marker, code = (line[0], line[1:]) if line else (' ', '')
    if in_hunk and marker not in (' ', '+', '-', '\\'):
      in_hunk = False
    if not in_hunk:
      if show_file_headers or not line.startswith(('--- ', '+++ ')):
        rows.append(('header', 0, line))
      continue
    if marker == '\\':
      continue
    if marker == '+':
      rows.append(('+', new_ln, code))
      new_ln += 1
    elif marker == '-':
      rows.append(('-', old_ln, code))
      old_ln += 1
    else:
      rows.append((' ', new_ln, code))
      old_ln += 1
      new_ln += 1
  return rows


def render_diff(diff: str, *, show_file_headers: bool = True) -> _text.Text:
  """A unified diff with syntax-highlighted code and red/green line tints."""
  highlighter = _diff_highlighter(strip_ansi(diff).splitlines())
  text = _text.Text()
  for index, (kind, lineno, code) in enumerate(
    _diff_rows(diff, show_file_headers=show_file_headers)
  ):
    if index:
      text.append('\n')
    if kind == 'header':
      text.append(code, style='bold')
    else:
      background = {'+': _DIFF_ADD_BG, '-': _DIFF_DEL_BG}.get(kind, '')
      text.append_text(_diff_line(highlighter, lineno, kind, code, background))
  return text


@dataclasses.dataclass(frozen=True)
class DiffPanel:
  """Full-width diff tints with wrapped code hanging after the number gutter."""

  diff: str
  show_file_headers: bool = True

  def __rich_console__(
    self, console: _console.Console, options: _console.ConsoleOptions
  ) -> _console.RenderResult:
    diff = ''.join(
      self.diff[:_TOOL_PREVIEW_MAX_CHARS].splitlines(keepends=True)[
        :_TOOL_PREVIEW_MAX_LINES
      ]
    )
    lines = strip_ansi(diff).splitlines()
    styles = {
      '+': console.get_style(_DIFF_ADD_BG),
      '-': console.get_style(_DIFF_DEL_BG),
      ' ': console.get_style('ui.code'),
    }
    highlighters = {
      kind: _diff_highlighter(
        lines,
        value.bgcolor.name if value.bgcolor else themes.DEFAULT_BACKGROUND,
        themes.syntax_theme(console, 'diff'),
      )
      for kind, value in styles.items()
    }
    table = _table.Table.grid(expand=True)
    table.add_column(no_wrap=True)
    table.add_column(ratio=1, overflow='fold')
    for kind, lineno, code in _diff_rows(
      diff, show_file_headers=self.show_file_headers
    ):
      if kind == 'header':
        table.add_row(' ', _text.Text(code, style='bold', overflow='fold'))
        continue
      highlighted = highlighters[kind].highlight(code)
      highlighted.rstrip()
      highlighted.no_wrap = False
      highlighted.overflow = 'fold'
      gutter = (
        f' {lineno:>4} {kind} ' if options.max_width >= 12 else f'{kind} '
      )
      table.add_row(
        _text.Text(gutter, style='ui.muted'),
        highlighted,
        style=styles[kind],
      )
    yield _CodePanel(table)
    if len(diff) < len(self.diff):
      yield _text.Text('Diff preview truncated.', style='ui.warning')


type CallHints = dict[str, tuple[str, str, str]]
"""Each tool call's name, raw JSON arguments, and activity kind, by call id."""


def unanswered_call(
  hint: tuple[str, str, str], *, compact: bool
) -> _console.RenderableType:
  """A tool call whose result never came: the call line, and nothing under it."""
  name, args, activity_kind = hint
  if compact:
    label = 'Subagent' if activity_kind == 'agent' else name
    if name == 'bash':
      args = command_preview(args)
    return tool_block(f'{label} {args}'.rstrip(), wrap=name == 'bash')
  return tool_text(name, args, None, activity_kind=activity_kind)


def running_tool_renderable(
  name: str, args: str, activity_kind: str, body: str
) -> _console.RenderableType:
  if (display := skill_display_for_tool(name, args)) is not None:
    return skill_status_text(display, 'Loading')
  return tool_text(
    name, args, result_summary(body), activity_kind=activity_kind
  )


def tool_text(
  name: str,
  args: str,
  result: _console.RenderableType | None,
  *,
  error: bool = False,
  activity_kind: str = 'tool',
) -> _console.RenderableType:
  """A transcript-mode call with arguments and a result preview."""
  name = strip_ansi(name)
  args = strip_ansi(args)
  is_agent = activity_kind == 'agent' and not error
  if is_agent:
    call = _text.Text(name.replace('_', ' ').title(), style='ui.info.bold')
    if args:
      call.append(' · ', style='ui.muted')
      call.append(args, style='ui.muted')
  else:
    call = _text.Text(style='ui.text')
    call.append(name, style='ui.error.bold' if error else 'ui.info.bold')
    call.append('(', style='ui.muted')
    call.append_text(_tool_arguments(args))
    call.append(')', style='ui.muted')
  return tool_block(call, result, error=error, wrap=name == 'bash')


def tool_block(
  header: str | _text.Text,
  body: _console.RenderableType | None = None,
  *,
  error: bool = False,
  wrap: bool = False,
) -> _console.RenderableType:
  """A tool activity with the transcript's shared icon gutter."""
  title = (
    _tool_title(header, error=error)
    if isinstance(header, str)
    else header.copy()
  )
  title.no_wrap = not wrap
  title.overflow = 'fold' if wrap else 'ellipsis'
  return ToolBlock(title, body, error)


@dataclasses.dataclass(frozen=True)
class ToolBlock:
  """Keep offscreen history free of Rich's per-cell layout object graph."""

  title: _text.Text
  body: _console.RenderableType | None
  error: bool

  def _grid(self) -> _table.Table:
    parts: list[_console.RenderableType] = [self.title]
    if self.body is not None:
      result = _table.Table.grid(padding=(0, 1), expand=True)
      result.add_column(width=1)
      result.add_column(ratio=1, overflow='fold')
      result.add_row(_text.Text('└', style='dim'), self.body)
      parts.append(result)
    grid = _table.Table.grid(padding=(0, 1), expand=True)
    grid.add_column(width=GUTTER_WIDTH - 1)
    grid.add_column(ratio=1, overflow='fold')
    grid.add_row(
      _text.Text('✗', style='ui.error')
      if self.error
      else _text.Text('·', style='dim'),
      _console.Group(*parts),
    )
    return grid

  def __rich_measure__(
    self, console: _console.Console, options: _console.ConsoleOptions
  ) -> _measure.Measurement:
    return self._grid().__rich_measure__(console, options)

  def __rich_console__(
    self, console: _console.Console, options: _console.ConsoleOptions
  ) -> _console.RenderResult:
    del console, options
    yield self._grid()


def _tool_arguments(arguments: str) -> _text.Text:
  """Emphasize quoted strings without recoloring paths, numbers, or booleans."""
  rendered = _text.Text(strip_ansi(arguments))
  rendered.highlight_regex(_TOOL_STRINGS_RE, style_prefix='ui.')
  return rendered


@dataclasses.dataclass(frozen=True)
class ToolLabel:
  """A palette-independent tool action, arguments, and status label."""

  action: str
  arguments: str = ''
  detail: str = ''
  error: bool = False

  def render(self) -> _text.Text:
    """Accent the action without dimming paths, queries, or command text."""
    label = _text.Text(style='ui.text')
    label.append(
      strip_ansi(self.action),
      style='ui.error.bold' if self.error else 'ui.info.bold',
    )
    if self.arguments:
      label.append(' ')
      label.append_text(_tool_arguments(self.arguments))
    if self.detail:
      label.append(
        strip_ansi(self.detail),
        style='ui.error' if self.error else 'ui.muted',
      )
    return label


def _tool_title(header: str, *, error: bool = False) -> _text.Text:
  verb, _, rest = strip_ansi(header).partition(' ')
  return ToolLabel(verb, rest, error=error).render()


@dataclasses.dataclass(frozen=True)
class Stopwatch:
  """Compact elapsed-time display for an in-flight activity."""

  started: float = dataclasses.field(default_factory=time.monotonic)

  def render(self) -> str:
    """The elapsed time as seconds or the two largest adjacent units."""
    return format_duration(time.monotonic() - self.started)


def format_duration(elapsed: float) -> str:
  """Format seconds using the two largest adjacent time units."""
  seconds = round(elapsed)
  if elapsed > 86_400:
    days, hours = divmod(seconds // 3_600, 24)
    return f'{days}d {hours}h'
  if elapsed > 3_600:
    hours, minutes = divmod(seconds // 60, 60)
    return f'{hours}h {minutes}m'
  if elapsed > 60:
    minutes, seconds = divmod(seconds, 60)
    return f'{minutes}m {seconds}s'
  return f'{seconds}s'


@dataclasses.dataclass(frozen=True)
class Expandable:
  """A collapsed tool row and the full output a click swaps in for it."""

  collapsed: _console.RenderableType
  expanded: _console.RenderableType

  def __rich_console__(
    self, console: _console.Console, options: _console.ConsoleOptions
  ) -> _console.RenderResult:
    del console, options
    yield self.collapsed


@dataclasses.dataclass(frozen=True)
class BashResult:
  """Highlight command previews only when their transcript rows are rendered."""

  command: str
  suffix: str
  body: _console.RenderableType | None
  error: bool
  wrap: bool

  def __rich_console__(
    self, console: _console.Console, options: _console.ConsoleOptions
  ) -> _console.RenderResult:
    del console, options
    code = _syntax.Syntax(
      self.command, 'bash', background_color='default', word_wrap=True
    )
    header = _tool_title('Ran', error=self.error)
    header.append(' ')
    header.append_text(code.highlight(self.command)[: len(self.command)])
    header.append(self.suffix, style='ui.error' if self.error else 'ui.muted')
    yield tool_block(header, self.body, error=self.error, wrap=self.wrap)


def bash_result(
  args: str,
  content: str,
  display: str | None,
  *,
  error: bool,
  exit_code: int | None = None,
  expandable: bool = False,
) -> _console.RenderableType:
  """Render bash's command display separately from its structured output text."""
  command = strip_ansi(display if display is not None else args)
  output = strip_ansi(content)
  completed = _BASH_COMPLETED_LINE_RE.search(output)
  if completed is not None:
    output = output[: completed.start()]
  # Bare stdout can look like a timeout line, so only a failed local call is
  # read as one.
  timeout = (
    _BASH_TIMEOUT_RE.match(output) if completed is not None and error else None
  )
  error = error or exit_code not in (None, 0)
  detail = f' · exit {exit_code}' if exit_code not in (None, 0) else ''
  status = _BASH_EXIT_STATUS_RE.match(output) if exit_code != 0 else None
  if timeout is not None:
    detail = ' · timed out'
    output = output[timeout.end() :].removeprefix('Partial stdout:\n')
  elif status is not None:
    code = exit_code if exit_code is not None else int(status.group(2))
    error = error or bool(status.group(1)) or code != 0
    detail = ' · timed out' if status.group(1) else f' · exit {code}'
    output = output[status.end() :].removeprefix('stdout:\n')
  suffix = (detail or ' · failed') if error else ''
  command = command_preview(command)
  lines = output.strip().splitlines()
  style = 'ui.error' if error else 'dim'
  preview = lines[0] if lines else ''
  if len(lines) > 1:
    preview += f' … (+{len(lines) - 1} lines)'
  collapsed = BashResult(
    command,
    suffix,
    _text.Text(preview, style=style, overflow='ellipsis', no_wrap=True)
    if lines
    else None,
    error=error,
    wrap=not expandable,
  )
  if not expandable:
    return collapsed
  expanded = BashResult(
    command,
    suffix,
    _text.Text('\n'.join(lines), style=style) if lines else None,
    error=error,
    wrap=True,
  )
  return Expandable(collapsed, expanded)


def tool_result_renderable(
  name: str,
  args: str,
  content: str,
  display: str | None,
  *,
  compact: bool,
  activity_kind: str,
  error: bool,
) -> _console.RenderableType | None:
  """Combine a tool result with its parked call into one block."""
  if name == 'bash':
    if compact:
      return bash_result(args, content, display, error=error)
    return tool_text(
      name,
      display if display is not None else args,
      _text.Text(
        strip_ansi(content), style='ui.error' if error else 'ui.muted'
      ),
      error=error,
    )
  if error:
    if compact:
      return tool_block(
        f'{name} {args}'.rstrip(),
        result_summary(content, style='ui.error'),
        error=True,
      )
    return tool_text(
      name,
      args,
      result_summary(content, style='ui.error'),
      error=True,
    )
  if (skill_display := skill_display_for_tool(name, args)) is not None:
    return skill_loaded_text(skill_display)
  if compact and activity_kind == 'agent':
    return tool_block(f'Subagent · {strip_ansi(args)}' if args else 'Subagent')
  # Only a tool's `display` payload (set by edit/write) may render as a diff;
  # raw content that merely looks like one — e.g. `git diff` run via bash —
  # renders as an ordinary result preview.
  if compact and display and looks_like_diff(display):
    paths = [
      line[4:]
      for line in display[:_TOOL_PREVIEW_MAX_CHARS].splitlines()
      if line.startswith('+++ ')
    ]
    if len(paths) == 1:
      label = 'Wrote' if name == 'write' else 'Edited'
      return tool_block(
        f'{label} {paths[0]}',
        DiffPanel(display, show_file_headers=False),
      )
    return tool_block('Edited files', DiffPanel(display))
  if compact:
    source = display or content
    return tool_block(
      f'{name} {args}'.rstrip(),
      result_summary(_result_preview(source), style='dim')
      if source.strip()
      else None,
    )
  if display:
    body: _console.RenderableType = (
      DiffPanel(display)
      if looks_like_diff(display)
      else result_summary(display)
    )
  else:
    body = result_summary(_result_preview(content))
  return tool_text(name, args, body, activity_kind=activity_kind)


def task_result_renderable(
  tool_name: str,
  task_id: str,
  source: str,
  *,
  success: bool,
  compact: bool,
  display: str | None = None,
) -> _console.RenderableType:
  """One task-result block, using the same renderer as its foreground tool."""
  if tool_name == 'bash' and compact:
    return bash_result(
      f'background task {task_id[:8]}', source, display, error=not success
    )
  source = strip_ansi(display if display is not None else source)
  kind = activity_kind(tool_name)
  body_source = source if kind == 'agent' else _result_preview(source)
  if kind == 'agent' and compact:
    return tool_block(
      'Subagent',
      _text.Text(
        'Complete' if success else 'Failed',
        style='dim' if success else 'ui.error',
      ),
      error=not success,
    )
  if kind == 'agent' and success:
    return tool_block(
      _text.Text(f'Subagent · task {task_id[:8]}', style='ui.info.bold'),
      _TranscriptMarkdown(body_source.strip()),
    )
  body = result_summary(
    body_source,
    style='ui.muted' if success else 'ui.error',
    compact=compact,
  )
  return tool_text(
    tool_name or 'task',
    f'task {task_id[:8]}',
    body,
    error=not success,
    activity_kind=kind,
  )


def cropped_history_notice(dropped: int, hint: str | None = None) -> _text.Text:
  """The line that a cropped history paints above the visible entries."""
  text = f'↑ {dropped:,} earlier entries not shown'
  if hint is not None:
    text += f' · {hint}'
  return _text.Text(text, style='ui.muted')
