"""Rich and ANSI rendering for conversation entries and tool calls."""

from __future__ import annotations

import dataclasses
import io
import os
import re
import time
from collections.abc import Collection, Mapping, Sequence
from typing import Any, ClassVar, cast, override

import pydantic
from mirror.protocol import agent_pb2
from rich import box as _box
from rich import console as _console
from rich import markdown as _markdown
from rich import padding as _padding
from rich import segment as _segment
from rich import syntax as _syntax
from rich import table as _table
from rich import text as _text

from mirror import distribution
from mirror.cli import themes
from mirror.cli.components import card as _card
from mirror.cli.composer import skill_completion as _skill_completion

# The glyph the input line uses, reused as the user-message prefix.
PROMPT = '›'
# Gutter glyph marking an assistant turn.
ASSISTANT_ICON = '●'
_RESULT_PREVIEW_MAX_LINES = 3
_PREVIEW_MAX_CHARS = 1024
_COMMAND_PREVIEW_MAX_CHARS = 240
_PARTIAL_ARGUMENTS = pydantic.TypeAdapter(dict[str, Any])
# Fixed icon-gutter width so every turn aligns despite varying glyph widths.
GUTTER_WIDTH = 2
_HUNK_RE = re.compile(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@')

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
  text = result.strip()
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
    arguments = _PARTIAL_ARGUMENTS.validate_json(
      arguments_json or '{}', experimental_allow_partial=True
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
        f'{name.removeprefix("mcp__").replace("__", ".", 1)}({call.get("arguments", {})})'
        for call in cast(list[pydantic.JsonValue], calls)
        if isinstance(call, dict) and isinstance(name := call.get('name'), str)
      )
    }
  if 'prompt' in arguments and len(arguments) == 1:
    summary = _one_line(arguments['prompt'])
  else:
    summary = _argument_summary(arguments)
  return _bounded_preview(summary)


def approval_args(arguments_json: str, *, tool_name: str) -> str:
  """Arguments to approve, with a bash command exactly as it will run."""
  if tool_name == 'bash':
    try:
      arguments = _PARTIAL_ARGUMENTS.validate_json(arguments_json or '{}')
    except pydantic.ValidationError:
      arguments = {}
    if isinstance(command := arguments.get('command'), str):
      return command.strip()
  return compact_args(arguments_json)


def tool_operation(arguments_json: str) -> str:
  """The structured action discriminator of a multiplexed tool, if present."""
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


def ansi_console(
  buf: io.StringIO, width: int, *, theme_name: str = 'dark'
) -> _console.Console:
  """A console that renders to ANSI in `buf`, whatever the real output is.

  Escapes are always written: the live pane parses them itself, and scrollback
  is printed through the same pre-render so both keep one notion of spacing.
  """
  return _console.Console(
    file=buf,
    force_terminal=True,
    no_color=theme_name == 'mono',
    color_system='standard' if theme_name == 'terminal' else 'truecolor',
    theme=themes.rich_theme(theme_name),
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


def without_hyperlinks(text: str) -> str:
  """Remove OSC-8 delimiters unsupported by prompt-toolkit's ANSI decoder."""
  return _SAFE_HYPERLINK_RE.sub('', text)


def renderable_to_ansi(
  renderable: _console.RenderableType, width: int, *, theme_name: str = 'dark'
) -> str:
  """Render Rich output as terminal ANSI at a fixed width."""
  buf = io.StringIO()
  ansi_console(buf, width, theme_name=theme_name).print(renderable, end='')
  return sanitize_ansi(buf.getvalue())


def render_blocks(
  renderables: Sequence[_console.RenderableType],
  width: int,
  *,
  theme_name: str = 'dark',
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
  *,
  setup_url: str | None = None,
  setup_provider: str = 'Reflection',
) -> _console.RenderableType:
  """Quiet welcome facts, with a frame only for interactive connection setup."""
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

  content: _console.RenderableType = _console.Group(info, facts)
  if setup_url is None:
    return content
  welcome = _text.Text()
  welcome.append(f'\nConnect to {setup_provider}\n', style='ui.info.bold')
  welcome.append(
    'Add an API key to start coding with Mirror.\n\n', style='default'
  )
  welcome.append('1  ', style='ui.info.bold')
  welcome.append('Create a key\n   ', style='bold')
  link = _text.Text(setup_url, style='ui.info')
  link.stylize(f'underline link {setup_url}')
  welcome.append_text(link)
  welcome.append('\n\n2  ', style='ui.info.bold')
  welcome.append('Paste your key below', style='bold')
  welcome.append(' — input is hidden.', style='ui.muted')
  content = _console.Group(content, welcome)
  return _card.frame(content, expand=False, width=76)


def user_text(
  content: str, skill_names: Collection[str] = ()
) -> _padding.Padding:
  """A full-width shaded prompt, with wrapped lines aligned to its text."""
  text = _text.Text(strip_ansi(content).strip(), overflow='fold')
  _skill_completion.highlight_mentions(text, skill_names)
  grid = _table.Table.grid(padding=(0, 1), expand=True)
  grid.add_column(width=1)
  grid.add_column(ratio=1, overflow='fold')
  grid.add_row(_text.Text(PROMPT, style='bold dim'), text)
  # Set the foreground too so the dark fill stays readable on light terminals.
  return _padding.Padding(grid, (1, 1, 1, 0), style='ui.message')


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

  def __init__(self, body: _console.RenderableType) -> None:
    self.body = body

  def __rich_console__(
    self, console: _console.Console, options: _console.ConsoleOptions
  ) -> _console.RenderResult:
    if options.max_width < 2:
      yield self.body
      return
    lines = console.render_lines(
      _padding.Padding(self.body, (1, 0), style='ui.code', expand=True),
      options.update(width=options.max_width - 1),
    )
    rail = console.get_style('ui.code-rail')
    for line in lines:
      yield _segment.Segment('┃', rail)
      yield from line
      yield _segment.Segment.line()


class _TranscriptCodeBlock(_markdown.CodeBlock):
  """Set code on the shared panel without changing its syntax foregrounds."""

  @override
  def __rich_console__(
    self, console: _console.Console, options: _console.ConsoleOptions
  ) -> _console.RenderResult:
    background = console.get_style('ui.code').bgcolor
    code = _syntax.Syntax(
      str(self.text).rstrip(),
      self.lexer_name,
      theme=self.theme,
      word_wrap=True,
      background_color=background.name if background else 'default',
      padding=(0, 1) if options.max_width >= 6 else 0,
    )
    yield _CodePanel(code)


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

  elements: ClassVar[dict[str, type[_markdown.MarkdownElement]]] = {
    **_markdown.Markdown.elements,
    'table_open': _TranscriptTable,
    'heading_open': _TranscriptHeading,
    'fence': _TranscriptCodeBlock,
    'code_block': _TranscriptCodeBlock,
    'blockquote_open': _TranscriptQuote,
    'list_item_open': _TranscriptListItem,
  }


class StreamingAnswer:
  """Retain completed Markdown blocks and parse only the changing final block."""

  def __init__(self, skill_names: Collection[str]) -> None:
    self._skill_names = skill_names
    self._source = ''
    self._stable_source_len = 0
    self._stable: list[_console.RenderableType] = []
    self._parts: tuple[_console.RenderableType, ...] = ()

  def update(self, source: str) -> tuple[_console.RenderableType, ...]:
    """Return unchanged prefix objects followed by the current rich tail."""
    source = strip_ansi(source).strip()
    if source == self._source:
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
    markdown = _TranscriptMarkdown(suffix, hyperlinks=False)
    boundaries = [
      (index, token.map[0])
      for index, token in enumerate(markdown.parsed)
      if token.level == 0 and token.nesting != -1 and token.map is not None
    ]
    if not boundaries:
      self._parts = tuple(self._stable)
      return self._parts
    lines = suffix.splitlines(keepends=True)
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
      block = _TranscriptMarkdown('')
      block.parsed = markdown.parsed[token_index:next_index]
      self._stable.append(
        assistant_render(
          block_source,
          hyperlinks=False,
          skill_names=self._skill_names,
          continuation=bool(self._stable),
          markdown=block,
        )
      )
    self._stable_source_len += sum(len(line) for line in lines[:last_line])
    markdown.parsed = markdown.parsed[last_index:]
    tail = assistant_render(
      ''.join(lines[last_line:]),
      hyperlinks=False,
      skill_names=self._skill_names,
      continuation=bool(self._stable),
      markdown=markdown,
    )
    self._parts = (*self._stable, tail)
    return self._parts


def assistant_render(
  content: str,
  *,
  hyperlinks: bool = True,
  skill_names: Collection[str] = (),
  continuation: bool = False,
  markdown: _TranscriptMarkdown | None = None,
) -> _console.RenderableType:
  """An assistant turn aligned with the user prompt's two-column gutter."""
  grid = _table.Table.grid(padding=(0, 1))
  grid.add_column(width=1)
  # Rich's code-block wrapping inherits the column's overflow policy.
  grid.add_column(overflow='fold')
  grid.add_row(
    _text.Text('' if continuation else ASSISTANT_ICON, style='ui.accent'),
    _StartTrimmed(
      markdown
      or _TranscriptMarkdown(strip_ansi(content).strip(), hyperlinks=hyperlinks)
    ),
  )
  return _skill_completion.highlight_renderable(grid, skill_names)


def assistant_turn_renderable(
  reasoning: str | None,
  content: str | None,
  refusal: str | None = None,
  *,
  hyperlinks: bool = True,
  skill_names: Collection[str] = (),
  compact_reasoning: bool = False,
) -> _console.RenderableType | None:
  """Render the visible reasoning and answer portions of an assistant turn."""
  parts: list[_console.RenderableType] = []
  if not compact_reasoning and isinstance(reasoning, str) and reasoning.strip():
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


def reasoning_text(content: str) -> _console.Group:
  """Label provider-supplied reasoning separately from the assistant answer."""
  return _console.Group(
    _text.Text('Reasoning', style='ui.muted.italic'),
    _TranscriptMarkdown(
      strip_ansi(content).strip(), style='ui.muted.italic', hyperlinks=False
    ),
  )


def log_text(level: str, text: str) -> _text.Text:
  return _text.Text(
    f'{strip_ansi(level)}: {strip_ansi(text)}', style='ui.warning'
  )


def error_text(message: str) -> _text.Text:
  return _text.Text(f'✗ {strip_ansi(message)}', style='ui.error')


def interrupt_text() -> _text.Text:
  return _text.Text('⊘ interrupted by user', style='ui.muted')


def skill_status_text(display: str, status: str) -> _text.Text:
  text = _text.Text(f'◆ {strip_ansi(status)} ', style='ui.muted')
  text.append(strip_ansi(display), style='ui.info')
  return text


def skill_loaded_text(display: str) -> _text.Text:
  return skill_status_text(display, 'loaded')


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
  if elapsed is not None:
    header.append(f' ({elapsed})', style='ui.muted')
  return tool_block(header, result_summary(detail) if show_detail else None)


def result_summary(
  preview: str, style: str = 'ui.muted', *, compact: bool = False
) -> _text.Text:
  """The bounded result body shown under a tool call.

  In compact mode only the first line is shown, truncated with an ellipsis.
  """
  lines = _bounded_preview(strip_ansi(preview)).splitlines() or ['']
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
  return any(line.startswith('@@ ') for line in text.splitlines())


def _diff_highlighter(
  diff_lines: list[str], background: str = 'default'
) -> _syntax.Syntax:
  """A syntax highlighter keyed off the diff's target file path."""
  path = next(
    (ln[4:].strip() for ln in diff_lines if ln.startswith('+++ ')), 'file.txt'
  )
  lexer = _syntax.Syntax.guess_lexer(path)
  return _syntax.Syntax(
    '', lexer, theme='ansi_dark', background_color=background
  )


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
class _DiffPanel:
  """Full-width diff tints with wrapped code hanging after the number gutter."""

  diff: str
  show_file_headers: bool = True

  def __rich_console__(
    self, console: _console.Console, options: _console.ConsoleOptions
  ) -> _console.RenderResult:
    lines = strip_ansi(self.diff).splitlines()
    styles = {
      '+': console.get_style(_DIFF_ADD_BG),
      '-': console.get_style(_DIFF_DEL_BG),
      ' ': console.get_style('ui.code'),
    }
    highlighters = {
      kind: _diff_highlighter(
        lines,
        value.bgcolor.name if value.bgcolor else themes.DEFAULT_BACKGROUND,
      )
      for kind, value in styles.items()
    }
    table = _table.Table.grid(expand=True)
    table.add_column(no_wrap=True)
    table.add_column(ratio=1, overflow='fold')
    for kind, lineno, code in _diff_rows(
      self.diff, show_file_headers=self.show_file_headers
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
    return skill_status_text(display, 'loading')
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
    call = _text.Text(f'{name}(', style='ui.warning')
    call.append(args, style='ui.muted')
    call.append(')', style='ui.warning')
  return tool_block(call, result, error=error, wrap=name == 'bash')


def tool_block(
  header: str | _text.Text,
  body: _console.RenderableType | None = None,
  *,
  error: bool = False,
  wrap: bool = False,
) -> _console.RenderableType:
  """A tool activity with the transcript's two-column gutter."""
  title = _tool_title(header) if isinstance(header, str) else header.copy()
  title.no_wrap = not wrap
  title.overflow = 'fold' if wrap else 'ellipsis'
  parts: list[_console.RenderableType] = [title]
  if body is not None:
    result = _table.Table.grid(padding=(0, 1), expand=True)
    result.add_column(width=1)
    result.add_column(ratio=1, overflow='fold')
    result.add_row(_text.Text('└', style='dim'), body)
    parts.append(result)
  grid = _table.Table.grid(padding=(0, 1), expand=True)
  grid.add_column(width=1)
  grid.add_column(ratio=1, overflow='fold')
  grid.add_row(
    _text.Text('✗', style='ui.error')
    if error
    else _text.Text('·', style='dim'),
    _console.Group(*parts),
  )
  return grid


def _tool_title(header: str) -> _text.Text:
  """Keep activity secondary to answers, emphasizing only the leading verb."""
  verb, space, rest = strip_ansi(header).partition(' ')
  title = _text.Text(style='ui.tool-heading')
  title.append(verb, style='bold')
  title.append(space + rest)
  return title


@dataclasses.dataclass(frozen=True)
class Stopwatch:
  """Elapsed-seconds display for an in-flight activity (compaction, tools)."""

  started: float = dataclasses.field(default_factory=time.monotonic)

  def render(self) -> str:
    """The elapsed time as a compact label like `12s`."""
    return f'{time.monotonic() - self.started:.0f}s'


def _bash_result(
  args: str, content: str, display: str | None, *, error: bool
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
  detail = ''
  if timeout is not None:
    detail = ' · timed out'
    output = output[timeout.end() :].removeprefix('Partial stdout:\n')
  elif (status := _BASH_EXIT_STATUS_RE.match(output)) is not None:
    code = int(status.group(2))
    error = error or bool(status.group(1)) or code != 0
    detail = ' · timed out' if status.group(1) else f' · exit {code}'
    output = output[status.end() :].removeprefix('stdout:\n')
  lines = output.strip().splitlines()
  preview = lines[0] if lines else ''
  if len(lines) > 1:
    preview += f' … (+{len(lines) - 1} lines)'
  body = (
    _text.Text(
      preview,
      style='ui.error' if error else 'dim',
      overflow='ellipsis',
      no_wrap=True,
    )
    if lines
    else None
  )
  suffix = (detail or ' · failed') if error else ''
  header = _tool_title(f'Ran {command_preview(command)}{suffix}')
  return tool_block(header, body, error=error, wrap=True)


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
      return _bash_result(args, content, display, error=error)
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
      return None
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
      line[4:] for line in display.splitlines() if line.startswith('+++ ')
    ]
    if len(paths) == 1:
      label = 'Wrote' if name == 'write' else 'Edited'
      return tool_block(
        f'{label} {paths[0]}',
        _DiffPanel(display, show_file_headers=False),
      )
    return tool_block('Edited files', _DiffPanel(display))
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
      _DiffPanel(display)
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
    return _bash_result(
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
