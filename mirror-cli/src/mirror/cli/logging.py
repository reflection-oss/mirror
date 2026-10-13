"""Logging setup for the mirror CLI entrypoints."""

from __future__ import annotations

import contextlib
import logging
import logging.handlers
import os
import pathlib
import sys
from collections.abc import Generator
from typing import TextIO

from prompt_toolkit.application import current
from prompt_toolkit.output import defaults


_FORMATTER = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')


@contextlib.contextmanager
def capture_output(conversation_id: str) -> Generator[TextIO]:
  """Keep incidental Python, native, and child output off the UI terminal."""
  path = pathlib.Path.home() / '.mirror' / 'logs' / conversation_id
  path.mkdir(parents=True, exist_ok=True)
  with (
    (path / 'cli-stdout').open('a', buffering=1) as captured_stdout,
    (path / 'cli-stderr').open('a', buffering=1) as captured_stderr,
    os.fdopen(os.dup(1), 'w', buffering=1) as terminal,
    os.fdopen(os.dup(2), 'w', buffering=1) as stderr,
  ):
    output = defaults.create_output(stdout=terminal)
    sys.stdout.flush()
    sys.stderr.flush()
    try:
      os.dup2(captured_stdout.fileno(), 1)
      os.dup2(captured_stderr.fileno(), 2)
      with (
        contextlib.redirect_stdout(captured_stdout),
        contextlib.redirect_stderr(captured_stderr),
        current.create_app_session(output=output),
      ):
        yield terminal
    finally:
      captured_stdout.flush()
      captured_stderr.flush()
      sys.stdout.flush()
      sys.stderr.flush()
      os.dup2(terminal.fileno(), 1)
      os.dup2(stderr.fileno(), 2)


def _file_handler(*parts: str) -> logging.FileHandler | None:
  try:
    path = pathlib.Path.home().joinpath('.mirror', 'logs', *parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path)
  except (OSError, RuntimeError):
    return None
  handler.setFormatter(_FORMATTER)
  return handler


class Logs:
  """Route startup logs into the selected conversation once it is known."""

  def __init__(
    self, root: logging.Logger, startup: logging.Handler | None
  ) -> None:
    self._root = root
    self._startup = startup
    self._buffer = logging.handlers.MemoryHandler(
      capacity=10_000,
      flushLevel=logging.CRITICAL + 1,
    )
    self._buffer.setFormatter(_FORMATTER)
    if startup is not None:
      root.addHandler(startup)
    root.addHandler(self._buffer)

  def set_conversation(self, conversation_id: str) -> None:
    """Flush startup records and route subsequent logs to a conversation."""
    conversation = _file_handler(conversation_id, 'cli-stderr')
    target = conversation or logging.NullHandler()
    self._buffer.setTarget(target)
    self._buffer.flush()
    self._root.removeHandler(self._buffer)
    if self._startup is not None:
      self._root.removeHandler(self._startup)
      self._startup.close()
    self._root.addHandler(target)


def configure_logging() -> Logs:
  """Capture startup logs in memory and in the no-ID log."""
  logging.disable(logging.NOTSET)
  root = logging.getLogger()
  for handler in tuple(root.handlers):
    root.removeHandler(handler)
  root.setLevel(logging.INFO)
  return Logs(root, _file_handler('no-id.log'))
