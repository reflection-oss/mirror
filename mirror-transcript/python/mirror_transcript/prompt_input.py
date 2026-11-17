"""Forwarded terminal bytes use prompt-toolkit's parser and Python's mode policy."""

from __future__ import annotations

import asyncio
import codecs
import contextlib
from collections.abc import Callable, Generator
from typing import override

from prompt_toolkit.input import base, vt100_parser
from prompt_toolkit.key_binding import key_processor


class Input(base.Input):
  """Keep the host descriptor for job control, but never read it in Python."""

  def __init__(self, original: base.Input) -> None:
    self.original = original
    self.sequence = 0
    self._callback: Callable[[], None] | None = None
    self._keys: list[key_processor.KeyPress] = []
    self._parser = vt100_parser.Vt100Parser(self._keys.append)
    self._decoder = codecs.getincrementaldecoder('utf-8')('surrogateescape')
    self._closed = False

  def feed(self, sequence: int, data: bytes) -> None:
    """Deliver bytes before acknowledging their consumption in a chrome frame."""
    self._parser.feed(self._decoder.decode(data))
    self.sequence = sequence
    self._notify()

  def _notify(self) -> None:
    if self._callback is not None:
      try:
        self._callback()
      except Exception as error:  # noqa: BLE001, input-handler isolation.
        asyncio.get_running_loop().call_exception_handler({
          'message': 'Exception in transcript input callback',
          'exception': error,
        })

  @override
  def fileno(self) -> int:
    return self.original.fileno()

  @override
  def typeahead_hash(self) -> str:
    return self.original.typeahead_hash()

  @override
  def read_keys(self) -> list[key_processor.KeyPress]:
    result = self._keys[:]
    self._keys.clear()
    return result

  @override
  def flush_keys(self) -> list[key_processor.KeyPress]:
    self._parser.flush()
    return self.read_keys()

  @property
  @override
  def closed(self) -> bool:
    return self._closed

  @override
  def raw_mode(self) -> contextlib.AbstractContextManager[None]:
    return self.original.raw_mode()

  @override
  def cooked_mode(self) -> contextlib.AbstractContextManager[None]:
    return self.original.cooked_mode()

  @override
  @contextlib.contextmanager
  def attach(self, input_ready_callback: Callable[[], None]) -> Generator[None]:
    previous, self._callback = self._callback, input_ready_callback
    try:
      if self._keys:
        self._notify()
      yield
    finally:
      self._callback = previous

  @override
  def detach(self) -> contextlib.AbstractContextManager[None]:
    return self.attach(lambda: None)

  @override
  def close(self) -> None:
    self._closed = True
    self._notify()
