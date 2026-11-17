"""Images attached to a draft: where they come from and how a draft names them.

The composer holds each image as a label such as `[Image #1]`. The label is
also the image's key in `Attachments`, which lasts for the process, so a draft
that is queued, restored, or handed to another workspace still resolves to the
same image. A sent message carries each image under its label.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import pathlib
import re
import shlex
import sys
import urllib.parse
import urllib.request
from collections.abc import Sequence

from mirror.protocol import agent_pb2

# The agent refuses a larger image, whose base64 form would pass the 5 MiB a
# provider accepts.
MAX_BYTES = 5 * 1024 * 1024 // 4 * 3
# The agent refuses more than this in one message, which has to fit one RPC
# frame.
MAX_MESSAGE_BYTES = 2 * MAX_BYTES
# An image whose draft was sent or abandoned long ago is not worth its memory.
_KEPT = 32
_LABEL = re.compile(r'\[Image #\d+\]')
_LABEL_AT_END = re.compile(r'\[Image #\d+\] ?\Z')

# Finder copies a file as its URL beside an image of its icon, so the URL wins;
# `filePathURL` turns the file-reference form of that URL into a path. A
# function body keeps osascript from printing the script's result.
_MACOS_PASTE = (
  '/usr/bin/osascript',
  '-l',
  'JavaScript',
  '-e',
  "ObjC.import('AppKit');"
  'function run() {'
  'const board = $.NSPasteboard.generalPasteboard;'
  'const out = $.NSFileHandle.fileHandleWithStandardOutput;'
  "const url = board.stringForType('public.file-url');"
  'const file = url.isNil() ? $() : $.NSURL.URLWithString(url).filePathURL;'
  'if (!file.isNil()) {'
  'const name = file.absoluteString;'
  'out.writeData(name.dataUsingEncoding($.NSUTF8StringEncoding));'
  'return;'
  '}'
  "let data = board.dataForType('public.png');"
  "if (data.isNil()) data = board.dataForType('public.jpeg');"
  'if (data.isNil()) {'
  "const tiff = board.dataForType('public.tiff');"
  'const bitmap = tiff.isNil()'
  ' ? $() : $.NSBitmapImageRep.imageRepWithData(tiff);'
  'if (!bitmap.isNil()) data = bitmap.representationUsingTypeProperties('
  '$.NSBitmapImageFileTypePNG, $());'
  '}'
  'if (!data.isNil()) out.writeData(data);'
  '}',
)
_WAYLAND_PASTE = ('wl-paste', '--no-newline', '--type', 'image/png')
_X11_PASTE = (
  'xclip',
  '-selection',
  'clipboard',
  '-target',
  'image/png',
  '-out',
)


class ClipboardError(Exception):
  """The clipboard could not be read; the message tells the user why."""


class Attachments:
  """The images pasted in this process, by the label that stands for each."""

  def __init__(self) -> None:
    self._images: dict[str, bytes] = {}
    self._added = 0

  def add(self, data: bytes) -> str:
    """Keep an image and return the label a draft uses for it."""
    self._added += 1
    label = f'[Image #{self._added}]'
    self._images[label] = data
    while len(self._images) > _KEPT:
      del self._images[next(iter(self._images))]
    return label

  def spans(self, text: str) -> tuple[tuple[int, int], ...]:
    """Where the labels of attached images start and end in `text`."""
    return tuple(
      match.span()
      for match in _LABEL.finditer(text)
      if match.group() in self._images
    )

  def label_before(self, text: str) -> str | None:
    """The attached label `text` ends with, and the space after it if any."""
    match = _LABEL_AT_END.search(text)
    if match is None or match.group().rstrip() not in self._images:
      return None
    return match.group()

  def attached(self, text: str) -> list[agent_pb2.ImageContent]:
    """The images `text` names, each once, in the order it names them."""
    labels = dict.fromkeys(text[start:end] for start, end in self.spans(text))
    return [
      agent_pb2.ImageContent(name=label, data=self._images[label])
      for label in labels
    ]


def is_image(data: bytes) -> bool:
  """Whether `data` starts like a PNG, JPEG, GIF, or WebP."""
  return data.startswith((
    b'\x89PNG\r\n\x1a\n',
    b'\xff\xd8\xff',
    b'GIF87a',
    b'GIF89a',
  )) or (data[:4] == b'RIFF' and data[8:12] == b'WEBP')


def too_large(data: bytes) -> str | None:
  """Tell the user why an image is too large to attach, if it is."""
  if len(data) <= MAX_BYTES:
    return None
  return (
    f'Image is larger than {MAX_BYTES / 1_000_000:.1f} MB, the most Mirror '
    'can attach. Crop or shrink it and paste again.'
  )


def too_many(images: Sequence[agent_pb2.ImageContent]) -> str | None:
  """Tell the user why one message cannot carry `images`, if it cannot."""
  if sum(len(image.data) for image in images) <= MAX_MESSAGE_BYTES:
    return None
  return (
    f'One message can carry {MAX_MESSAGE_BYTES / 1_000_000:.1f} MB of images. '
    'Remove an image or send them in separate messages.'
  )


def unsupported(model: str) -> str:
  """Tell the user that `model` cannot read the images in a draft."""
  return (
    f'Model {model} does not support image inputs. '
    'Remove images or switch models.'
  )


def pasted_path(text: str) -> pathlib.Path | None:
  """The absolute path that pasted text spells, as a terminal drops a file.

  A relative path stays text: it more likely names a file for the agent to
  work on than an image to show it.
  """
  text = text.strip()
  if text.startswith('file://'):
    name = urllib.request.url2pathname(urllib.parse.urlsplit(text).path)
  else:
    try:
      words = shlex.split(text)
    except ValueError:
      return None
    if len(words) != 1:
      return None
    name = words[0]
  path = pathlib.Path(name).expanduser()
  return path if path.is_absolute() else None


def read_file(path: pathlib.Path) -> bytes | None:
  """A regular file's bytes, up to one past the largest image, or None."""
  try:
    if not path.is_file():
      return None
    with path.open('rb') as file:
      return file.read(MAX_BYTES + 1)
  except OSError:
    return None


async def read_clipboard() -> bytes:
  """The image on the system clipboard, ready to attach.

  Raises:
    ClipboardError: The clipboard holds no image Mirror can attach.
  """
  if sys.platform == 'darwin':
    output = await _output(_MACOS_PASTE)
    if output is not None and output.startswith(b'file://'):
      path = pasted_path(output.decode('utf-8', 'replace'))
      output = await asyncio.to_thread(read_file, path) if path else None
  elif sys.platform.startswith('linux'):
    output = await _output(
      _WAYLAND_PASTE if os.environ.get('WAYLAND_DISPLAY') else _X11_PASTE
    )
  else:
    raise ClipboardError('Mirror cannot paste images on this platform.')
  if not output or not is_image(output):
    raise ClipboardError('No image on the clipboard.')
  if problem := too_large(output):
    raise ClipboardError(problem)
  return output


async def _output(command: Sequence[str]) -> bytes | None:
  """A clipboard helper's output, or None when it found nothing to print."""
  try:
    process = await asyncio.create_subprocess_exec(
      *command,
      stdin=asyncio.subprocess.DEVNULL,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.DEVNULL,
    )
  except FileNotFoundError as error:
    raise ClipboardError(
      f'Pasting an image needs {command[0]}, which is not installed.'
    ) from error
  except OSError as error:
    raise ClipboardError('Could not read the clipboard.') from error
  try:
    async with asyncio.timeout(5):
      output, _ = await process.communicate()
  except TimeoutError as error:
    raise ClipboardError('Could not read the clipboard.') from error
  finally:
    # A timeout or cancellation leaves the helper running; reap it.
    if process.returncode is None:
      with contextlib.suppress(ProcessLookupError):
        process.kill()
      await process.communicate()
  return output if process.returncode == 0 else None
