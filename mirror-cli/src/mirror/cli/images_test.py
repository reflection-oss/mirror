"""How a draft names its images, and which pasted text is an image's path."""

from __future__ import annotations

import pathlib

import pytest
from mirror.protocol import agent_pb2

from mirror.cli import images


def test_attached_returns_each_named_image_once_in_text_order() -> None:
  attachments = images.Attachments()
  first = attachments.add(b'first')
  second = attachments.add(b'second')

  attached = attachments.attached(
    f'{second} beside {first}, then {second} again, but not [Image #9]'
  )

  assert attached == [
    agent_pb2.ImageContent(name='[Image #2]', data=b'second'),
    agent_pb2.ImageContent(name='[Image #1]', data=b'first'),
  ]


@pytest.mark.parametrize(
  ('pasted', 'path'),
  [
    ('/tmp/My\\ Shot.png', pathlib.Path('/tmp/My Shot.png')),
    ("'/tmp/My Shot.png'", pathlib.Path('/tmp/My Shot.png')),
    ('file:///tmp/My%20Shot.png', pathlib.Path('/tmp/My Shot.png')),
    ('docs/logo.png', None),
    ('/tmp/one.png /tmp/two.png', None),
  ],
  ids=['escaped', 'quoted', 'file-url', 'relative', 'two-paths'],
)
def test_pasted_path_reads_one_absolute_path(
  pasted: str, path: pathlib.Path | None
) -> None:
  assert images.pasted_path(pasted) == path
