"""Find a model on Hugging Face, review its licence, and download it."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Sequence

from mirror.protocol import (
  local_models_pb2,
)

from mirror.cli import local_models_client, selection
from mirror.cli.extensions.inference_controls.llama_cpp import footer, screens

_MODALITIES = {
  'text-generation': 'text',
  'text2text-generation': 'text',
  'image-text-to-text': 'text + image',
  'video-text-to-text': 'text + video',
  'audio-text-to-text': 'text + audio',
  'any-to-any': 'omni',
}


def size(byte_count: int) -> str:
  """A byte count in GiB with one decimal."""
  return f'{byte_count / 2**30:.1f} GiB'


def _count(value: float) -> str:
  for limit, suffix in ((1e9, 'B'), (1e6, 'M'), (1e3, 'K')):
    if value >= limit:
      return f'{value / limit:.1f}{suffix}'
  return f'{value:.0f}'


def _held(item: local_models_pb2.LocalModelCandidate) -> str:
  """What the disk already holds of a repository or quantization."""
  if item.downloaded > 1:
    return f'{item.downloaded} downloaded'
  if item.downloaded:
    return 'Downloaded'
  return 'Partial download' if item.partial else ''


def _describe(item: local_models_pb2.LocalModelCandidate) -> str:
  parts = [
    'Trusted' if item.trusted else '',
    _held(item),
    _MODALITIES.get(item.pipeline, ''),
    _count(item.parameters) if item.parameters else '',
    f'{footer.tokens(item.max_context)} ctx' if item.max_context else '',
    f'{_count(item.downloads)} downloads' if item.downloads else '',
  ]
  return ' · '.join(part for part in parts if part)


def catalog(
  runtime: local_models_client.Client,
  base: str,
  home: Callable[[], screens.Screen],
) -> screens.Screen:
  """Search Hugging Face; Esc and an accepted download lead to `home`."""

  async def stream(query: str) -> AsyncGenerator[screens.Update]:
    reply = await runtime.execute(
      local_models_pb2.ExecuteRequest(base_url=base, catalog=query)
    )
    # The daemon orders candidates.
    yield screens.Update(
      tuple(
        selection.Choice(item.name, item.name, _describe(item))
        for item in reply.candidates
      )
    )

  return screens.Screen(
    selection.Selector(
      'llama.catalog',
      'Download · search Hugging Face',
      (),
      lambda name: screens.goto(variants(runtime, base, name, home)),
      cancel_action=screens.goto(home()),
      prefiltered=True,
      header='Featured and trusted publishers first',
      placeholder='Search Hugging Face models',
    ),
    stream,
  )


def _variants_header(
  candidates: Sequence[local_models_pb2.LocalModelCandidate],
) -> str:
  """The most context a quantization allows, or why there is nothing to pick."""
  if not candidates:
    return 'No downloadable GGUF'
  maximum = max(item.max_context for item in candidates)
  return (
    f'Up to {footer.tokens(maximum)} context'
    if maximum
    else 'Context size unknown'
  )


def variants(
  runtime: local_models_client.Client,
  base: str,
  family: str,
  home: Callable[[], screens.Screen],
) -> screens.Screen:
  """The quantizations of one repository."""

  async def stream(_: str) -> AsyncGenerator[screens.Update]:
    reply = await runtime.execute(
      local_models_pb2.ExecuteRequest(base_url=base, variants=family)
    )
    yield screens.Update(
      tuple(
        selection.Choice(
          item.name,
          'Default'
          if item.name.endswith(':latest')
          else item.name.partition(':')[2],
          ' · '.join(
            part
            for part in (
              _held(item),
              size(item.download_bytes),
              item.fit,
            )
            if part
          ),
        )
        for item in reply.candidates
      ),
      _variants_header(reply.candidates),
    )

  return screens.Screen(
    selection.Selector(
      'llama.variants',
      family,
      (),
      lambda name: screens.goto(review(runtime, base, name, home)),
      cancel_action=screens.goto(catalog(runtime, base, home)),
    ),
    stream,
  )


def review(
  runtime: local_models_client.Client,
  base: str,
  name: str,
  home: Callable[[], screens.Screen],
) -> screens.Screen:
  """The download's size and licence; accepting starts it in the background."""
  reviewed = local_models_pb2.LocalModelDownload()

  async def stream(_: str) -> AsyncGenerator[screens.Update]:
    reply = await runtime.execute(local_models_pb2.ExecuteRequest(review=name))
    reviewed.CopyFrom(reply.download)
    total = sum(blob.size for blob in reviewed.blobs)
    licence = reviewed.license.strip()
    yield screens.Update(
      (selection.Choice('download', 'Accept licence and download'),),
      f'{reviewed.quantization} · {size(total)} · '
      f'https://huggingface.co/{name.partition(":")[0]}\n{licence}',
    )

  # The daemon owns the transfer, so nothing here keeps it alive: closing this
  # screen or the CLI does not stop it.
  return screens.Screen(
    selection.Selector(
      'llama.review',
      name,
      (),
      lambda _: screens.perform(
        lambda: runtime.execute(
          local_models_pb2.ExecuteRequest(base_url=base, download=reviewed)
        ),
        home(),
      ),
      cancel_action=screens.goto(
        variants(runtime, base, name.partition(':')[0], home)
      ),
    ),
    stream,
  )
