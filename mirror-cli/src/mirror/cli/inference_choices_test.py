"""Which protocols count as ready to list their models."""

from __future__ import annotations

import pathlib

import pytest

from mirror.cli import inference_choices, models


@pytest.mark.parametrize(
  ('api', 'endpoint'),
  [
    ('openai-responses', 'OPENAI_BASE_URL'),
    ('openai-chat-completions', 'OPENROUTER_BASE_URL'),
    ('anthropic-messages', 'ANTHROPIC_BASE_URL'),
  ],
)
def test_ready_accepts_a_keyless_endpoint(
  monkeypatch: pytest.MonkeyPatch, api: str, endpoint: str
) -> None:
  monkeypatch.delenv(models.key_env(api), raising=False)
  monkeypatch.setenv(endpoint, 'http://127.0.0.1:8000/v1')
  assert inference_choices.ready(api)


@pytest.mark.parametrize(
  ('api', 'endpoint', 'url'),
  [
    ('openai-responses', 'OPENAI_BASE_URL', ''),
    # A provider's own endpoint is not self-hosted, so it still needs a key.
    ('anthropic-messages', 'ANTHROPIC_BASE_URL', 'https://api.anthropic.com'),
    (
      'openai-chat-completions',
      'OPENROUTER_BASE_URL',
      'https://openrouter.ai/api/v1/',
    ),
  ],
)
def test_ready_needs_a_key_without_a_self_hosted_endpoint(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: pathlib.Path,
  api: str,
  endpoint: str,
  url: str,
) -> None:
  # No key is saved under HOME either.
  monkeypatch.setenv('HOME', str(tmp_path))
  monkeypatch.delenv(models.key_env(api), raising=False)
  monkeypatch.setenv(endpoint, url)
  assert not inference_choices.ready(api)
