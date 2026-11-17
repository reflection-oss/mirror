"""Model resolution and provider prerequisites."""

from __future__ import annotations

import json
import pathlib

import pytest

from mirror.cli import models


def test_find_model_resolves_unversioned_beam_to_reflection(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: pathlib.Path,
) -> None:
  config = tmp_path / 'models.json'
  config.write_text(json.dumps([]))
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', str(config))

  option = models.find_model('Beam-501B-A23B')

  assert option.model_id == 'Beam-501B-A23B'
  assert option.api == 'reflection'


@pytest.mark.parametrize(
  ('option', 'supported'),
  [
    (models.ModelOption('Beam-1.0-501B-A23B', 'Beam', api='reflection'), False),
    (models.ModelOption('reflection/beam', 'Beam', api='vercel'), False),
    (
      models.ModelOption('claude-opus-5-5', 'Opus', api='anthropic-messages'),
      True,
    ),
  ],
  ids=['reflection-api', 'reflection-on-a-gateway', 'another-provider'],
)
def test_supports_images_is_false_only_for_reflection_models(
  option: models.ModelOption, supported: bool
) -> None:
  assert models.supports_images(option) is supported


def test_vercel_uses_its_gateway_key_and_fixed_endpoint(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  assert models.default_model('vercel') == 'reflection/beam'
  assert models.key_env('vercel') == 'AI_GATEWAY_API_KEY'
  assert models.base_env('vercel') is None
  monkeypatch.delenv('AI_GATEWAY_API_KEY', raising=False)
  # An OpenRouter endpoint or key must not satisfy the gateway.
  monkeypatch.setenv('OPENROUTER_API_KEY', 'synthetic-openrouter-key')
  monkeypatch.setenv('OPENROUTER_BASE_URL', 'http://configured')
  with pytest.raises(ValueError, match='Set AI_GATEWAY_API_KEY'):
    models.validate_connection('vercel', '')
  monkeypatch.setenv('AI_GATEWAY_API_KEY', 'synthetic-gateway-key')
  models.validate_connection('vercel', '')
  models.validate_connection('vercel', 'https://ai-gateway.vercel.sh/v1/')
  with pytest.raises(ValueError, match='not custom endpoints'):
    models.validate_connection('vercel', 'https://other.invalid/v1')


def test_vercel_selection_is_rejected_before_it_is_saved_without_a_key(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  option = models.ModelOption('reflection/beam', 'vercel', api='vercel')
  monkeypatch.delenv('AI_GATEWAY_API_KEY', raising=False)
  with pytest.raises(ValueError, match='Set AI_GATEWAY_API_KEY'):
    models.validate_selection(option, 'max')
  monkeypatch.setenv('AI_GATEWAY_API_KEY', 'synthetic-gateway-key')
  models.validate_selection(option, 'max')
