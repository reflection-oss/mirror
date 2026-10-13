"""Tests for config-driven model definitions."""

from __future__ import annotations

import json

from mirror.cli import models


def test_load_models_returns_empty_without_config(
  monkeypatch, tmp_path
) -> None:
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', str(tmp_path / 'missing.json'))
  result = models.load_models()
  assert result == ()


def test_load_models_reads_config_file(monkeypatch, tmp_path) -> None:
  config = tmp_path / 'models.json'
  config.write_text(
    json.dumps([
      {
        'id': 'my-local-model',
        'label': 'Local Llama',
        'description': 'Self-hosted',
        'api': 'openai-chat-completions',
        'base_url': 'http://localhost:8080/v1',
      },
      {
        'id': 'openai/gpt-5.6-pro',
        'label': 'GPT-5.6 Pro',
        'api': 'openai-chat-completions',
      },
    ])
  )
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', str(config))

  result = models.load_models()
  assert len(result) == 2
  assert result[0].model_id == 'my-local-model'
  assert result[0].label == 'Local Llama'
  assert result[0].description == 'Self-hosted'
  assert result[0].api == 'openai-chat-completions'
  assert result[0].base_url == 'http://localhost:8080/v1'
  assert result[1].description == ''
  assert result[1].base_url == ''


def test_load_models_rejects_missing_id(monkeypatch, tmp_path) -> None:
  config = tmp_path / 'models.json'
  config.write_text(json.dumps([{'label': 'No id'}]))
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', str(config))
  try:
    models.load_models()
    raise AssertionError('expected ValueError')
  except ValueError as error:
    assert 'missing string "id"' in str(error)


def test_find_model_returns_match(monkeypatch, tmp_path) -> None:
  config = tmp_path / 'models.json'
  config.write_text(
    json.dumps([
      {'id': 'custom-model', 'label': 'Custom', 'api': 'openai-responses'},
    ])
  )
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', str(config))

  option = models.find_model('custom-model')
  assert option.api == 'openai-responses'


def test_find_model_returns_bare_option_for_unknown(
  monkeypatch, tmp_path
) -> None:
  config = tmp_path / 'models.json'
  config.write_text(json.dumps([]))
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', str(config))

  option = models.find_model('unknown-model')
  assert option.model_id == 'unknown-model'
  assert option.label == 'unknown-model'
  assert option.api == ''


def test_save_and_load_last_model(monkeypatch, tmp_path) -> None:
  config = tmp_path / 'models.json'
  config.write_text('[]')
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', str(config))

  assert models.load_last_model() is None
  models.save_last_model('zai-org/GLM-5.2-FP8')
  assert models.load_last_model() == 'zai-org/GLM-5.2-FP8'
