from __future__ import annotations

import json
import pathlib

import pytest

from mirror.cli.extensions.connections import transport


@pytest.fixture(autouse=True)
def no_firecrawl_key(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.delenv('FIRECRAWL_API_KEY', raising=False)


def test_local_declarations_keep_only_credential_references(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv('MIRROR_MCP_DIR', str(tmp_path))
  monkeypatch.setenv('FIXTURE_TOKEN', 'synthetic-secret')
  declaration = {
    'url': 'https://example.test/mcp',
    'auth': {'type': 'bearer', 'tokenEnvVar': 'FIXTURE_TOKEN'},
    'enabledTools': ['echo'],
  }
  (tmp_path / 'fixture.json').write_text(json.dumps(declaration))
  connections = transport.local_connections(
    on_auth_required=lambda _: pytest.fail('unexpected sign-in')
  )
  assert connections.connections[0].config == declaration
  assert 'synthetic-secret' not in repr(connections)


def test_invalid_configuration_does_not_hide_other_declarations(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv('MIRROR_MCP_DIR', str(tmp_path))
  (tmp_path / 'broken.json').write_text('{')
  (tmp_path / 'public.json').write_text('{"url":"https://public.test/mcp"}')
  (tmp_path / 'private.json').write_text(
    '{"url":"https://private.test/mcp","auth":{"type":"oauth"}}'
  )
  pending: list[str] = []
  connections = transport.local_connections(on_auth_required=pending.append)
  assert connections.connected_servers == {'public', 'parallel'}
  assert pending == ['private']


@pytest.mark.parametrize('key', [None, '', '  ', 'configured-firecrawl-key'])
def test_web_search_defaults_to_parallel_only_without_firecrawl(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, key: str | None
) -> None:
  monkeypatch.setenv('MIRROR_MCP_DIR', str(tmp_path / 'missing'))
  monkeypatch.delenv('FIRECRAWL_API_KEY', raising=False)
  if key is not None:
    monkeypatch.setenv('FIRECRAWL_API_KEY', key)
  connections = transport.local_connections(
    on_auth_required=lambda _: pytest.fail('unexpected sign-in')
  ).connections
  if key and key.strip():
    assert connections == ()
  else:
    assert len(connections) == 1
    assert connections[0].name == 'parallel'
    assert connections[0].config == {
      'url': 'https://search.parallel.ai/mcp',
      'enabledTools': ['web_search', 'web_fetch'],
    }
  assert not (tmp_path / 'missing').exists()


@pytest.mark.parametrize('name', ['parallel', 'my-search'])
@pytest.mark.parametrize('firecrawl', [False, True])
def test_explicit_parallel_connection_is_preserved_without_duplicates(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  name: str,
  firecrawl: bool,
) -> None:
  monkeypatch.setenv('MIRROR_MCP_DIR', str(tmp_path))
  monkeypatch.setenv('FIRECRAWL_API_KEY', 'configured-key' if firecrawl else '')
  declaration = {
    'url': 'https://search.parallel.ai/mcp/',
    'auth': {'type': 'bearer', 'tokenEnvVar': 'PARALLEL_API_KEY'},
    'enabledTools': ['web_search'],
  }
  (tmp_path / f'{name}.json').write_text(json.dumps(declaration))
  connections = transport.local_connections(
    on_auth_required=lambda _: pytest.fail('unexpected sign-in')
  ).connections
  assert len(connections) == 1
  assert connections[0].name == name
  assert connections[0].config == declaration


def test_parallel_can_be_disabled_across_reloads(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv('MIRROR_MCP_DIR', str(tmp_path))
  monkeypatch.delenv('FIRECRAWL_API_KEY', raising=False)
  (tmp_path / 'parallel.json').write_text(
    json.dumps({
      'url': 'https://search.parallel.ai/mcp',
      'enabledTools': [],
    })
  )
  for _ in range(2):
    assert (
      transport.local_connections(
        on_auth_required=lambda _: pytest.fail('unexpected sign-in')
      ).connections
      == ()
    )
