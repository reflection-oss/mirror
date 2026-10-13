from __future__ import annotations

import json
import pathlib

import pytest

from mirror.cli.extensions.connections import transport


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
  assert connections.connected_servers == {'public'}
  assert pending == ['private']
