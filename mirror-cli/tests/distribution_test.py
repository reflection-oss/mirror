"""Resolve runtime metadata without requiring the development package name."""

import pathlib
from importlib import metadata
from unittest import mock

import pytest

from mirror import distribution


def test_installed_prefers_public_distribution() -> None:
  public = mock.Mock(spec=metadata.Distribution)
  with mock.patch.object(
    metadata, 'distribution', return_value=public
  ) as lookup:
    assert distribution.installed() is public
  lookup.assert_called_once_with('reflection-mirror')


def test_installed_accepts_development_distribution() -> None:
  development = mock.Mock(spec=metadata.Distribution)
  with mock.patch.object(
    metadata,
    'distribution',
    side_effect=[
      metadata.PackageNotFoundError('reflection-mirror'),
      development,
    ],
  ) as lookup:
    assert distribution.installed() is development
  assert lookup.call_args_list == [
    mock.call('reflection-mirror'),
    mock.call('mirror'),
  ]


def test_installed_reports_missing_metadata() -> None:
  with (
    mock.patch.object(
      metadata, 'distribution', side_effect=metadata.PackageNotFoundError
    ),
    pytest.raises(metadata.PackageNotFoundError),
  ):
    distribution.installed()


def test_installed_does_not_hide_broken_public_metadata() -> None:
  with (
    mock.patch.object(
      metadata, 'distribution', side_effect=PermissionError
    ) as lookup,
    pytest.raises(PermissionError),
  ):
    distribution.installed()
  lookup.assert_called_once_with('reflection-mirror')


def test_remote_endpoint_reads_packaged_server(tmp_path: pathlib.Path) -> None:
  (tmp_path / distribution.REMOTE_ENDPOINT).write_text(
    '{"endpoint": "gateway.example:443"}'
  )
  assert distribution.remote_endpoint(tmp_path) == 'gateway.example:443'


def test_remote_endpoint_is_absent_without_packaged_server(
  tmp_path: pathlib.Path,
) -> None:
  assert distribution.remote_endpoint(tmp_path) is None


def test_remote_endpoint_rejects_empty_packaged_server(
  tmp_path: pathlib.Path,
) -> None:
  (tmp_path / distribution.REMOTE_ENDPOINT).write_text('{"endpoint": ""}')
  with pytest.raises(ValueError, match='Invalid packaged remote endpoint'):
    distribution.remote_endpoint(tmp_path)
