"""Resolve runtime metadata without requiring the development package name."""

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
  lookup.assert_called_once_with('mirror-agent')


def test_installed_accepts_development_distribution() -> None:
  development = mock.Mock(spec=metadata.Distribution)
  with mock.patch.object(
    metadata,
    'distribution',
    side_effect=[metadata.PackageNotFoundError('mirror-agent'), development],
  ) as lookup:
    assert distribution.installed() is development
  assert lookup.call_args_list == [
    mock.call('mirror-agent'),
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
  lookup.assert_called_once_with('mirror-agent')
