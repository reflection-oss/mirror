"""Exercise uv's schema cache keys without installing or downloading packages."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

import pytest


def _compile(project: pathlib.Path) -> subprocess.CompletedProcess[str]:
  uv = os.environ.get('UV_BIN') or shutil.which('uv')
  assert uv is not None, 'uv is required for packaging tests'
  return subprocess.run(
    [
      str(pathlib.Path(uv).resolve()),
      'pip',
      'compile',
      '--offline',
      '--no-build-isolation',
      '--no-deps',
      '--python',
      sys.executable,
      '--cache-dir',
      str(project.parent / 'cache'),
      '-',
    ],
    input=f'cache-probe @ {project.as_uri()}\n',
    cwd=project,
    # Interpreter-discovery warnings are unrelated to schema cache diagnostics.
    env={**os.environ, 'RUST_LOG': 'uv_cache_info=debug'},
    capture_output=True,
    text=True,
    check=True,
    timeout=30,
  )


@pytest.mark.parametrize(
  'layout', ['../protocols/mirror_agent/protos', 'protos']
)
@pytest.mark.parametrize(
  'schema_name', ['agent.proto', 'client.proto', 'mirror.proto']
)
def test_schema_cache_keys(
  tmp_path: pathlib.Path, layout: str, schema_name: str
) -> None:
  project = tmp_path / 'mirror-cli'
  (project / 'src').mkdir(parents=True)
  (project / 'extensions').mkdir()
  schema = project / layout / schema_name
  schema.parent.mkdir(parents=True, exist_ok=True)
  schema.write_text('syntax = "proto3";\n')
  config = (pathlib.Path(__file__).parents[1] / 'pyproject.toml').read_text()
  cache_config = config.split('[tool.uv]', 1)[1].split('\n[', 1)[0]
  (project / 'pyproject.toml').write_text(
    '[build-system]\nrequires = []\nbuild-backend = "setup"\n'
    'backend-path = ["."]\n[project]\nname = "cache-probe"\n'
    'dynamic = ["version"]\n[tool.uv]\n' + cache_config
  )
  # Dynamic metadata exercises uv's cache without depending on a build backend.
  (project / 'setup.py').write_text(f"""from pathlib import Path

def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    source = Path({str(schema)!r}).read_text()
    Path('built.txt').write_text(source)
    destination = Path(metadata_directory) / 'cache_probe-1.0.dist-info'
    destination.mkdir()
    (destination / 'METADATA').write_text(
        'Metadata-Version: 2.3\\nName: cache-probe\\nVersion: 1.0\\n'
    )
    return destination.name
""")

  first = _compile(project)
  assert 'WARN' not in first.stderr, first.stderr
  built = project / 'built.txt'
  assert built.read_text() == schema.read_text()
  built_at = built.stat().st_mtime_ns
  cached = _compile(project)
  assert 'WARN' not in cached.stderr, cached.stderr
  assert built.stat().st_mtime_ns == built_at

  schema.write_text('syntax = "proto3";\nmessage Added {}\n')
  changed = _compile(project)
  assert 'WARN' not in changed.stderr, changed.stderr
  assert built.read_text() == schema.read_text()
