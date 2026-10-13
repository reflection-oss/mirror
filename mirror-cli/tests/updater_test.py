"""Update detection, isolation, and startup handoff with synthetic releases."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import pty
import shutil
import subprocess
import sys
import zipfile
from unittest import mock

import pytest

from mirror import distribution, updater
from mirror import main as launcher

_OLD = '0.1.0'
_NEW = '0.2.0'
_RELEASE = updater.Release(
  _NEW,
  'https://files.pythonhosted.org/packages/mirror_agent-0.2.0-py3-none-any.whl',
  'a' * 64,
)


@pytest.fixture(autouse=True)
def isolated_home(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'data'))
  monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
  monkeypatch.delenv('MIRROR_SKIP_UPDATE', raising=False)
  monkeypatch.delenv('MIRROR_RUST_WORKER', raising=False)


def _enable(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
  public = mock.Mock(version=_OLD)
  public.locate_file.return_value = tmp_path / 'updater.py'
  monkeypatch.setattr(updater, '__file__', str(tmp_path / 'updater.py'))
  monkeypatch.setattr(
    updater.importlib.metadata, 'distribution', lambda _: public
  )


def _lock(release: updater.Release = _RELEASE) -> bytes:
  return (
    f'[[packages]]\nname = "mirror-agent"\nversion = "{release.version}"\n'
    f'wheels = [{{url = "{release.url}", hashes = {{sha256 = "{release.sha256}"}}}}]\n'
  ).encode()


def test_resolution_uses_only_pypi() -> None:
  with mock.patch.object(updater, '_run', return_value=_lock()) as run:
    assert updater.latest(_OLD, timeout=3) == _RELEASE
  command = run.call_args.args[0]
  assert (
    command[command.index('--default-index') + 1] == 'https://pypi.org/simple/'
  )
  assert '--no-deps' in command
  assert run.call_args.kwargs['stdin'] == b'mirror-agent>=0.1.0\n'
  assert run.call_args.kwargs['timeout'] == 3


@pytest.mark.parametrize(
  'url',
  [
    'https://pypi.org/mirror.whl',
    'https://files.pythonhosted.org/packages/../other/mirror.whl',
    'https://files.pythonhosted.org/packages/%2e%2e/other/mirror.whl',
    'https://files.pythonhosted.org/packages/mirror.whl?token=SYNTHETIC_SECRET',
  ],
)
def test_candidate_cannot_change_source(url: str) -> None:
  with mock.patch.object(
    updater, '_run', return_value=_lock(dataclasses.replace(_RELEASE, url=url))
  ):
    with pytest.raises(ValueError):
      updater.latest(_OLD, timeout=3)


@pytest.mark.parametrize('version', [_OLD, '0.0.9'])
def test_equal_and_older_versions_are_not_offered(version: str) -> None:
  with mock.patch.object(
    updater,
    '_run',
    return_value=_lock(dataclasses.replace(_RELEASE, version=version)),
  ):
    assert updater.latest(_OLD, timeout=3) is None


@pytest.mark.parametrize(
  'payload',
  [
    b'not toml',
    b'packages=[]',
    b'packages=[1]',
    _lock(dataclasses.replace(_RELEASE, sha256='bad')),
  ],
)
def test_invalid_resolution_is_rejected(payload: bytes) -> None:
  with mock.patch.object(updater, '_run', return_value=payload):
    with pytest.raises((TypeError, ValueError)):
      updater.latest(_OLD, timeout=3)


def test_source_checkout_does_not_enable_updates(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr(updater, '__file__', str(tmp_path / 'updater.py'))
  with mock.patch.object(updater, 'latest') as check:
    assert updater.prepare(
      check=True, force=True, interactive=True
    ) == updater.Launch(pathlib.Path(sys.executable))
    check.assert_not_called()


@pytest.mark.parametrize(
  'variable', ['MIRROR_SKIP_UPDATE', 'MIRROR_RUST_WORKER']
)
def test_development_bypasses_updates(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
  _enable(tmp_path, monkeypatch)
  monkeypatch.setenv(variable, '1')
  with mock.patch.object(updater, '_installed_version') as read:
    assert updater.prepare(
      check=True, force=True, interactive=True
    ) == updater.Launch(pathlib.Path(sys.executable))
    read.assert_not_called()


@pytest.mark.parametrize('answer', ['', 'n', 'no'])
def test_decline_starts_current_without_installing(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
  _enable(tmp_path, monkeypatch)
  with (
    mock.patch.object(updater, 'latest', return_value=_RELEASE),
    mock.patch('builtins.input', return_value=answer),
    mock.patch.object(updater, 'install') as install,
  ):
    assert updater.prepare(
      check=True, force=False, interactive=True
    ) == updater.Launch(pathlib.Path(sys.executable))
    install.assert_not_called()


def test_accept_prepares_before_handoff(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _enable(tmp_path, monkeypatch)
  selected = tmp_path / 'release' / 'bin' / 'python'
  with (
    mock.patch.object(updater, 'latest', return_value=_RELEASE) as check,
    mock.patch('builtins.input', return_value='yes'),
    mock.patch.object(updater, 'install', return_value=selected) as install,
  ):
    assert updater.prepare(
      check=True, force=True, interactive=True
    ) == updater.Launch(selected)
    assert check.call_args.kwargs['timeout'] == 120
    install.assert_called_once()


def test_noninteractive_does_not_check_or_prompt(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _enable(tmp_path, monkeypatch)
  with mock.patch.object(updater, 'latest') as check:
    updater.prepare(check=True, force=True, interactive=False)
    check.assert_not_called()


def test_failure_is_payload_free_and_falls_back(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
) -> None:
  _enable(tmp_path, monkeypatch)
  with mock.patch.object(
    updater, 'latest', side_effect=RuntimeError('SYNTHETIC_SECRET')
  ):
    assert updater.prepare(
      check=True, force=True, interactive=True
    ) == updater.Launch(pathlib.Path(sys.executable))
  output = capsys.readouterr()
  assert 'starting the existing release' in output.err
  assert 'SYNTHETIC_SECRET' not in output.err


def test_subprocess_environment_and_stderr_do_not_leak(
  monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
  monkeypatch.setenv('REFLECTION_API_KEY', 'SYNTHETIC_SECRET')
  monkeypatch.setenv('UV_INDEX', 'https://unexpected.example.test/')
  result = updater._run(
    (
      sys.executable,
      '-c',
      'import os, sys; assert "REFLECTION_API_KEY" not in os.environ; '
      'assert "UV_INDEX" not in os.environ; '
      'print("SYNTHETIC_SECRET", file=sys.stderr); print("ok")',
    ),
  )
  assert result == b'ok\n'
  assert 'SYNTHETIC_SECRET' not in ''.join(capfd.readouterr())


def test_timeout_kills_and_reaps_process() -> None:
  with mock.patch.object(updater.os, 'killpg', wraps=updater.os.killpg) as kill:
    with pytest.raises(subprocess.TimeoutExpired):
      updater._run(
        (sys.executable, '-c', 'import time; time.sleep(60)'),
        timeout=0.01,
      )
    kill.assert_called_once()


def _wheel(path: pathlib.Path, version: str) -> pathlib.Path:
  wheel = path / f'mirror_agent-{version}-py3-none-any.whl'
  metadata = f'mirror_agent-{version}.dist-info'
  with zipfile.ZipFile(wheel, 'w') as archive:
    archive.writestr(
      metadata + '/METADATA',
      f'Metadata-Version: 2.1\nName: mirror-agent\nVersion: {version}\nRequires-Dist: packaging\n',
    )
    archive.writestr(
      metadata + '/WHEEL',
      'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
    )
    archive.writestr(metadata + '/RECORD', '')
    archive.writestr(
      'mirror/main.py', pathlib.Path(launcher.__file__).read_text()
    )
    archive.writestr(
      'mirror/updater.py', pathlib.Path(updater.__file__).read_text()
    )
    archive.writestr(
      'mirror/cli/main.py',
      'import json, pathlib, sys\n'
      f'print(json.dumps({{"version": "{version}", "argv": sys.argv[1:], "cwd": str(pathlib.Path.cwd())}}))\n',
    )
  return wheel


def _installer(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, wheel: pathlib.Path
) -> None:
  uv = shutil.which('uv')
  assert uv is not None
  # This fixture translates one synthetic HTTPS wheel URL to a local wheel;
  # real uv installs the environment and public dependencies, without authentication.
  index = tmp_path / 'index' / 'mirror-agent'
  index.mkdir(parents=True, exist_ok=True)
  digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
  (index / 'index.html').write_text(
    f'<a href="{wheel.as_uri()}#sha256={digest}">{wheel.name}</a>'
  )
  shim = tmp_path / 'shim'
  shim.mkdir(exist_ok=True)
  executable = shim / 'uv'
  executable.write_text(
    f'#!{sys.executable}\n'
    'import os, subprocess, sys, tomllib\n'
    'args = sys.argv[1:]\n'
    'if "compile" in args:\n'
    f'  args[args.index("--default-index") + 1] = {index.parent.as_uri()!r}\n'
    f'  result = subprocess.run([{uv!r}, *args], input=sys.stdin.read(), text=True, capture_output=True)\n'
    '  if result.returncode: sys.exit(result.returncode)\n'
    '  url = tomllib.loads(result.stdout)["packages"][0]["wheels"][0]["url"]\n'
    f'  print(result.stdout.replace(url, {_RELEASE.url!r})); sys.exit(0)\n'
    'if "pip" in args:\n'
    f'  requirement = sys.stdin.read().replace({_RELEASE.url!r}, {wheel.as_uri()!r})\n'
    f'  sys.exit(subprocess.run([{uv!r}, *args], input=requirement, text=True).returncode)\n'
    f'os.execv({uv!r}, [{uv!r}, *args])\n'
  )
  executable.chmod(0o755)
  monkeypatch.setenv('PATH', f'{shim}{os.pathsep}{os.environ["PATH"]}')


def test_real_side_by_side_install_and_selected_offline_start(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  wheel = _wheel(tmp_path, _NEW)
  _installer(tmp_path, monkeypatch, wheel)
  root = tmp_path / 'releases'
  old_config = tmp_path / '.mirror' / 'config.json'
  old_config.parent.mkdir()
  old_config.write_text('{"keep": true}')
  old_session = old_config.with_name('state.db')
  old_session.write_bytes(b'old-session')
  release = dataclasses.replace(
    _RELEASE, sha256=hashlib.sha256(wheel.read_bytes()).hexdigest()
  )
  python = updater.install(release, root, _OLD)
  result = subprocess.run(
    [str(python), '-m', 'mirror.cli.main', '--continue'],
    check=True,
    capture_output=True,
    text=True,
  )
  assert json.loads(result.stdout)['version'] == _NEW
  assert json.loads(result.stdout)['argv'] == ['--continue']
  assert python != pathlib.Path(sys.executable)
  assert distribution.installed().version == '0.1.0'
  assert old_config.read_text() == '{"keep": true}'
  assert old_session.read_bytes() == b'old-session'
  _enable(tmp_path, monkeypatch)
  with (
    mock.patch.object(updater, '_root', return_value=root),
    mock.patch.object(updater, 'latest') as check,
  ):
    assert updater.prepare(
      check=False, force=False, interactive=False
    ) == updater.Launch(python, 'mirror.main')
    check.assert_not_called()


def test_install_lock_prevents_concurrent_preparation(
  tmp_path: pathlib.Path,
) -> None:
  with (tmp_path / 'install.lock').open('a') as lock:
    updater.fcntl.flock(lock, updater.fcntl.LOCK_EX)
    with mock.patch.object(updater, '_run') as run:
      with pytest.raises(RuntimeError, match='Another update'):
        updater.install(_RELEASE, tmp_path, _OLD)
      run.assert_not_called()


def test_real_startup_accepts_update_and_runs_it_without_relaunch(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  uv = shutil.which('uv')
  assert uv is not None
  original = _wheel(tmp_path, _OLD)
  new = _wheel(tmp_path, _NEW)
  base = tmp_path / 'original'
  updater._run((
    uv,
    '--no-config',
    'venv',
    '--python',
    sys.executable,
    str(base),
  ))
  python = base / 'bin' / 'python'
  updater._run(
    (
      uv,
      '--no-config',
      'pip',
      'install',
      '--default-index',
      'https://pypi.org/simple/',
      '--python',
      str(python),
      str(original),
    ),
  )
  _installer(tmp_path, monkeypatch, new)
  workspace = tmp_path / 'workspace'
  workspace.mkdir()
  master, slave = pty.openpty()
  try:
    with subprocess.Popen(
      [str(python), '-m', 'mirror.main', '--update', '--continue'],
      stdin=slave,
      stderr=slave,
      stdout=subprocess.PIPE,
      cwd=workspace,
      env=updater._environment(),
    ) as process:
      os.write(master, b'y\n')
      output, _ = process.communicate(timeout=30)
      assert process.returncode == 0
      assert json.loads(output) == {
        'version': _NEW,
        'argv': ['--continue'],
        'cwd': str(workspace.resolve()),
      }
  finally:
    os.close(master)
    os.close(slave)
  original_output = updater._run((str(python), '-m', 'mirror.cli.main'))
  assert json.loads(original_output)['version'] == _OLD
  # The next noninteractive invocation uses the selected release without a check.
  cached = updater._run((str(python), '-m', 'mirror.main', '--version'))
  assert json.loads(cached)['version'] == _NEW


def test_bad_hash_does_not_change_selection(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  wheel = _wheel(tmp_path, _NEW)
  _installer(tmp_path, monkeypatch, wheel)
  root = tmp_path / 'releases'
  with pytest.raises(RuntimeError):
    updater.install(_RELEASE, root, _OLD)
  assert not (root / 'selected').exists()
  assert not (root / _RELEASE.directory).exists()


def test_failed_update_keeps_selected_environment(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  wheel = _wheel(tmp_path, _NEW)
  _installer(tmp_path, monkeypatch, wheel)
  root = tmp_path / 'releases'
  release = dataclasses.replace(
    _RELEASE, sha256=hashlib.sha256(wheel.read_bytes()).hexdigest()
  )
  previous = updater.install(release, root, _OLD)
  selected = (root / 'selected').read_bytes()
  newer = dataclasses.replace(release, version='0.3.0')
  with mock.patch.object(
    updater, '_run', side_effect=RuntimeError('fixture failure')
  ):
    with pytest.raises(RuntimeError):
      updater.install(newer, root, _OLD)
  assert (root / 'selected').read_bytes() == selected
  assert previous.is_file()
  assert (root / release.directory / 'ready').is_file()


def test_completed_environment_is_reused_and_not_downgraded(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  wheel = _wheel(tmp_path, _NEW)
  _installer(tmp_path, monkeypatch, wheel)
  release = dataclasses.replace(
    _RELEASE, sha256=hashlib.sha256(wheel.read_bytes()).hexdigest()
  )
  root = tmp_path / 'releases'
  python = updater.install(release, root, _OLD)
  with mock.patch.object(updater, '_run') as run:
    assert updater.install(release, root, _OLD) == python
    assert (
      updater.install(dataclasses.replace(release, version=_OLD), root, _OLD)
      == python
    )
    run.assert_not_called()
