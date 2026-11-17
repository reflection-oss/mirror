"""Build Mirror's generated protobuf and gRPC modules."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import tempfile
import tomllib

import grpc_tools
import setuptools
from grpc_tools import protoc
from setuptools.command import bdist_wheel, build_py, editable_wheel, sdist

_PROTO_PATH = 'mirror/protocol'
_WORKER_PATH = pathlib.Path('mirror/bin/mirror-agent')
_WORKER_LICENSES = ('LICENSE', 'NOTICE', 'RIPGREP-LICENSE-MIT')
# Sandboxes mount this Linux build for the wheel's architecture.
_SANDBOXD_PATH = pathlib.Path('mirror/bin/linux/mirror-sandboxd')


def _worker_source() -> pathlib.Path | None:
  """The optional Rust worker supplied for a platform wheel."""
  value = os.environ.get('MIRROR_BUILD_RUST_WORKER')
  if value is None:
    return None
  source = pathlib.Path(value)
  if not source.is_file():
    raise FileNotFoundError(f'MIRROR_BUILD_RUST_WORKER is not a file: {source}')
  if not os.access(source, os.X_OK):
    raise PermissionError(
      f'MIRROR_BUILD_RUST_WORKER is not executable: {source}'
    )
  return source


def _native_notices() -> pathlib.Path:
  """Require generated attribution whenever a native worker is packaged."""
  value = os.environ.get('MIRROR_THIRD_PARTY_NOTICES')
  if not value:
    raise RuntimeError(
      'Native wheels require MIRROR_THIRD_PARTY_NOTICES; use build.sh'
    )
  source = pathlib.Path(value)
  if not source.read_bytes().strip():
    raise ValueError(f'Native notices are empty: {source}')
  return source


def _sandboxd_source() -> pathlib.Path:
  """Require the Linux sandboxd whenever a native worker is packaged."""
  value = os.environ.get('MIRROR_BUILD_SANDBOXD')
  if not value:
    raise RuntimeError(
      'Native wheels require MIRROR_BUILD_SANDBOXD; use build.sh'
    )
  source = pathlib.Path(value)
  if not source.is_file() or not os.access(source, os.X_OK):
    raise ValueError(
      f'MIRROR_BUILD_SANDBOXD is not an executable file: {source}'
    )
  return source


def _daemon_source() -> pathlib.Path:
  """Require the shared daemon whenever a native worker is packaged."""
  value = os.environ.get('MIRROR_BUILD_DAEMON')
  if not value:
    raise RuntimeError(
      'Native wheels require MIRROR_BUILD_DAEMON; use build.sh'
    )
  source = pathlib.Path(value)
  if not source.is_file() or not os.access(source, os.X_OK):
    raise ValueError(f'MIRROR_BUILD_DAEMON is not an executable file: {source}')
  return source


def _transcript_root() -> pathlib.Path:
  root = pathlib.Path(__file__).parent
  shared = root.parent / 'mirror-transcript'
  return shared if shared.is_dir() else root / 'mirror-transcript'


def _transcript_source() -> pathlib.Path | None:
  value = os.environ.get('MIRROR_BUILD_TRANSCRIPT')
  if value is None:
    if _worker_source() is not None:
      raise RuntimeError(
        'Native wheels require MIRROR_BUILD_TRANSCRIPT; use build.sh'
      )
    return None
  source = pathlib.Path(value).resolve(strict=True)
  version = tomllib.loads((_transcript_root() / 'pyproject.toml').read_text())[
    'project'
  ]['version']
  actual = subprocess.check_output(
    [str(source), '--version'], text=True, timeout=5
  ).strip()
  if actual != f'mirror-transcript {version} protocol 1':
    raise ValueError(f'Native transcript version mismatch: {actual!r}')
  return source


def _proto_sources(root: pathlib.Path) -> tuple[pathlib.Path, ...]:
  """Locate the shared schemas in a checkout or packaged source tree."""
  shared = root.parent / 'protocols' / 'mirror_agent' / 'protos'
  source_root = shared if shared.is_dir() else root / 'protos'
  return tuple(
    source_root / name
    for name in ('agent.proto', 'daemon.proto', 'mirror.proto')
  )


class _BuildPy(build_py.build_py):
  """Generate protobuf modules directly into the wheel build directory."""

  def run(self) -> None:
    # A reused build tree must not bundle bindings from another namespace.
    if pathlib.Path(self.build_lib).exists():
      shutil.rmtree(self.build_lib)
    super().run()
    root = pathlib.Path(__file__).parent
    sources = _proto_sources(root)
    grpc_include = pathlib.Path(grpc_tools.__file__).parent / '_proto'
    with tempfile.TemporaryDirectory() as proto_root_string:
      proto_root = pathlib.Path(proto_root_string)
      staged_root = proto_root / _PROTO_PATH
      staged_root.mkdir(parents=True)
      staged = [staged_root / source.name for source in sources]
      for source, destination in zip(sources, staged, strict=True):
        shutil.copyfile(source, destination)
      result = protoc.main([
        'grpc_tools.protoc',
        f'--proto_path={proto_root}',
        f'--proto_path={grpc_include}',
        f'--python_out={self.build_lib}',
        f'--pyi_out={self.build_lib}',
        f'--grpc_python_out={self.build_lib}',
        *(str(source) for source in staged),
      ])
    if result:
      raise RuntimeError(f'protoc failed with exit code {result}')
    native = pathlib.Path(self.build_lib) / 'mirror_transcript'
    transcript_root = _transcript_root()
    result = protoc.main([
      'grpc_tools.protoc',
      f'--proto_path={transcript_root}',
      f'--python_out={native}',
      f'--pyi_out={native}',
      str(transcript_root / 'transcript.proto'),
    ])
    if result:
      raise RuntimeError(f'transcript protoc failed with exit code {result}')
    shutil.copyfile(transcript_root / 'LICENSE', native / 'LICENSE')
    transcript_binary = _transcript_source()
    if transcript_binary is not None:
      (native / 'bin').mkdir()
      shutil.copy2(transcript_binary, native / 'bin' / 'mirror-transcript')
      shutil.copyfile(
        _native_notices(), native / 'bin' / 'THIRD_PARTY_NOTICES.txt'
      )
    worker = _worker_source()
    if worker is not None:
      destination = pathlib.Path(self.build_lib) / _WORKER_PATH
      destination.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(worker, destination)
      for name in _WORKER_LICENSES:
        shutil.copyfile(
          root.parent / 'mirror-agent' / name, destination.parent / name
        )
      shutil.copyfile(
        _native_notices(), destination.parent / 'THIRD_PARTY_NOTICES.txt'
      )
      sandboxd = pathlib.Path(self.build_lib) / _SANDBOXD_PATH
      sandboxd.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(_sandboxd_source(), sandboxd)
      shutil.copy2(_daemon_source(), destination.with_name('mirror-daemon'))

  def get_outputs(self, include_bytecode: bool = True) -> list[str]:
    outputs = super().get_outputs(include_bytecode)
    generated = pathlib.Path(self.build_lib) / _PROTO_PATH
    outputs.extend(
      str(generated / f'{stem}{suffix}')
      for stem in ('agent', 'daemon', 'mirror')
      for suffix in ('_pb2.py', '_pb2.pyi', '_pb2_grpc.py')
    )
    outputs.extend(
      str(path)
      for path in (pathlib.Path(self.build_lib) / 'mirror_transcript').rglob(
        '*'
      )
      if path.is_file()
    )
    if _worker_source() is not None:
      directory = pathlib.Path(self.build_lib) / _WORKER_PATH.parent
      outputs.extend(
        str(directory / name) for name in (_WORKER_PATH.name, *_WORKER_LICENSES)
      )
      outputs.append(str(directory / 'THIRD_PARTY_NOTICES.txt'))
      outputs.append(str(directory / 'mirror-daemon'))
      outputs.append(str(pathlib.Path(self.build_lib) / _SANDBOXD_PATH))
    return outputs


class _Distribution(setuptools.Distribution):
  """Install native wheels directly into platlib, not a purelib data directory."""

  def has_ext_modules(self) -> bool:
    return (
      _worker_source() is not None
      or _transcript_source() is not None
      or super().has_ext_modules()
    )


class _BdistWheel(bdist_wheel.bdist_wheel):
  """Tag wheels containing the Rust worker for their target platform."""

  def get_tag(self) -> tuple[str, str, str]:
    if _worker_source() is None and _transcript_source() is None:
      return super().get_tag()
    platform = os.environ.get('MIRROR_WHEEL_PLATFORM')
    if platform is None:
      raise RuntimeError(
        'MIRROR_WHEEL_PLATFORM is required with MIRROR_BUILD_RUST_WORKER'
      )
    if not platform.replace('_', '').isalnum():
      raise ValueError(f'invalid MIRROR_WHEEL_PLATFORM: {platform}')
    return 'py3', 'none', platform


class _Sdist(sdist.sdist):
  """Include the shared schema when packaging the Python source tree."""

  def make_release_tree(self, base_dir: str, files: list[str]) -> None:
    super().make_release_tree(base_dir, files)
    transcript = pathlib.Path(base_dir) / 'mirror-transcript'
    transcript.mkdir(exist_ok=True)
    for name in ('LICENSE', 'pyproject.toml', 'transcript.proto'):
      shutil.copyfile(_transcript_root() / name, transcript / name)
    shutil.copytree(
      _transcript_root() / 'python',
      transcript / 'python',
      ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'),
      dirs_exist_ok=True,
    )
    sources = _proto_sources(pathlib.Path(__file__).parent)
    destination = pathlib.Path(base_dir) / 'protos'
    destination.mkdir(parents=True, exist_ok=True)
    for source in sources:
      shutil.copyfile(source, destination / source.name)


class _EditableWheel(editable_wheel.editable_wheel):
  """Use setuptools' link tree so generated modules remain importable."""

  def finalize_options(self) -> None:
    super().finalize_options()
    self.mode = 'strict'


_EXTENSION_PACKAGES = {
  f'mirror.cli.extensions.{entry.parent.name}': f'extensions/{entry.parent.name}'
  for entry in sorted(
    pathlib.Path(__file__).parent.glob('extensions/*/extension.py')
  )
}

setuptools.setup(
  distclass=_Distribution,
  packages=[
    *setuptools.find_namespace_packages('src'),
    *_EXTENSION_PACKAGES,
    'mirror_transcript',
  ],
  package_dir={
    '': 'src',
    **_EXTENSION_PACKAGES,
    'mirror_transcript': os.path.relpath(
      _transcript_root() / 'python' / 'mirror_transcript',
      pathlib.Path(__file__).parent,
    ),
  },
  package_data={'mirror_transcript': ['py.typed']},
  cmdclass={
    'bdist_wheel': _BdistWheel,
    'build_py': _BuildPy,
    'editable_wheel': _EditableWheel,
    'sdist': _Sdist,
  },
)
