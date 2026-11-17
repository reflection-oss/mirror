"""Package an admitted host executable and build-time protobuf bindings."""

import os
import pathlib
import shutil
import subprocess
import tomllib

import setuptools
from setuptools.command import bdist_wheel, build_py


class _BuildPy(build_py.build_py):
  def run(self) -> None:
    root = pathlib.Path(__file__).resolve().parent
    binary = pathlib.Path(os.environ['MIRROR_TRANSCRIPT_BINARY']).resolve(
      strict=True
    )
    notices = pathlib.Path(os.environ['MIRROR_TRANSCRIPT_NOTICES']).resolve(
      strict=True
    )
    protoc = pathlib.Path(os.environ['MIRROR_TRANSCRIPT_PROTOC']).resolve(
      strict=True
    )
    compiler = subprocess.check_output(
      [str(protoc), '--version'], text=True, timeout=5
    ).strip()
    if compiler != 'libprotoc 33.4':
      raise ValueError(f'Expected protoc 33.4, got {compiler!r}')
    version = tomllib.loads((root / 'pyproject.toml').read_text())['project'][
      'version'
    ]
    actual = subprocess.check_output(
      [str(binary), '--version'], text=True, timeout=5
    ).strip()
    if actual != f'mirror-transcript {version} protocol 1':
      raise ValueError(f'Native transcript version mismatch: {actual!r}')
    if not notices.read_bytes().strip():
      raise ValueError('Native third-party notices are empty')
    build = pathlib.Path(self.build_lib)
    if build.exists():
      shutil.rmtree(build)
    super().run()
    package = build / 'mirror_transcript'
    subprocess.run(
      [
        str(protoc),
        f'--proto_path={root}',
        f'--python_out={package}',
        f'--pyi_out={package}',
        str(root / 'transcript.proto'),
      ],
      check=True,
      timeout=30,
    )
    destination = package / 'bin'
    destination.mkdir()
    shutil.copy2(binary, destination / 'mirror-transcript')
    shutil.copyfile(notices, destination / 'THIRD_PARTY_NOTICES.txt')

  def get_outputs(self, include_bytecode: bool = True) -> list[str]:
    package = pathlib.Path(self.build_lib) / 'mirror_transcript'
    return super().get_outputs(include_bytecode) + [
      str(package / name)
      for name in (
        'transcript_pb2.py',
        'transcript_pb2.pyi',
        'bin/mirror-transcript',
        'bin/THIRD_PARTY_NOTICES.txt',
      )
    ]


class _Distribution(setuptools.Distribution):
  def has_ext_modules(self) -> bool:
    return True


class _BdistWheel(bdist_wheel.bdist_wheel):
  def get_tag(self) -> tuple[str, str, str]:
    return 'py3', 'none', super().get_tag()[2]


if __name__ == '__main__':
  setuptools.setup(
    distclass=_Distribution,
    cmdclass={'build_py': _BuildPy, 'bdist_wheel': _BdistWheel},
  )
