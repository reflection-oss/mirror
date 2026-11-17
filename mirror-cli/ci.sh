#!/usr/bin/env bash
# Standalone build + test of the mirror package via uv, independent of the
# monorepo and Bazel. This is what a public/open-source CI would run, and it
# exercises the package exactly as an installed wheel (the point of src-layout).

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo 'error: uv is required (install via mise or a setup-uv CI step).' >&2
  exit 1
fi

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

# Copy into a scratch dir so tests import the *installed* mirror (from the
# built wheel), not the local `src/` tree.
uv run --no-project --python 3.12 python - "$SOURCE_DIR" "$WORK_DIR" <<'PY'
import pathlib
import shutil
import sys

source, checkout = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
dest = checkout / 'mirror-cli'
dest.mkdir()
skip = {'.git', '.venv', '.pytest_cache', '__pycache__', 'build', 'dist'}
ignore = shutil.ignore_patterns(*skip, '*.egg-info', 'BUILD', 'BUILD.bazel')
for child in source.iterdir():
  if child.name in skip or child.name.endswith('.egg-info'):
    continue
  target = dest / child.name
  if child.is_dir():
    shutil.copytree(child, target, ignore=ignore, dirs_exist_ok=True)
  else:
    shutil.copy2(child, target)
shutil.copytree(
  source.parent / 'mirror-transcript',
  checkout / 'mirror-transcript',
  ignore=ignore,
)
shutil.copytree(
  source.parent / 'protocols' / 'mirror_agent',
  checkout / 'protocols' / 'mirror_agent',
  ignore=ignore,
)
PY

cd "$WORK_DIR/mirror-cli"
# Check both distributable archives, not only an editable installation.
uv build --sdist --wheel --out-dir "$WORK_DIR/dist"
uv run --no-project --python 3.12 python - "$WORK_DIR/dist" <<'PYTHON'
import pathlib
import sys
import tarfile
import zipfile

artifacts = pathlib.Path(sys.argv[1])
with tarfile.open(next(artifacts.glob('*.tar.gz'))) as source:
  for name in ('LICENSE', 'NOTICE'):
    members = [member for member in source.getmembers() if pathlib.PurePosixPath(member.name).parts[1:] == (name,)]
    assert len(members) == 1, name
    content = source.extractfile(members[0])
    assert content is not None
    with content:
      assert content.read() == pathlib.Path(name).read_bytes()
with zipfile.ZipFile(next(artifacts.glob('*.whl'))) as wheel:
  for name in ('LICENSE', 'NOTICE'):
    members = [member for member in wheel.namelist() if member.endswith(f'.dist-info/licenses/{name}')]
    assert len(members) == 1, name
    assert wheel.read(members[0]) == pathlib.Path(name).read_bytes()
PYTHON
if [[ -n "${REFLECTION_CI_WHEEL_DIR:-}" ]]; then
  mkdir -p "$REFLECTION_CI_WHEEL_DIR"
  cp "$WORK_DIR"/dist/*.whl "$REFLECTION_CI_WHEEL_DIR/"
fi
# Exercise the native packaging hook without compiling or executing a worker.
mkdir -p ../mirror-agent
for name in LICENSE NOTICE RIPGREP-LICENSE-MIT; do
  printf 'native %s fixture\n' "$name" >"../mirror-agent/$name"
done
printf 'dependency notice fixture\n' >"$WORK_DIR/THIRD_PARTY_NOTICES.txt"
printf '#!/bin/sh\nexit 0\n' >"$WORK_DIR/worker"
chmod +x "$WORK_DIR/worker"
printf '#!/bin/sh\nexit 0\n' >"$WORK_DIR/sandboxd"
chmod +x "$WORK_DIR/sandboxd"
transcript_version="$(uv run --no-project --python 3.12 python -c 'import pathlib,tomllib; print(tomllib.loads(pathlib.Path("../mirror-transcript/pyproject.toml").read_text())["project"]["version"])')"
printf '#!/bin/sh\nprintf "mirror-transcript %s protocol 1\\n"\n' "$transcript_version" >"$WORK_DIR/transcript"
chmod +x "$WORK_DIR/transcript"
export MIRROR_BUILD_TRANSCRIPT="$WORK_DIR/transcript"
# Native packaging must fail closed even when build.sh is bypassed.
if env -u MIRROR_BUILD_TRANSCRIPT MIRROR_BUILD_RUST_WORKER="$WORK_DIR/worker" \
  MIRROR_WHEEL_PLATFORM=linux_x86_64 uv build --wheel --out-dir "$WORK_DIR/missing-transcript" \
  >"$WORK_DIR/missing-transcript.log" 2>&1; then
  echo 'error: native wheel accepted a missing transcript renderer' >&2
  exit 1
fi
grep -q 'Native wheels require MIRROR_BUILD_TRANSCRIPT' "$WORK_DIR/missing-transcript.log"
if env -u MIRROR_THIRD_PARTY_NOTICES MIRROR_BUILD_RUST_WORKER="$WORK_DIR/worker" \
  MIRROR_WHEEL_PLATFORM=linux_x86_64 uv build --wheel --out-dir "$WORK_DIR/missing-notices" \
  >"$WORK_DIR/missing-notices.log" 2>&1; then
  echo 'error: native wheel accepted missing notices' >&2
  exit 1
fi
grep -q 'Native wheels require MIRROR_THIRD_PARTY_NOTICES' "$WORK_DIR/missing-notices.log"
printf '  \n' >"$WORK_DIR/empty-notices.txt"
if MIRROR_THIRD_PARTY_NOTICES="$WORK_DIR/empty-notices.txt" MIRROR_BUILD_RUST_WORKER="$WORK_DIR/worker" \
  MIRROR_WHEEL_PLATFORM=linux_x86_64 uv build --wheel --out-dir "$WORK_DIR/empty-notices" \
  >"$WORK_DIR/empty-notices.log" 2>&1; then
  echo 'error: native wheel accepted empty notices' >&2
  exit 1
fi
grep -q 'Native notices are empty' "$WORK_DIR/empty-notices.log"
if MIRROR_THIRD_PARTY_NOTICES="$WORK_DIR/THIRD_PARTY_NOTICES.txt" MIRROR_BUILD_RUST_WORKER="$WORK_DIR/worker" \
  MIRROR_WHEEL_PLATFORM=linux_x86_64 uv build --wheel --out-dir "$WORK_DIR/missing-sandboxd" \
  >"$WORK_DIR/missing-sandboxd.log" 2>&1; then
  echo 'error: native wheel accepted a missing sandboxd' >&2
  exit 1
fi
grep -q 'Native wheels require MIRROR_BUILD_SANDBOXD' "$WORK_DIR/missing-sandboxd.log"
MIRROR_BUILD_RUST_WORKER="$WORK_DIR/worker" MIRROR_WHEEL_PLATFORM=linux_x86_64 \
  MIRROR_THIRD_PARTY_NOTICES="$WORK_DIR/THIRD_PARTY_NOTICES.txt" MIRROR_BUILD_SANDBOXD="$WORK_DIR/sandboxd" MIRROR_BUILD_DAEMON="$WORK_DIR/worker" \
  uv build --wheel --out-dir "$WORK_DIR/native"
uv run --no-project --python 3.12 python - "$WORK_DIR" <<'PYTHON'
import pathlib
import sys
import zipfile

root = pathlib.Path(sys.argv[1])
with zipfile.ZipFile(next((root / 'native').glob('*.whl'))) as wheel:
  assert wheel.read('mirror/bin/mirror-agent') == (root / 'worker').read_bytes()
  daemon = wheel.getinfo('mirror/bin/mirror-daemon')
  assert wheel.read(daemon) == (root / 'worker').read_bytes()
  assert daemon.external_attr >> 16 & 0o111
  for name in ('LICENSE', 'NOTICE', 'RIPGREP-LICENSE-MIT'):
    assert wheel.read(f'mirror/bin/{name}') == (root / 'mirror-agent' / name).read_bytes()
  assert wheel.read('mirror/bin/THIRD_PARTY_NOTICES.txt') == (root / 'THIRD_PARTY_NOTICES.txt').read_bytes()
  transcript = wheel.getinfo('mirror_transcript/bin/mirror-transcript')
  assert wheel.read(transcript) == (root / 'transcript').read_bytes()
  assert transcript.external_attr >> 16 & 0o111
  assert wheel.read('mirror_transcript/bin/THIRD_PARTY_NOTICES.txt') == (root / 'THIRD_PARTY_NOTICES.txt').read_bytes()
  sandboxd = wheel.getinfo('mirror/bin/linux/mirror-sandboxd')
  assert wheel.read(sandboxd) == (root / 'sandboxd').read_bytes()
  assert sandboxd.external_attr >> 16 & 0o111
PYTHON
unset MIRROR_BUILD_TRANSCRIPT
# An sdist must rebuild without the sibling checkout or pre-generated bindings.
uv run --no-project --python 3.12 python - "$WORK_DIR" <<'PYTHON'
import pathlib
import sys
import tarfile

root = pathlib.Path(sys.argv[1])
with tarfile.open(next((root / 'dist').glob('*.tar.gz'))) as archive:
  archive.extractall(root / 'sdist', filter='data')
PYTHON
uv build --wheel --out-dir "$WORK_DIR/from-sdist" "$WORK_DIR"/sdist/*
# A prior namespace's generated files must not leak into a rebuilt wheel.
mkdir -p build/lib/obsolete_protocol
printf '# stale generated binding\n' >build/lib/obsolete_protocol/agent_pb2.py
uv sync --locked --no-editable --python 3.12
VENV_DIR="$PWD/.venv"
"$VENV_DIR/bin/pyright" --pythonpath "$VENV_DIR/bin/python" src \
  extensions/voice/extension.py extensions/voice/process.py extensions/voice/space_hold.py \
  extensions/tokyonight/extension.py
"$VENV_DIR/bin/python" - <<'PY'
import importlib
import pathlib

for source in sorted(pathlib.Path('src').rglob('*.py')):
  if source.stem == '__main__' or source.stem.endswith('_test'):
    continue
  parts = source.relative_to('src').with_suffix('').parts
  module = '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)
  importlib.import_module(module)
PY
# The Python-only packaging fixtures are not executable transcript substitutes.
if [[ -z "${MIRROR_TRANSCRIPT_BIN:-}" ]]; then
  (
    cd "$SOURCE_DIR/.."
    cargo build --locked -p mirror-transcript --bin mirror-transcript --target-dir "$WORK_DIR/target"
  )
  MIRROR_TRANSCRIPT_BIN="$WORK_DIR/target/debug/mirror-transcript"
fi
export MIRROR_TRANSCRIPT_BIN
[[ -x "$MIRROR_TRANSCRIPT_BIN" ]]
"$MIRROR_TRANSCRIPT_BIN" --version
# Unset the API key so live tests skip — the default suite stays hermetic.
if (($#)); then
  test_paths=("$@")
else
  test_paths=(src tests extensions)
fi
collected=0
run_pytest() {
  local status=0
  # Tests bypass the launcher; keep its gRPC fork handlers disabled here too.
  # Report the stuck test and its stacks before the enclosing CI job times out.
  env -u OPENROUTER_API_KEY PYTHONDONTWRITEBYTECODE=1 GRPC_ENABLE_FORK_SUPPORT=0 \
    "$VENV_DIR/bin/pytest" -v -p no:cacheprovider "$@" \
    -o faulthandler_timeout=120 -o faulthandler_exit_on_timeout=true "${test_paths[@]}" || status=$?
  # Exit 5 means the pass's marker selected none of the given tests.
  ((status == 5)) && return
  ((status == 0)) || exit "$status"
  collected=1
}
# Spread the tests over one pytest-xdist worker per CPU, then run alone the
# tests whose timing bounds miss under that contention.
run_pytest -n auto -m 'not no_xdist'
run_pytest -m no_xdist
if ((!collected)); then
  echo 'error: no tests collected' >&2
  exit 5
fi
