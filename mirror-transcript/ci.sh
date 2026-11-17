#!/usr/bin/env bash
# Source/process checks; installed-wheel qualification is a separate gate.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
temporary="$(mktemp -d)"
trap 'rm -rf "$temporary"' EXIT
mkdir -p "$temporary/python/mirror_transcript" "$temporary/tests"
cp "$here"/python/mirror_transcript/*.py "$temporary/python/mirror_transcript/"
cp "$here"/tests/*.py "$temporary/tests/"
cp "$here"/{pyproject.toml,uv.lock,setup.py} "$temporary/"
sources=("$temporary"/python/mirror_transcript/*.py "$temporary"/tests/*.py "$temporary/setup.py")
"${PROTOC:?}" --proto_path="$here" --python_out="$temporary/python/mirror_transcript" \
  --pyi_out="$temporary/python/mirror_transcript" "$here/transcript.proto"
export UV_PROJECT_ENVIRONMENT="$temporary/venv"
unset VIRTUAL_ENV
"${UV_BIN:-uv}" sync --project "$temporary" --locked --all-extras --no-install-project --python 3.12
export PYTHONPATH="$temporary/python"
"$temporary/venv/bin/pyright" --project "$temporary/pyproject.toml" --pythonpath "$temporary/venv/bin/python" "${sources[@]}"
"$temporary/venv/bin/python" -m pytest "$temporary/tests" "$@"
