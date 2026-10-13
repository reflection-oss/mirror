#!/usr/bin/env bash
# Export, build, and verify a native wheel.
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [output-directory]" >&2
  exit 2
fi
source_dir="$(cd "$(dirname "$0")" && pwd)"
output="${1:-$source_dir/dist}"
mkdir -p "$output"
output="$(cd "$output" && pwd)"
if [[ -f "$source_dir/BUILD.bazel" ]]; then
  if [[ "$(uname -s)" == Darwin ]]; then
    if ! coreutils="$(brew --prefix coreutils 2>/dev/null)" ||
      [[ ! -x "$coreutils/libexec/gnubin/mktemp" ]] || ! command -v flock >/dev/null; then
      echo 'error: install the build launcher prerequisites: brew install coreutils flock' >&2
      exit 1
    fi
    export PATH="$coreutils/libexec/gnubin:$PATH"
    # macOS's default temporary directory is not shared by Colima.
    case "${TMPDIR:-}" in
      '' | /var/folders/* | /private/var/folders/*)
        export TMPDIR="$HOME/.cache/mirror-build"
        mkdir -p "$TMPDIR"
        ;;
    esac
  fi
fi
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
if [[ -f "$source_dir/BUILD.bazel" ]]; then
  cd "$source_dir"
  bazel run --config=local :copybara -- mirror export_release "$source_dir/.." --folder-dir "$scratch/export"
  bash "$scratch/export/mirror-cli/build.sh" "$scratch/wheels"
  bazel run --config=local :copybara -- mirror check-wheel "$scratch"/wheels/*.whl
  cp "$scratch"/wheels/*.whl "$output/"
  if [[ -d "$scratch/wheels/native" ]]; then
    cp -R "$scratch/wheels/native" "$output/"
  fi
  exit
fi
if [[ -f "$source_dir/src/mirror/protocol.py" ]]; then
  echo 'error: export the Python source tree before building distribution wheels.' >&2
  exit 1
fi
case "$(uname -s):$(uname -m)" in
  Darwin:arm64) platform=macosx_15_0_arm64 ;;
  Linux:aarch64) platform=linux_aarch64 ;;
  Linux:x86_64) platform=linux_x86_64 ;;
  *)
    echo 'error: unsupported native wheel platform' >&2
    exit 1
    ;;
esac
flavor="${MIRROR_BUILD_FLAVOR:-public}"
case "$flavor" in
  public)
    if [[ -d "$source_dir/src/mirror/cli/packaged_extensions" ]]; then
      echo 'error: public builds must not contain employee extensions' >&2
      exit 1
    fi
    ;;
  internal) test -d "$source_dir/src/mirror/cli/packaged_extensions" ;;
  *)
    echo 'error: MIRROR_BUILD_FLAVOR must be public or internal' >&2
    exit 2
    ;;
esac
sign_macos_binary() {
  if [[ "$platform" == macosx_15_0_arm64 && -n "${MIRROR_CODESIGN_IDENTITY:-}" ]]; then
    codesign --force --sign "$MIRROR_CODESIGN_IDENTITY" --options runtime --timestamp "$1"
    codesign --verify --strict "$1"
  fi
}
read -r distribution version < <(uv run --no-project --python 3.12 python -c \
  'import pathlib, sys, tomllib; project = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())["project"]; print(project["name"], project["version"])' \
  "$source_dir/pyproject.toml")
echo "Building Mirror $version for $platform"
# Never reuse binaries or configuration from the development checkout.
export CARGO_TARGET_DIR="$scratch/target"
# Pin the embedded tool for repeatable releases; upgrades must pass RPC and dylib checks.
CARGO_PROFILE_RELEASE_STRIP=true cargo install ripgrep --version 14.1.0 --locked \
  --root "$scratch/ripgrep" --message-format=json-render-diagnostics >"$scratch/ripgrep.jsonl"
MIRROR_BUNDLED_RG="$scratch/ripgrep/bin/rg"
export MIRROR_BUNDLED_RG
# Homebrew ripgrep can link unbundled PCRE2; a working build host is not proof of portability.
if [[ "$(uname -s)" == Darwin ]]; then
  otool -L "$MIRROR_BUNDLED_RG" | awk '
    NR > 1 && $1 !~ /^\/usr\/lib\// && $1 !~ /^\/System\/Library\// {
      print "error: ripgrep requires a non-system library: " $1 > "/dev/stderr"
      failed = 1
    }
    END { exit failed }
  '
fi
if [[ -n "${MIRROR_SIGNED_RG:-}" ]]; then
  [[ "$platform" == macosx_15_0_arm64 && -z "${MIRROR_CODESIGN_IDENTITY:-}" ]]
  cp "$MIRROR_SIGNED_RG" "$MIRROR_BUNDLED_RG"
  chmod 755 "$MIRROR_BUNDLED_RG"
  codesign --verify --strict "$MIRROR_BUNDLED_RG"
fi
# The embedded bytes must already contain ripgrep's own signature.
sign_macos_binary "$MIRROR_BUNDLED_RG"
(
  cd "$source_dir/.."
  cargo build --locked --release -p mirror-agent --bin mirror-agent --features bundled-ripgrep \
    --message-format=json-render-diagnostics >"$scratch/worker.jsonl"
)
# Sandboxes mount a Linux sandboxd built for this machine's architecture. Linked
# against glibc 2.17, it runs in any glibc-based guest image.
case "$(uname -m)" in
  arm64 | aarch64) sandboxd_target=aarch64-unknown-linux-gnu ;;
  x86_64) sandboxd_target=x86_64-unknown-linux-gnu ;;
esac
if [[ ! -d "$("${RUSTC:-rustc}" --print sysroot)/lib/rustlib/$sandboxd_target" ]]; then
  echo "error: install the sandboxd target first: rustup target add $sandboxd_target" >&2
  exit 1
fi
# Zig links against that glibc from macOS and from newer build images alike.
zig=(uv tool run --python 3.12 --from cargo-zigbuild==0.23.4 --with ziglang==0.17.0)
(
  cd "$source_dir/.."
  CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
    "${zig[@]}" cargo-zigbuild zigbuild --locked --release -p mirror-sandboxd --bin mirror-sandboxd \
    --target "$sandboxd_target.2.17" --message-format=json-render-diagnostics >"$scratch/sandboxd.jsonl"
)
sandboxd="$CARGO_TARGET_DIR/$sandboxd_target/release/mirror-sandboxd"
zig_package="$("${zig[@]}" python -c 'import os, ziglang; print(os.path.dirname(ziglang.__file__))')"
# Use each build's actual crate sources, including the separately compiled ripgrep.
MIRROR_THIRD_PARTY_NOTICES="$scratch/THIRD_PARTY_NOTICES.txt"
export MIRROR_THIRD_PARTY_NOTICES
uv run --no-project --python 3.12 python "$source_dir/native_notices.py" \
  "$MIRROR_THIRD_PARTY_NOTICES" "$scratch/worker.jsonl" "$scratch/ripgrep.jsonl" \
  --cargo-json "$scratch/sandboxd.jsonl" --zig-runtime "$zig_package" \
  --rust-docs "$("${RUSTC:-rustc}" --print sysroot)/share/doc/rust"
worker="$CARGO_TARGET_DIR/release/mirror-agent"
sign_macos_binary "$worker"
MIRROR_BUILD_RUST_WORKER="$worker" MIRROR_WHEEL_PLATFORM="$platform" MIRROR_BUILD_SANDBOXD="$sandboxd" \
  uv build --python 3.12 --wheel --out-dir "$scratch/wheels" "$source_dir"
UV_TOOL_DIR="$scratch/tools" UV_TOOL_BIN_DIR="$scratch/bin" \
  uv tool install --python 3.12 "$scratch"/wheels/*.whl
uv pip install --python "$scratch/tools/$distribution/bin/python" \
  freezegun pytest pytest-asyncio pyte
(
  cd "$scratch"
  env -u PYTHONPATH -u MIRROR_RUST_WORKER "$scratch/bin/mirror" --help
  env -u PYTHONPATH -u MIRROR_RUST_WORKER MIRROR_TEST_PACKAGED_WHEEL=1 \
    "$scratch/tools/$distribution/bin/python" -m pytest -q \
    "$source_dir/tests/cli/main_test.py::test_installed_worker_tools" \
    "$source_dir/tests/cli/packaged_cli_test.py"
)
cp "$scratch"/wheels/*.whl "$output/"
native_output="${MIRROR_NATIVE_OUTPUT:-}"
if [[ "$platform" == macosx_15_0_arm64 && -n "${MIRROR_CODESIGN_IDENTITY:-}" ]]; then
  native_output="${native_output:-$output/native}"
fi
if [[ -n "$native_output" ]]; then
  # Notarization and ELF auditing need access to the bytes embedded in the worker.
  mkdir -p "$native_output"
  cp "$worker" "$native_output/mirror-agent"
  cp "$MIRROR_BUNDLED_RG" "$native_output/rg"
  cp "$sandboxd" "$native_output/mirror-sandboxd"
  cp "$MIRROR_THIRD_PARTY_NOTICES" "$native_output/THIRD_PARTY_NOTICES.txt"
  cp "$source_dir/../mirror-agent/"{LICENSE,NOTICE,RIPGREP-LICENSE-MIT} "$native_output/"
fi
echo "Runtime-tested Mirror $version wheel (publication checks still required): $output"
