#!/usr/bin/env bash
# Export, build, and verify a native wheel, or stage a Linux agent for ABI auditing.
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [output-directory]" >&2
  exit 2
fi
source_dir="$(cd "$(dirname "$0")" && pwd)"
component="${MIRROR_BUILD_COMPONENT:-cli}"
case "$component" in
  cli) ;;
  agent)
    if [[ -f "$source_dir/BUILD.bazel" || "$(uname -s)" != Linux ]]; then
      echo 'error: agent mode requires an exported source tree and native Linux' >&2
      exit 2
    fi
    ;;
  *)
    echo 'error: MIRROR_BUILD_COMPONENT must be cli or agent' >&2
    exit 2
    ;;
esac
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
# Background cargo builds outlive set -e; kill their process trees before removing scratch.
kill_tree() {
  local child children
  children="$(pgrep -P "$1" 2>/dev/null || true)"
  kill -TERM "$1" 2>/dev/null || true
  for child in $children; do kill_tree "$child"; done
}
cleanup() {
  local job
  for job in $(jobs -pr); do kill_tree "$job"; done
  wait 2>/dev/null || true
  rm -rf "$scratch"
}
trap cleanup EXIT
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
  Darwin:arm64)
    # Rust and its C dependencies must target the wheel's runtime floor, not the SDK/host.
    export MACOSX_DEPLOYMENT_TARGET=14.0
    platform=macosx_14_0_arm64
    ;;
  Linux:aarch64) platform=linux_aarch64 ;;
  Linux:x86_64) platform=linux_x86_64 ;;
  *)
    echo 'error: unsupported native wheel platform' >&2
    exit 1
    ;;
esac
packaged_tests="${MIRROR_PACKAGED_TESTS:-raw}"
case "$packaged_tests" in
  raw | final) ;;
  *)
    echo 'error: MIRROR_PACKAGED_TESTS must be raw or final' >&2
    exit 2
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
check_macos_binary() {
  [[ "$(uname -s)" == Darwin ]] || return 0
  [[ "$(lipo -archs "$1")" == arm64 ]]
  # Check the runtime floor, not the SDK version; signed RG may come from another job.
  otool -l "$1" | awk '
    $1 == "platform" { platform = $2 }
    $1 == "minos" {
      count++
      split($2, version, ".")
      if (platform != 1 || version[1] > 14 ||
          (version[1] == 14 && (version[2] > 0 || version[3] > 0))) failed = 1
    }
    END { exit (count != 1 || failed) }
  '
  otool -L "$1" | awk '
    NR > 1 && $1 !~ /^\/usr\/lib\// && $1 !~ /^\/System\/Library\// {
      print "error: binary requires a non-system library: " $1 > "/dev/stderr"
      failed = 1
    }
    END { exit failed }
  '
}
sign_macos_binary() {
  if [[ "$platform" == macosx_14_0_arm64 && -n "${MIRROR_CODESIGN_IDENTITY:-}" ]]; then
    codesign --force --sign "$MIRROR_CODESIGN_IDENTITY" --options runtime --timestamp "$1"
    codesign --verify --strict "$1"
  fi
}
if [[ "$component" == cli ]]; then
  read -r distribution version < <(uv run --no-project --python 3.12 python -c \
    'import pathlib, sys, tomllib; project = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())["project"]; print(project["name"], project["version"])' \
    "$source_dir/pyproject.toml")
  echo "Building Mirror $version for $platform"
else
  echo "Building standalone Mirror agent for $platform"
fi
# Never reuse binaries or configuration from the development checkout.
export CARGO_TARGET_DIR="$scratch/target"
cargo_build() {
  local target_dir="$1" log="$2"
  shift 2
  (
    cd "$source_dir/.."
    CARGO_TARGET_DIR="$target_dir" "$@" --message-format=json-render-diagnostics >"$log"
  )
}
if [[ "$component" == cli ]]; then
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
  zig_package="$("${zig[@]}" python -c 'import os, ziglang; print(os.path.dirname(ziglang.__file__))')"
  cargo_build "$scratch/target-daemon" "$scratch/daemon.jsonl" \
    cargo build --locked --release -p mirror-daemon --bin mirror-daemon &
  daemon_build=$!
  cargo_build "$scratch/target-transcript" "$scratch/transcript.jsonl" \
    cargo build --locked --release -p mirror-transcript --bin mirror-transcript &
  transcript_build=$!
  CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
    cargo_build "$scratch/target-sandboxd" "$scratch/sandboxd.jsonl" \
    "${zig[@]}" cargo-zigbuild zigbuild --locked --release -p mirror-sandboxd --bin mirror-sandboxd \
    --target "$sandboxd_target.2.17" &
  sandboxd_build=$!
fi
# Pin the embedded tool for repeatable releases; upgrades must pass RPC and dylib checks.
CARGO_PROFILE_RELEASE_STRIP=true cargo install ripgrep --version 14.1.0 --locked \
  --root "$scratch/ripgrep" --message-format=json-render-diagnostics >"$scratch/ripgrep.jsonl"
MIRROR_BUNDLED_RG="$scratch/ripgrep/bin/rg"
export MIRROR_BUNDLED_RG
if [[ -n "${MIRROR_SIGNED_RG:-}" ]]; then
  [[ "$platform" == macosx_14_0_arm64 && -z "${MIRROR_CODESIGN_IDENTITY:-}" ]]
  cp "$MIRROR_SIGNED_RG" "$MIRROR_BUNDLED_RG"
  chmod 755 "$MIRROR_BUNDLED_RG"
  codesign --verify --strict "$MIRROR_BUNDLED_RG"
fi
check_macos_binary "$MIRROR_BUNDLED_RG"
# The embedded bytes must already contain ripgrep's own signature.
sign_macos_binary "$MIRROR_BUNDLED_RG"
if [[ "$component" == agent ]]; then
  tests_target="$scratch/target-tests"
  (
    cd "$source_dir/.."
    env -u MIRROR_RG_PATH CARGO_TARGET_DIR="$tests_target" \
      CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
      cargo test --locked --release --no-run -p mirror-agent --features bundled-ripgrep \
      --test headless_test --test worker_shutdown_test --test worker_environment_test
  ) &
  tests_build=$!
fi
cargo_build "$CARGO_TARGET_DIR" "$scratch/worker.jsonl" \
  cargo build --locked --release -p mirror-agent --bin mirror-agent --features bundled-ripgrep
if [[ "$component" == agent ]]; then
  native_output="${MIRROR_NATIVE_OUTPUT:-$output}"
  mkdir -p "$native_output"
  native_output="$(cd "$native_output" && pwd)"
  cp "$CARGO_TARGET_DIR/release/mirror-agent" "$native_output/mirror-agent"
  cp "$MIRROR_BUNDLED_RG" "$native_output/rg"
  cp "$source_dir/../mirror-agent/"{LICENSE,NOTICE,RIPGREP-LICENSE-MIT} "$native_output/"
  chmod 755 "$native_output/mirror-agent" "$native_output/rg"
  # Attribute these Cargo bytes, not another build system's dependency graph.
  uv run --no-project --python 3.12 python "$source_dir/native_notices.py" \
    "$native_output/THIRD_PARTY_NOTICES.txt" "$scratch/worker.jsonl" "$scratch/ripgrep.jsonl" \
    --rust-docs "$("${RUSTC:-rustc}" --print sysroot)/share/doc/rust"
  uv run --no-project --python 3.12 --with absl-py python "${MIRROR_NATIVE_ARTIFACTS:?agent mode requires the packaging helper}" \
    --action=record --native="$native_output" --workspace="$source_dir/.." \
    --ripgrep_log="$scratch/ripgrep.jsonl"
  wait "$tests_build"
  (
    cd "$source_dir/.."
    env -u MIRROR_RG_PATH MIRROR_AGENT_BIN="$native_output/mirror-agent" CARGO_TARGET_DIR="$tests_target" \
      CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
      cargo test --locked --release -p mirror-agent --features bundled-ripgrep \
      --test headless_test --test worker_shutdown_test --test worker_environment_test
  )
  # Retain the compiled public-tool test for the extracted archive, without Cargo or Python.
  headless_tests=()
  for candidate in "$tests_target"/release/deps/headless_test-*; do
    if [[ -f "$candidate" && -x "$candidate" ]]; then
      headless_tests+=("$candidate")
    fi
  done
  [[ ${#headless_tests[@]} == 1 ]]
  cp "${headless_tests[0]}" "$native_output/headless_test"
  echo "Runtime-tested standalone agent staged for ABI auditing: $native_output"
  exit
fi
wait "$daemon_build"
daemon="$scratch/target-daemon/release/mirror-daemon"
sign_macos_binary "$daemon"
wait "$transcript_build"
transcript="$scratch/target-transcript/release/mirror-transcript"
check_macos_binary "$transcript"
sign_macos_binary "$transcript"
wait "$sandboxd_build"
sandboxd="$scratch/target-sandboxd/$sandboxd_target/release/mirror-sandboxd"
# Use each build's actual crate sources, including the separately compiled ripgrep.
MIRROR_THIRD_PARTY_NOTICES="$scratch/THIRD_PARTY_NOTICES.txt"
export MIRROR_THIRD_PARTY_NOTICES
uv run --no-project --python 3.12 python "$source_dir/native_notices.py" \
  "$MIRROR_THIRD_PARTY_NOTICES" "$scratch/worker.jsonl" "$scratch/ripgrep.jsonl" \
  --cargo-json "$scratch/sandboxd.jsonl" --cargo-json "$scratch/daemon.jsonl" --cargo-json "$scratch/transcript.jsonl" --zig-runtime "$zig_package" \
  --rust-docs "$("${RUSTC:-rustc}" --print sysroot)/share/doc/rust"
worker="$CARGO_TARGET_DIR/release/mirror-agent"
check_macos_binary "$worker"
sign_macos_binary "$worker"
MIRROR_BUILD_RUST_WORKER="$worker" MIRROR_WHEEL_PLATFORM="$platform" MIRROR_BUILD_SANDBOXD="$sandboxd" MIRROR_BUILD_DAEMON="$daemon" MIRROR_BUILD_TRANSCRIPT="$transcript" \
  uv build --python 3.12 --wheel --out-dir "$scratch/wheels" "$source_dir"
UV_TOOL_DIR="$scratch/tools" UV_TOOL_BIN_DIR="$scratch/bin" \
  uv tool install --python 3.12 "$scratch"/wheels/*.whl
(
  cd "$scratch"
  env -u PYTHONPATH -u MIRROR_RUST_WORKER -u MIRROR_TRANSCRIPT_BIN "$scratch/bin/mirror" --help
)
if [[ "$packaged_tests" == raw ]]; then
  uv pip install --python "$scratch/tools/$distribution/bin/python" \
    freezegun pytest pytest-asyncio pyte
  (
    cd "$scratch"
    env -u PYTHONPATH -u MIRROR_RUST_WORKER -u MIRROR_TRANSCRIPT_BIN MIRROR_TEST_PACKAGED_WHEEL=1 \
      "$scratch/tools/$distribution/bin/python" -m pytest -q \
      "$source_dir/tests/cli/main_test.py::test_installed_worker_tools" \
      "$source_dir/tests/cli/packaged_cli_test.py"
  )
fi
cp "$scratch"/wheels/*.whl "$output/"
native_output="${MIRROR_NATIVE_OUTPUT:-}"
if [[ "$platform" == macosx_14_0_arm64 && -n "${MIRROR_CODESIGN_IDENTITY:-}" ]]; then
  native_output="${native_output:-$output/native}"
fi
if [[ -n "$native_output" ]]; then
  # Notarization and ELF auditing need access to the bytes embedded in the worker.
  mkdir -p "$native_output"
  cp "$worker" "$native_output/mirror-agent"
  cp "$daemon" "$native_output/mirror-daemon"
  cp "$transcript" "$native_output/mirror-transcript"
  cp "$MIRROR_BUNDLED_RG" "$native_output/rg"
  cp "$sandboxd" "$native_output/mirror-sandboxd"
  cp "$MIRROR_THIRD_PARTY_NOTICES" "$native_output/THIRD_PARTY_NOTICES.txt"
  cp "$source_dir/../mirror-agent/"{LICENSE,NOTICE,RIPGREP-LICENSE-MIT} "$native_output/"
fi
echo "Runtime-tested Mirror $version wheel (publication checks still required): $output"
