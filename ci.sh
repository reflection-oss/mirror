#!/usr/bin/env bash
# Source-only Python checks; Rust checks use Cargo separately.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$root/mirror-cli/ci.sh" "$@"
