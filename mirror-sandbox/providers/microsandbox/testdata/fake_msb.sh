#!/bin/sh
# Stands in for microsandbox's `msb` in tests. State lives beside this script:
# `<name>.json` is a sandbox's `msb inspect` document with `__STATUS__` in
# place of the status that `<name>.status` holds, `template.json` becomes each
# new sandbox's document, `version` overrides the reported version,
# `create_fails` makes `create` fail after making the sandbox, and
# `create_races` makes another create take the name first. Every invocation is
# appended to `calls`.
set -u
state=$(dirname "$0")
printf '%s\n' "$*" >>"$state/calls"

# Like msb, `--quiet` also silences `rm`'s and `start`'s missing-sandbox error.
missing() {
  case " $quiet_errors " in
    *" --quiet "*) ;;
    *) echo "error: sandbox not found: $1" >&2 ;;
  esac
  exit 1
}

last() {
  for argument; do :; done
  printf '%s' "$argument"
}

command=$1
shift
quiet_errors=
case "$command" in
  rm | start) quiet_errors="$*" ;;
esac
case "$command" in
  --version)
    if [ -f "$state/version" ]; then cat "$state/version"; else echo "msb 0.7.7"; fi
    ;;
  create)
    name=
    while [ "$#" -gt 0 ]; do
      if [ "$1" = --name ]; then name=$2; fi
      shift
    done
    if [ -f "$state/$name.json" ]; then
      echo "error: sandbox already exists: sandbox '$name' already exists" >&2
      exit 1
    fi
    # Like another create for the same request taking the name first.
    if [ -f "$state/create_races" ]; then
      cp "$state/template.json" "$state/$name.json"
      echo Running >"$state/$name.status"
      echo "error: sandbox already exists: sandbox '$name' already exists" >&2
      exit 1
    fi
    cp "$state/template.json" "$state/$name.json"
    echo Running >"$state/$name.status"
    # Like a create that times out after making the sandbox.
    if [ -f "$state/create_fails" ]; then
      echo "error: timed out waiting for the sandbox" >&2
      exit 1
    fi
    ;;
  inspect)
    name=$(last "$@")
    [ -f "$state/$name.json" ] || missing "$name"
    sed "s/__STATUS__/$(cat "$state/$name.status")/" "$state/$name.json"
    ;;
  start | resume)
    name=$(last "$@")
    [ -f "$state/$name.json" ] || missing "$name"
    echo Running >"$state/$name.status"
    ;;
  stop)
    name=$(last "$@")
    [ -f "$state/$name.json" ] || missing "$name"
    echo Stopped >"$state/$name.status"
    ;;
  rm)
    name=$(last "$@")
    [ -f "$state/$name.json" ] || missing "$name"
    rm -f "$state/$name.json" "$state/$name.status"
    ;;
  exec)
    workdir=.
    name=
    while [ "$#" -gt 0 ] && [ "$1" != -- ]; do
      case "$1" in
        --workdir) workdir=$2 && shift ;;
        --env | --timeout) shift ;;
        -*) ;;
        *) name=$1 ;;
      esac
      shift
    done
    shift
    [ -f "$state/$name.json" ] || missing "$name"
    # `sandboxd serve` echoes its input, so a test can see the stream is open.
    if [ "$1" = /usr/local/bin/mirror-sandboxd ]; then exec cat; fi
    cd "$workdir" && exec "$@"
    ;;
  *)
    echo "the fake msb does not support $command" >&2
    exit 2
    ;;
esac
