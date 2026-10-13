"""Check the installed release channel before importing or starting the CLI."""

from __future__ import annotations

import argparse
import os
import sys

from mirror import updater


def run() -> None:
  """Offer a startup update, then replace this launcher with the chosen CLI."""
  parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
  parser.add_argument('--update', action='store_true')
  parser.add_argument('--no-update', action='store_true')
  options, arguments = parser.parse_known_args()
  exec_mode = 'exec' in arguments
  try:
    launch = updater.prepare(
      check=not options.no_update
      and not exec_mode
      and not any(
        arg in ('--help', '-h', '--version', '--showcase') for arg in arguments
      ),
      force=options.update and not exec_mode,
      interactive=not exec_mode and sys.stdin.isatty() and sys.stderr.isatty(),
    )
  except KeyboardInterrupt:
    raise SystemExit(130) from None
  # No TUI imports, database access, model auth, or workers precede this handoff.
  forwarded = sys.argv[1:] if launch.module == 'mirror.main' else arguments
  # Children exec fresh programs; gRPC's at-fork thread shutdown can deadlock.
  os.execve(
    str(launch.python),
    [str(launch.python), '-m', launch.module, *forwarded],
    os.environ | {'GRPC_ENABLE_FORK_SUPPORT': '0'},
  )


if __name__ == '__main__':
  run()
