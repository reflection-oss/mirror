"""Check the installed release channel before importing or starting the CLI."""

from __future__ import annotations

import argparse
import os
import sys

from mirror import updater


def run() -> None:
  """Update without a session, or hand off to the selected CLI."""
  update_only = sys.argv[1:2] == ['update']
  if update_only:
    argparse.ArgumentParser(
      prog='mirror update',
      description='Check for an update, ask before installing, then exit. Requires a terminal.',
      allow_abbrev=False,
    ).parse_args(sys.argv[2:])
  parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
  parser.add_argument('--no-update', action='store_true')
  options, arguments = parser.parse_known_args()
  exec_mode = 'exec' in arguments or 'daemon' in arguments
  try:
    launch = updater.prepare(
      check=not options.no_update
      and not exec_mode
      and not any(
        arg in ('--help', '-h', '--version', '--showcase') for arg in arguments
      ),
      interactive=not exec_mode and sys.stdin.isatty() and sys.stderr.isatty(),
      update_only=update_only,
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
