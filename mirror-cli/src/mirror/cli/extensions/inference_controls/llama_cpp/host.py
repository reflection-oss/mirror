"""Host facts about llama.cpp that the daemon does not report."""

from __future__ import annotations

import asyncio
import os
import pathlib
import re

from mirror.cli import models

_PRESETS_TEMPLATE = """\
; Per-model llama-server settings, applied the next time the server starts.
;
; One [section] per model, named as in /model (repo:QUANT), or [*] for all models.
; Keys are llama-server flags without the leading dashes (see `llama-server --help`).
; Mirror starts the server with its own host, port and offline flags, which take
; precedence over this file. Remove the leading `;` to enable an example.
;
; [unsloth/Qwen3.5-4B-GGUF:IQ2_M]
; ctx-size = 32768
; spec-type = ngram-mod
;
; [unsloth/Qwen3.6-35B-A3B-MTP-GGUF:Q4_K_M]
; spec-type = draft-mtp
;
; [some/model-GGUF:Q4_K_M]
; spec-draft-model = /path/to/draft-model.gguf
"""

_SERVER_COMMAND = re.compile(r'^(?:\S*/)?llama-server(?:\s|$)')


def ensure_presets_file(path: pathlib.Path) -> None:
  """Create the settings file with commented examples; never touch an existing one."""
  try:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
  except FileExistsError:
    return
  with os.fdopen(descriptor, 'w', encoding='utf-8') as file:
    file.write(_PRESETS_TEMPLATE)


def server_url(query: str) -> str:
  """Turn what the user typed (a port, host:port or URL) into a loopback /v1 URL."""
  query = query.strip()
  if query.isdigit():
    query = f'http://127.0.0.1:{query}'
  elif query and '://' not in query:
    query = f'http://{query}'
  if query and not query.rstrip('/').endswith('/v1'):
    query = f'{query.rstrip("/")}/v1'
  return models.local_base_url(query)


def running_servers(listing: str) -> tuple[str, ...]:
  """Loopback URLs of the llama-server processes in `ps` output the user started.

  Mirror's own router (started by its guard) and any router's model instances are
  skipped: they are not servers to connect to.
  """
  processes: dict[int, tuple[int, str]] = {}
  for line in listing.splitlines():
    parts = line.split(None, 2)
    if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
      processes[int(parts[0])] = (int(parts[1]), parts[2])
  urls: dict[int, str] = {}
  for parent_id, command in processes.values():
    if not _SERVER_COMMAND.match(command):
      continue
    parent = processes.get(parent_id)
    if parent is not None and (
      _SERVER_COMMAND.match(parent[1]) or 'local-models-guard' in parent[1]
    ):
      continue
    arguments = command.replace('=', ' ').split()
    host = _flag_value(arguments, '--host') or '127.0.0.1'
    port = _flag_value(arguments, '--port') or '8080'
    if host in ('127.0.0.1', 'localhost', '0.0.0.0') and port.isdigit():
      if int(port):
        urls[int(port)] = f'http://127.0.0.1:{port}/v1'
  return tuple(urls[port] for port in sorted(urls))


def _flag_value(arguments: list[str], flag: str) -> str | None:
  return (
    arguments[arguments.index(flag) + 1] if flag in arguments[:-1] else None
  )


async def detect_servers() -> tuple[str, ...]:
  """The llama-server processes running on this machine that the user started."""
  try:
    process = await asyncio.create_subprocess_exec(
      'ps',
      '-axww',
      '-o',
      'pid=,ppid=,command=',
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.DEVNULL,
    )
    output, _ = await process.communicate()
  except OSError:
    return ()
  return running_servers(output.decode(errors='replace'))
