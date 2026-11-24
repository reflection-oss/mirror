"""Sessions on a remote Mirror server, for `mirror acp --remote`.

Each ACP session is a session the server runs and keeps. The local daemon
holds the connection, as it does for a remote tab in the TUI, so nothing but
the conversation runs on this machine.

The server hears of a new session when the session first connects, which the
ACP server puts off until the first prompt. A client that opens a session only
to read the models it offers leaves nothing behind on the server.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import pathlib
import sys
import uuid
from collections.abc import Mapping, Sequence

import grpc
from mirror.protocol import (
  agent_pb2,
  daemon_pb2,
  mirror_pb2,
)

from mirror.cli import (
  acp,
  daemon_client,
  hosted_workspace,
  local_rpc_client,
  models,
  onboarding,
  remote_rpc_client,
  rpc_client,
)
from mirror.cli import host_tools as _host_tools
from mirror.cli.extensions.inference_controls import server_models


@dataclasses.dataclass(frozen=True)
class Endpoint:
  """A remote Mirror server's AgentRPC address."""

  address: str
  secure: bool = True


def _register(metadata: daemon_client.Client, session_id: str) -> None:
  """Record the session with the daemon as a remote workspace of the same ID."""
  snapshot = metadata.snapshot(
    daemon_pb2.WatchWorkspacesRequest(include_closed=True)
  )
  record = next(
    (record for record in snapshot.workspaces if record.id == session_id), None
  )
  if record is None:
    record = metadata.create(
      daemon_pb2.CreateWorkspaceRequest(
        workspace_id=session_id,
        remote=daemon_pb2.RemoteSource(session_id=session_id),
      )
    )
  if record.kind != daemon_pb2.WORKSPACE_KIND_REMOTE:
    raise RuntimeError(
      f'session {session_id} is already registered as a local workspace'
    )


class _Attachment:
  """A hosted session's connection, which starts the session if it is new."""

  def __init__(
    self,
    daemon: local_rpc_client.Transport,
    endpoint: Endpoint,
    session_id: str,
    directory: pathlib.Path,
    *,
    create: bool,
  ) -> None:
    self._daemon = daemon
    self._endpoint = endpoint
    self._session_id = session_id
    self._directory = directory
    self._create = create
    self._registered = False

  async def connect(
    self,
    host_tools: Sequence[_host_tools.ToolDefinition] = (),
    *,
    attachment: agent_pb2.ClientMessage | None = None,
  ) -> rpc_client.Connection:
    """Register the session with the daemon, start it if new, then attach."""
    if not self._registered:
      metadata = await self._daemon.daemon_client()
      try:
        await asyncio.to_thread(_register, metadata, self._session_id)
        if self._create:
          await remote_rpc_client.start_remote_session(
            self._endpoint.address,
            self._session_id,
            secure=self._endpoint.secure,
            workspace=await hosted_workspace.for_directory(self._directory),
            daemon_address=metadata.local_address,
          )
      finally:
        metadata.close()
      self._registered = True
    return await self._daemon.connect(host_tools, attachment=attachment)

  async def close(self) -> None:
    """Release the attachment; the session stays on the server."""
    await self._daemon.close()


def transport(
  worker_command: Sequence[str], endpoint: Endpoint
) -> acp.TransportFactory:
  """Start a session on the server, or rejoin one it kept.

  A new session checks out the repository the client's directory is a clone
  of, or starts without one when the directory has no usable `origin`.

  Args:
    worker_command: The installed agent bundle, which locates the daemon.
    endpoint: The server that runs the sessions.

  Returns:
    A factory that yields a transport and the session's ID. The transport
    starts the session when it first connects, unless the configuration the
    factory is given sets `create_session` false.
  """

  async def factory(
    config: Mapping[str, object],
    workspace: pathlib.Path,
    conversation_id: str,
    on_message: rpc_client.MessageHandler,
  ) -> tuple[rpc_client.Transport, str]:
    create = bool(config.get('create_session', True))
    session_id = str(uuid.uuid4()) if create else conversation_id
    daemon = local_rpc_client.Transport(
      worker_command,
      dict(config)
      | {'conversation_id': session_id, 'workspace_id': session_id},
      workspace=workspace,
      log_dir=workspace,
      on_message=on_message,
      remote_endpoint=endpoint.address,
      remote_secure=endpoint.secure,
    )
    return (
      _Attachment(daemon, endpoint, session_id, workspace, create=create),
      session_id,
    )

  return factory


async def offered_models(endpoint: Endpoint) -> tuple[models.ModelOption, ...]:
  """The session models the server lists, or none when they cannot be read."""
  try:
    listed = await server_models.list_models(
      endpoint.address, secure=endpoint.secure
    )
  except (OSError, RuntimeError, TimeoutError, grpc.RpcError) as error:
    print(
      f'mirror acp: could not list server models ({type(error).__name__})',
      file=sys.stderr,
    )
    return ()
  return tuple(
    models.ModelOption(model.id, model.label or model.id, api=model.api)
    for model in listed
    if not model.subagent_only
  )


async def session_page(
  endpoint: Endpoint, offset: int, limit: int
) -> tuple[Sequence[mirror_pb2.Session], int]:
  """One page of the caller's sessions on the server, and how many there are."""
  page = await remote_rpc_client.remote_session_page(
    endpoint.address, offset=offset, limit=limit, secure=endpoint.secure
  )
  return page.sessions, page.total


def consent(workspace: pathlib.Path) -> None:
  """Require the first-use notice; a hosted session opens no local folder.

  Raises:
    SystemExit: The notice has not been accepted in an interactive session.
  """
  del workspace
  if not onboarding.is_usage_acknowledged():
    raise SystemExit(
      'mirror: open Mirror interactively to review and accept the usage '
      'notice before using acp'
    )


def run(
  worker_command: Sequence[str],
  worker_config: acp.WorkerConfigFactory,
  endpoint: Endpoint,
  *,
  model: str | None,
  reasoning_effort: str,
) -> int:
  """Serve ACP on stdin/stdout over sessions the server at `endpoint` runs.

  Args:
    worker_command: The installed agent bundle, which locates the daemon.
    worker_config: The launch settings for a conversation ID.
    endpoint: The server that runs the sessions.
    model: The server model new sessions start on; None is the server's default.
    reasoning_effort: The reasoning effort new sessions start with.

  Returns:
    The process exit status.
  """

  def hosted_config(conversation_id: str) -> Mapping[str, object]:
    return dict(worker_config(conversation_id)) | {
      'model': model or acp.SERVER_DEFAULT
    }

  return acp.serve_stdio(
    acp.Server(
      worker_config=hosted_config,
      transport=transport(worker_command, endpoint),
      reasoning_effort=reasoning_effort,
      consent=consent,
      model_source=functools.partial(offered_models, endpoint),
      session_pages=functools.partial(session_page, endpoint),
    )
  )
