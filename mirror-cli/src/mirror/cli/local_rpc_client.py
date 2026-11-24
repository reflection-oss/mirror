"""Local daemon discovery and attachment; workers belong to the daemon."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import uuid
from collections.abc import (
  AsyncIterator,
  Awaitable,
  Callable,
  Mapping,
  Sequence,
)
from typing import Protocol, cast

import grpc.aio
from google.protobuf import json_format
from mirror.protocol import (
  agent_pb2,
  daemon_pb2,
  daemon_pb2_grpc,
)

from mirror import distribution
from mirror import config
from mirror.cli import approval_modes
from mirror.cli import (
  daemon_client,
  models,
  mcp_configuration,
  rpc_client,
  safety,
  sandboxes,
  shell,
  system_prompt,
)
from mirror.cli.extensions.connections import transport as mcp_transport
from mirror.cli import host_tools as _host_tools


class DaemonConnectionError(RuntimeError):
  """Daemon discovery or automatic replacement failed."""


class _DaemonStub(Protocol):
  def ConfigureDaemon(  # noqa: N802, generated RPC name
    self, request: daemon_pb2.ConfigureDaemonRequest, *, timeout: float
  ) -> Awaitable[daemon_pb2.ConfigureDaemonResponse]: ...

  def AttachWorkspace(  # noqa: N802, generated RPC name
    self, request: daemon_pb2.AttachWorkspaceRequest, *, timeout: float
  ) -> Awaitable[daemon_pb2.AttachWorkspaceResponse]: ...

  def WatchWorkspaces(  # noqa: N802, generated RPC name
    self, request: daemon_pb2.WatchWorkspacesRequest
  ) -> AsyncIterator[daemon_pb2.WatchWorkspacesResponse]: ...


def _configuration(
  values: Mapping[str, object],
  inference: agent_pb2.InferenceConfig,
  tools: Sequence[_host_tools.ToolDefinition],
) -> daemon_pb2.AgentConfiguration:
  fields = (
    'system_prompt',
    'prompt_injection_policy',
    'compaction_model',
    'compaction_model_api',
    'compaction_model_base_url',
    'max_iterations',
    'compaction_threshold',
    'context_window',
    'max_tokens',
    'memory_dir',
    'include_environment',
    'root_search_guard',
    'checkpoints_enabled',
    'additional_skills',
    'web_provider',
  )
  supplied = {
    name: values[name] for name in fields if values.get(name) is not None
  }
  config = daemon_pb2.AgentConfiguration()
  json_format.ParseDict(supplied, config)
  config.inference.CopyFrom(inference)
  config.web_credentials.CopyFrom(
    daemon_pb2.WebCredentials(
      you_api_key=os.environ.get('YOU_API_KEY'),
      tavily_api_key=os.environ.get('TAVILY_API_KEY'),
      firecrawl_api_key=os.environ.get('FIRECRAWL_API_KEY'),
    )
  )
  config.approval_mode = approval_modes.to_protocol(
    values.get('approval_mode') or 'auto'
  )
  if values.get('tools') is not None:
    json_format.ParseDict({'names': values['tools']}, config.tools)
  config.mcp_connections.CopyFrom(
    mcp_configuration.runtime_connections(
      cast(
        Sequence[mcp_configuration.McpConnection],
        values.get('mcp_connections', ()),
      )
    )
  )
  for tool in tools:
    config.host_tools.add(
      name=tool.name,
      description=tool.description,
      parameters_json=json.dumps(tool.parameters, sort_keys=True),
      needs_approval=tool.needs_approval,
      timeout_seconds=tool.timeout_seconds,
    )
  api = inference.api
  base_env = (
    'MIRROR_CODEX_BASE_URL'
    if api == 'openai-codex-responses'
    else models.base_env(api)
  )
  if base_env and not config.inference.base_url:
    config.inference.base_url = os.environ.get(base_env, '')
  environment_key = (
    os.environ.get(models.key_env(api))
    if api != 'openai-codex-responses'
    else None
  )
  if key := values.get('api_key') or environment_key:
    if not isinstance(key, str):
      raise ValueError('provider API key must be a string')
    config.api_key = key
  if config.HasField('compaction_model'):
    compaction_api = config.compaction_model_api or api
    compaction_key = values.get('compaction_api_key')
    if not compaction_key and compaction_api != 'openai-codex-responses':
      compaction_key = os.environ.get(models.key_env(compaction_api))
    if compaction_key:
      if not isinstance(compaction_key, str):
        raise ValueError('compaction API key must be a string')
      config.compaction_api_key = compaction_key
    compaction_base_env = (
      'MIRROR_CODEX_BASE_URL'
      if compaction_api == 'openai-codex-responses'
      else models.base_env(compaction_api)
    )
    if compaction_base_env and not config.compaction_model_base_url:
      if compaction_url := os.environ.get(compaction_base_env):
        config.compaction_model_base_url = compaction_url
  if path := values.get('auth_path') or os.environ.get(
    'MIRROR_CODEX_AUTH_PATH'
  ):
    if not isinstance(path, str):
      raise ValueError('provider auth path must be a string')
    config.auth_path = path
  return config


async def create_workspace(
  worker_command: Sequence[str],
  worker_config: Mapping[str, object],
  workspace: pathlib.Path,
) -> daemon_pb2.Workspace:
  """Create a workspace binding before constructing a headless client."""
  metadata = await Transport(
    worker_command,
    worker_config,
    workspace=workspace,
    log_dir=workspace,
  ).daemon_client()
  try:
    return await asyncio.to_thread(
      metadata.create,
      daemon_pb2.CreateWorkspaceRequest(
        workspace_id=str(uuid.uuid4()), directory=str(workspace.resolve())
      ),
    )
  finally:
    metadata.close()


class Transport:
  """Attach to a daemon-owned workspace without a direct-worker fallback."""

  def __init__(
    self,
    worker_command: Sequence[str],
    worker_config: Mapping[str, object],
    *,
    workspace: pathlib.Path,
    log_dir: pathlib.Path,
    on_message: rpc_client.MessageHandler | None = None,
    on_warning: Callable[[str], None] | None = None,
    remote_endpoint: str | None = None,
    remote_secure: bool = True,
  ) -> None:
    del log_dir  # Raw subprocess diagnostics never become client logs.
    if len(worker_command) != 1:
      raise ValueError('the daemon requires one native installation bundle')
    worker = pathlib.Path(worker_command[0]).resolve()
    self._on_message = on_message
    self._on_warning = on_warning
    self._remote_endpoint = (
      f'{"https" if remote_secure else "http"}://{remote_endpoint}'
      if remote_endpoint is not None
      else None
    )
    self._values = worker_config
    self._workspace = (
      workspace if worker_config.get('daemon_address') else workspace.resolve()
    )
    self._client_id = uuid.uuid4().hex
    self._database = pathlib.Path(
      str(
        worker_config.get('database')
        or os.environ.get('MIRROR_STATE_DB')
        or pathlib.Path.home() / '.mirror' / 'state.db'
      )
    ).resolve()
    profile = hashlib.sha256(str(self._database).encode()).hexdigest()[:16]
    self._state_root = pathlib.Path(
      os.environ.get('MIRROR_DAEMON_STATE_ROOT')
      or pathlib.Path.home() / '.mirror' / 'run' / profile
    ).resolve()
    self._daemon = worker.with_name('mirror-daemon')
    self._channel: grpc.aio.Channel | None = None
    self._attachment_id: str | None = None

  async def daemon_client(self) -> daemon_client.Client:
    """Discover metadata without opening a worker."""
    return daemon_client.Client(
      await self._ensure(),
      daemon_id=str(self._values['daemon_instance_id'])
      if self._values.get('daemon_instance_id')
      else None,
      remote_host=str(self._values['daemon_host'])
      if self._values.get('daemon_host')
      else None,
    )

  async def connect(
    self,
    host_tools: Sequence[_host_tools.ToolDefinition] = (),
    *,
    attachment: agent_pb2.ClientMessage | None = None,
  ) -> rpc_client.Connection:
    workspace_id = self._values.get('workspace_id')
    if not isinstance(workspace_id, str) or not workspace_id:
      raise ValueError('Create or select a daemon workspace before attaching')
    # SSH attachments never resolve laptop provider secrets or host tool routes.
    configuration = None
    if not self._values.get('daemon_address'):
      inference = (
        attachment.set_inference_config
        if attachment is not None
        and attachment.HasField('set_inference_config')
        else rpc_client.WorkerConfiguration(
          self._values, cwd=self._workspace
        ).inference_config(
          str(self._values.get('model', '')),
          'max',
          remote=self._remote_endpoint is not None,
        )
      )
      configuration = (
        daemon_pb2.AgentConfiguration(
          inference=inference,
          approval_mode=approval_modes.to_protocol(
            self._values.get('approval_mode') or 'auto'
          ),
        )
        if self._remote_endpoint is not None
        else _configuration(self._values, inference, host_tools)
      )
    return await self._attach(await self._ensure(), workspace_id, configuration)

  async def _attach(
    self,
    address: str,
    workspace_id: str,
    configuration: daemon_pb2.AgentConfiguration | None,
    *,
    existing_worker_id: str | None = None,
  ) -> rpc_client.Connection:
    self._channel = grpc.aio.insecure_channel(
      address,
      options=rpc_client.GRPC_CHANNEL_OPTIONS + daemon_client.CHANNEL_OPTIONS,
    )
    try:
      stub = cast(_DaemonStub, daemon_pb2_grpc.DaemonRPCStub(self._channel))
      opened = await stub.AttachWorkspace(
        daemon_pb2.AttachWorkspaceRequest(
          workspace_id=workspace_id,
          client_id=self._client_id,
          configuration=configuration,
          use_daemon_configuration=bool(self._values.get('daemon_address')),
          require_new_worker=existing_worker_id is None
          and bool(self._values.get('require_new_worker', False)),
          remote_endpoint=self._remote_endpoint,
          expected_session_id=str(self._values['conversation_id']),
          expected_daemon_id=str(self._values['daemon_instance_id'])
          if self._values.get('daemon_instance_id')
          else None,
          existing_worker_id=existing_worker_id,
        ),
        timeout=10 if existing_worker_id is not None else 960,
      )
      if self._on_message is not None and existing_worker_id is None:
        for message in opened.startup_messages:
          await rpc_client.deliver_message(self._on_message, message)
      if opened.HasField('startup_error'):
        raise RuntimeError(opened.startup_error)
      self._attachment_id = opened.attachment_id
      if (
        opened.workspace.id != workspace_id
        or opened.workspace.session_id != self._values['conversation_id']
        or (
          existing_worker_id is not None
          and opened.worker_instance_id != existing_worker_id
        )
      ):
        raise RuntimeError('Daemon workspace conversation binding changed')
      worker_id = opened.worker_instance_id
      # Reattachment needs live tool definitions, never provider credentials.
      host_configuration = (
        daemon_pb2.AgentConfiguration(host_tools=configuration.host_tools)
        if configuration is not None
        else None
      )

      async def reconnect(manual: bool) -> rpc_client.Connection:
        try:
          return await self._attach(
            address,
            workspace_id,
            host_configuration,
            existing_worker_id=worker_id,
          )
        except grpc.aio.AioRpcError as error:
          if not manual or error.code() != grpc.StatusCode.UNAVAILABLE:
            raise
        return await self._attach(
          await self._ensure(),
          workspace_id,
          host_configuration,
          existing_worker_id=worker_id,
        )

      return rpc_client.Connection(
        self._channel,
        agent_pb2.ClientMessage(
          session_id=opened.workspace.session_id, message_id=uuid.uuid4().hex
        ),
        initial_settings=opened.workspace,
        settings=lambda: _settings(stub, workspace_id),
        daemon_attachment_id=opened.attachment_id,
        execution_host=opened.execution_host,
        reconnect=reconnect if worker_id else None,
      )
    except BaseException:
      await self.close()
      raise

  def workspace_connection(self) -> tuple[grpc.aio.Channel, str]:
    """Process authority is the live attachment, never a saved client/session ID."""
    if self._channel is None or self._attachment_id is None:
      raise shell.ProcessError('Workspace is not attached')
    return self._channel, self._attachment_id

  async def manage(self, action: str) -> str:
    """Start safely, explicitly restart, or observe/stop without starting."""
    if self._values.get('daemon_address'):
      raise DaemonConnectionError('Manage the SSH connection through /ssh.')
    if action == 'address':
      announcement = await self._invoke('connect-only')
      address = _daemon_address(announcement)
      try:
        await self._configure_daemon(address)
      except grpc.aio.AioRpcError as error:
        raise DaemonConnectionError(
          f'Could not configure daemon defaults: {error.details()}'
        ) from error
      return announcement
    if action == 'restart':
      await self._ensure(replace=True)
      return 'Mirror daemon restarted; interrupted local turns remain paused.'
    if action == 'start':
      await self._ensure()
      return await self._invoke('status')
    if action not in ('status', 'stop'):
      raise ValueError(f'Unknown daemon command: {action}')
    return await self._invoke(action)

  async def _configure_daemon(self, address: str) -> None:
    """Register this host's defaults without exposing them to an SSH frontend."""
    values = dict(self._values) | {
      'system_prompt': system_prompt.base_prompt(),
      'prompt_injection_policy': safety.PROMPT_INJECTION_POLICY,
      'include_environment': True,
      'mcp_connections': [
        connection for connection in mcp_transport.daemon_connections()
      ],
    }
    api = str(values.get('model_api') or values.get('api') or 'reflection')
    default = models.ModelOption(
      str(values.get('model') or models.default_model(api)),
      '',
      api=api,
      base_url=str(
        values.get('model_base_url') or values.get('base_url') or ''
      ),
    )
    configurations = [
      _configuration(
        values,
        agent_pb2.InferenceConfig(
          model=option.model_id,
          api=option.api or api,
          base_url=option.base_url,
          reasoning_effort=models.inherited_reasoning_effort(
            option.api or api, str(values.get('reasoning_effort') or 'max')
          ),
        ),
        (),
      )
      for option in (default, *models.load_models())
    ]
    sandbox = sandboxes.Sandbox.from_settings(
      config.Config(config.default_path()).read()
    )
    async with grpc.aio.insecure_channel(
      address,
      options=rpc_client.GRPC_CHANNEL_OPTIONS + daemon_client.CHANNEL_OPTIONS,
    ) as channel:
      stub = cast(_DaemonStub, daemon_pb2_grpc.DaemonRPCStub(channel))
      await stub.ConfigureDaemon(
        daemon_pb2.ConfigureDaemonRequest(
          directory=str(self._workspace),
          configurations=configurations,
          remote_endpoint=self._remote_endpoint,
          sandbox=daemon_pb2.SandboxSource(
            image=sandbox.image,
            egress=sandbox.egress,
            cpus=sandbox.cpus,
            memory_mib=sandbox.memory_mib,
          ),
        ),
        timeout=10,
      )

  async def _ensure(self, *, replace: bool = False) -> str:
    if address := self._values.get('daemon_address'):
      if not isinstance(address, str) or not address.startswith('unix:/'):
        raise DaemonConnectionError(
          'SSH requires a private forwarded Unix socket'
        )
      return address
    announcement = await self._invoke(
      'connect-only'
      if self._values.get('daemon_connect_only') and not replace
      else 'ensure',
      replace=replace,
    )
    return _daemon_address(announcement)

  async def _invoke(self, action: str, *, replace: bool = False) -> str:
    if not self._daemon.is_file():
      raise DaemonConnectionError(f'Mirror daemon is missing: {self._daemon}')
    process = await asyncio.create_subprocess_exec(
      str(self._daemon),
      f'--{action}',
      *(['--replace'] if replace else []),
      '--application-version',
      distribution.installed().version,
      '--state-root',
      str(self._state_root),
      '--database',
      str(self._database),
      '--worker-home',
      str(pathlib.Path.home()),
      '--worker-path',
      os.environ.get('PATH', '/usr/bin:/bin'),
      *(
        ['--llama-cpp-executable', binary]
        if (binary := os.environ.get('MIRROR_LLAMA_CPP_EXECUTABLE'))
        else []
      ),
      env={'TMPDIR': tempfile.gettempdir()},
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
    )
    try:
      stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=35)
      if process.returncode != 0:
        recovery = (
          '\nRun `mirror daemon restart` to replace the daemon and interrupt '
          'active workers; saved history is preserved.'
          if action == 'ensure' and not replace
          else ''
        )
        raise DaemonConnectionError(
          f'Could not connect to Mirror daemon: {stderr.decode().strip()}{recovery}'
        )
      for line in stderr.decode(errors='replace').splitlines():
        if line.startswith('MIRROR_DAEMON_WARNING='):
          warning = line.removeprefix('MIRROR_DAEMON_WARNING=')
          if self._on_warning is not None:
            self._on_warning(warning)
          else:
            print(warning, file=sys.stderr)
      return stdout.decode().strip()
    finally:
      if process.returncode is None:
        process.kill()
        await process.wait()

  async def close_workspace(self) -> None:
    """Explicitly stop the bound workspace; unlike detach, this affects all clients."""
    workspace_id = self._values.get('workspace_id')
    if not isinstance(workspace_id, str) or not workspace_id:
      raise ValueError('No workspace is bound')
    metadata = await self.daemon_client()
    try:
      await asyncio.to_thread(metadata.close_workspace, workspace_id)
    finally:
      metadata.close()

  async def close(self) -> None:
    """Release only this attachment; other clients retain their worker."""
    channel, self._channel = self._channel, None
    self._attachment_id = None
    if channel is not None:
      await channel.close(grace=None)


def _daemon_address(announcement: str) -> str:
  if (
    not announcement.startswith('MIRROR_DAEMON_ADDRESS=unix:/')
    or '\n' in announcement
  ):
    raise DaemonConnectionError('Daemon reported an unexpected endpoint')
  return announcement.removeprefix('MIRROR_DAEMON_ADDRESS=')


async def _settings(
  stub: _DaemonStub, workspace_id: str
) -> AsyncIterator[daemon_pb2.Workspace]:
  async for snapshot in stub.WatchWorkspaces(
    daemon_pb2.WatchWorkspacesRequest()
  ):
    for workspace in snapshot.workspaces:
      if workspace.id == workspace_id and workspace.HasField('inference'):
        yield workspace
