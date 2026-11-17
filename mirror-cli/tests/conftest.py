"""Fixtures shared by the mirror test suite."""

from __future__ import annotations

import concurrent.futures
import contextlib
import queue
import uuid
from collections.abc import Iterator
from typing import cast

import freezegun
import grpc
import grpc.aio
import pytest
from freezegun import api as freezegun_api
from freezegun import config as freezegun_config
from google.protobuf import descriptor
from mirror.protocol import (
  daemon_pb2,
  daemon_pb2_grpc,
)

from mirror.cli import daemon_client


@pytest.fixture
def frozen_ui_time() -> Iterator[freezegun_api.FrozenDateTimeFactory]:
  """Freeze animation frames while thread and asyncio scheduling stay live."""
  ignored = freezegun_config.settings.default_ignore_list
  # Freezegun otherwise uses real time anywhere prompt_toolkit is on the stack.
  freezegun.configure(
    default_ignore_list=[name for name in ignored if name != 'prompt_toolkit']
  )
  try:
    with freezegun.freeze_time('2000-01-01', real_asyncio=True) as clock:
      assert isinstance(clock, freezegun_api.FrozenDateTimeFactory)
      yield clock
  finally:
    freezegun.configure(default_ignore_list=ignored)


@pytest.fixture(autouse=True)
def disable_extensions(monkeypatch: pytest.MonkeyPatch) -> None:
  """Do not execute external extensions in public harness tests."""
  monkeypatch.setenv('MIRROR_EXTENSIONS_ENABLED', 'false')


@pytest.fixture(autouse=True)
def isolate_models_config(
  tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Keep the model config, and the last-used model beside it, out of $HOME."""
  directory = tmp_path_factory.mktemp('models-config')
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', str(directory / 'models.json'))


@pytest.fixture(autouse=True)
def isolate_state_database(
  tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Keep the daemon's agent storage out of $HOME."""
  directory = tmp_path_factory.mktemp('state')
  monkeypatch.setenv('MIRROR_STATE_DB', str(directory / 'state.db'))


class _WorkspaceService(daemon_pb2_grpc.DaemonRPCServicer):
  """Metadata-only DaemonRPC fixture; these tests never attach an agent."""

  def __init__(
    self,
    snapshot: daemon_pb2.WatchWorkspacesResponse,
    history: daemon_pb2.ListConversationsResponse,
    conversation_responses: queue.Queue[
      concurrent.futures.Future[daemon_pb2.ListConversationsResponse]
    ],
    prompts: daemon_pb2.GetPromptHistoryResponse,
    creations: queue.Queue[
      tuple[
        daemon_pb2.CreateWorkspaceRequest,
        concurrent.futures.Future[daemon_pb2.Workspace],
      ]
    ],
    deletions: queue.Queue[
      tuple[daemon_pb2.DeleteWorkspaceRequest, concurrent.futures.Future[None]]
    ],
    deleted: queue.Queue[str],
    queries: queue.Queue[daemon_pb2.WatchWorkspacesRequest],
    responses: queue.Queue[
      concurrent.futures.Future[daemon_pb2.WatchWorkspacesResponse]
    ],
  ) -> None:
    self.snapshot = snapshot
    self.history = history
    self.conversation_responses = conversation_responses
    self.prompts = prompts
    self.creations = creations
    self.deletions = deletions
    self.deleted = deleted
    self.queries = queries
    self.responses = responses

  def ListConversations(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.ListConversationsRequest,
    context: grpc.ServicerContext,
  ) -> daemon_pb2.ListConversationsResponse:
    del context
    with contextlib.suppress(queue.Empty):
      return self.conversation_responses.get_nowait().result(timeout=5)
    items = [
      item
      for item in self.history.conversations
      if not request.session_ids or item.id in request.session_ids
    ]
    if request.recent_first:
      items.sort(
        key=lambda item: (
          item.updated_at.seconds,
          item.updated_at.nanos,
          item.id,
        ),
        reverse=True,
      )
      if request.HasField('before_updated_at'):
        cursor = (
          request.before_updated_at.seconds,
          request.before_updated_at.nanos,
          request.after_id,
        )
        items = [
          item
          for item in items
          if (item.updated_at.seconds, item.updated_at.nanos, item.id) < cursor
        ]
    else:
      items = sorted(
        (item for item in items if item.id > request.after_id),
        key=lambda item: item.id,
      )
    page = items[: min(request.limit or 100, 2)]
    more = len(items) > len(page)
    return daemon_pb2.ListConversationsResponse(
      conversations=page,
      next_after_id=page[-1].id if more else '',
      next_before_updated_at=page[-1].updated_at
      if more and request.recent_first
      else None,
    )

  def GetPromptHistory(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.GetPromptHistoryRequest,
    context: grpc.ServicerContext,
  ) -> daemon_pb2.GetPromptHistoryResponse:
    del request, context
    return self.prompts

  def CreateWorkspace(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.CreateWorkspaceRequest,
    context: grpc.ServicerContext,
  ) -> daemon_pb2.CreateWorkspaceResponse:
    if request.WhichOneof('source') == 'worktree_from':
      result: concurrent.futures.Future[daemon_pb2.Workspace] = (
        concurrent.futures.Future()
      )
      self.creations.put((request, result))
      try:
        record = result.result(timeout=5)
      except ValueError as error:
        context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
      return daemon_pb2.CreateWorkspaceResponse(workspace=record)
    remote = request.WhichOneof('source') == 'remote'
    assert remote or request.WhichOneof('source') == 'directory'
    if remote:
      assert request.remote.session_id == request.workspace_id
      assert not request.remote.endpoint
      assert not request.remote.HasField('workspace')
    record = self.snapshot.workspaces.add(
      id=request.workspace_id,
      name=request.name,
      directory=request.directory,
      preferences=request.preferences,
      session_id=request.remote.session_id if remote else str(uuid.uuid4()),
      kind=daemon_pb2.WORKSPACE_KIND_REMOTE
      if remote
      else daemon_pb2.WORKSPACE_KIND_DIRECTORY,
      state=daemon_pb2.WORKSPACE_STATE_AVAILABLE,
    )
    return daemon_pb2.CreateWorkspaceResponse(workspace=record)

  def UpdateWorkspace(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.UpdateWorkspaceRequest,
    context: grpc.ServicerContext,
  ) -> daemon_pb2.UpdateWorkspaceResponse:
    if (
      not request.update_mask.paths
      or not request.update_mask.IsValidForDescriptor(
        # FieldMask also accepts the interchangeable upb descriptor implementation.
        cast(descriptor.Descriptor, daemon_pb2.Workspace.DESCRIPTOR)
      )
    ):
      context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'invalid update mask')
    for path in request.update_mask.paths:
      if (
        path != 'name'
        and path != 'preferences'
        and not path.startswith('preferences.')
      ):
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'unsupported field')
    update = daemon_pb2.Workspace(
      name=request.name,
      preferences=request.preferences
      if request.HasField('preferences')
      else None,
    )
    for record in self.snapshot.workspaces:
      if record.id == request.workspace_id:
        request.update_mask.MergeMessage(
          update,
          record,
          replace_message_field=True,
          replace_repeated_field=True,
        )
        return daemon_pb2.UpdateWorkspaceResponse(workspace=record)
    context.abort(grpc.StatusCode.NOT_FOUND, 'workspace not found')

  def SwitchWorkspaceSession(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.SwitchWorkspaceSessionRequest,
    context: grpc.ServicerContext,
  ) -> daemon_pb2.SwitchWorkspaceSessionResponse:
    record = next(
      (
        record
        for record in self.snapshot.workspaces
        if record.id == request.workspace_id
      ),
      None,
    )
    if record is None:
      context.abort(grpc.StatusCode.NOT_FOUND, 'workspace not found')
    if record.session_id != request.expected_session_id:
      context.abort(grpc.StatusCode.ABORTED, 'workspace session changed')
    if record.HasField('inference'):
      context.abort(
        grpc.StatusCode.FAILED_PRECONDITION, 'close workspace first'
      )
    target = (
      request.session_id
      if request.HasField('session_id')
      else str(uuid.uuid4())
    )
    if request.HasField('session_id') and target not in (
      record.session_id,
      *record.previous_session_ids,
    ):
      context.abort(
        grpc.StatusCode.NOT_FOUND, 'conversation does not belong to workspace'
      )
    if target != record.session_id:
      previous = set(record.previous_session_ids) | {record.session_id}
      previous.discard(target)
      record.previous_session_ids[:] = sorted(previous)
      record.session_id = target
    record.ClearField('inference')
    record.ClearField('approval_mode')
    record.state = daemon_pb2.WORKSPACE_STATE_AVAILABLE
    return daemon_pb2.SwitchWorkspaceSessionResponse(workspace=record)

  def OpenWorkspace(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.OpenWorkspaceRequest,
    context: grpc.ServicerContext,
  ) -> daemon_pb2.OpenWorkspaceResponse:
    for record in self.snapshot.workspaces:
      if record.id == request.workspace_id:
        record.state = daemon_pb2.WORKSPACE_STATE_AVAILABLE
        return daemon_pb2.OpenWorkspaceResponse(workspace=record)
    context.abort(grpc.StatusCode.NOT_FOUND, 'workspace not found')

  def CloseWorkspace(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.CloseWorkspaceRequest,
    context: grpc.ServicerContext,
  ) -> daemon_pb2.CloseWorkspaceResponse:
    for record in self.snapshot.workspaces:
      if record.id == request.workspace_id:
        record.state = daemon_pb2.WORKSPACE_STATE_CLOSED
        record.ClearField('inference')
        record.ClearField('approval_mode')
        return daemon_pb2.CloseWorkspaceResponse()
    context.abort(grpc.StatusCode.NOT_FOUND, 'workspace not found')

  def DeleteWorkspace(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.DeleteWorkspaceRequest,
    context: grpc.ServicerContext,
  ) -> daemon_pb2.DeleteWorkspaceResponse:
    for record in self.snapshot.workspaces:
      if record.id == request.workspace_id:
        if request.delete_files:
          result: concurrent.futures.Future[None] = concurrent.futures.Future()
          self.deletions.put((request, result))
          try:
            result.result(timeout=5)
          except ValueError as error:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
        self.snapshot.workspaces.remove(record)
        self.deleted.put(request.workspace_id)
        return daemon_pb2.DeleteWorkspaceResponse()
    context.abort(grpc.StatusCode.NOT_FOUND, 'workspace not found')

  def WatchWorkspaces(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.WatchWorkspacesRequest,
    context: grpc.ServicerContext,
  ) -> Iterator[daemon_pb2.WatchWorkspacesResponse]:
    self.queries.put(request)
    try:
      response = self.responses.get_nowait()
    except queue.Empty:
      yield self.snapshot
      return
    try:
      snapshot = response.result(timeout=5)
    except grpc.aio.AioRpcError as error:
      context.abort(error.code(), error.details())
    except RuntimeError as error:
      context.abort(grpc.StatusCode.UNAVAILABLE, str(error))
    yield snapshot


@pytest.fixture
def conversation_history() -> daemon_pb2.ListConversationsResponse:
  return daemon_pb2.ListConversationsResponse()


@pytest.fixture
def conversation_responses() -> queue.Queue[
  concurrent.futures.Future[daemon_pb2.ListConversationsResponse]
]:
  return queue.Queue()


@pytest.fixture
def prompt_history() -> daemon_pb2.GetPromptHistoryResponse:
  return daemon_pb2.GetPromptHistoryResponse()


@pytest.fixture
def workspace_snapshot() -> daemon_pb2.WatchWorkspacesResponse:
  return daemon_pb2.WatchWorkspacesResponse()


@pytest.fixture
def workspace_creations() -> queue.Queue[
  tuple[
    daemon_pb2.CreateWorkspaceRequest,
    concurrent.futures.Future[daemon_pb2.Workspace],
  ]
]:
  return queue.Queue()


@pytest.fixture
def workspace_deletions() -> queue.Queue[
  tuple[daemon_pb2.DeleteWorkspaceRequest, concurrent.futures.Future[None]]
]:
  return queue.Queue()


@pytest.fixture
def workspace_deleted() -> queue.Queue[str]:
  return queue.Queue()


@pytest.fixture
def workspace_queries() -> queue.Queue[daemon_pb2.WatchWorkspacesRequest]:
  return queue.Queue()


@pytest.fixture
def workspace_responses() -> queue.Queue[
  concurrent.futures.Future[daemon_pb2.WatchWorkspacesResponse]
]:
  return queue.Queue()


@pytest.fixture
def workspace_address(
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
  conversation_history: daemon_pb2.ListConversationsResponse,
  conversation_responses: queue.Queue[
    concurrent.futures.Future[daemon_pb2.ListConversationsResponse]
  ],
  prompt_history: daemon_pb2.GetPromptHistoryResponse,
  workspace_creations: queue.Queue[
    tuple[
      daemon_pb2.CreateWorkspaceRequest,
      concurrent.futures.Future[daemon_pb2.Workspace],
    ]
  ],
  workspace_deletions: queue.Queue[
    tuple[daemon_pb2.DeleteWorkspaceRequest, concurrent.futures.Future[None]]
  ],
  workspace_deleted: queue.Queue[str],
  workspace_queries: queue.Queue[daemon_pb2.WatchWorkspacesRequest],
  workspace_responses: queue.Queue[
    concurrent.futures.Future[daemon_pb2.WatchWorkspacesResponse]
  ],
) -> Iterator[str]:
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    server = grpc.server(executor)
    daemon_pb2_grpc.add_DaemonRPCServicer_to_server(  # pyright: ignore[reportUnknownMemberType] - generated registration has no stub.
      _WorkspaceService(
        workspace_snapshot,
        conversation_history,
        conversation_responses,
        prompt_history,
        workspace_creations,
        workspace_deletions,
        workspace_deleted,
        workspace_queries,
        workspace_responses,
      ),
      server,
    )
    port = server.add_insecure_port('127.0.0.1:0')
    server.start()
    try:
      yield f'127.0.0.1:{port}'
    finally:
      server.stop(0).wait(timeout=5)


@pytest.fixture
def workspace_client(workspace_address: str) -> Iterator[daemon_client.Client]:
  client = daemon_client.Client(workspace_address)
  try:
    yield client
  finally:
    client.close()
