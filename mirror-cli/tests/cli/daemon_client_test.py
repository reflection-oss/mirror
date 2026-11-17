import asyncio
import concurrent.futures
import queue

import grpc
import grpc.aio
import pytest
from google.protobuf import timestamp_pb2
from mirror.protocol import (
  daemon_pb2,
  mirror_pb2,
)

from mirror.cli import daemon_client


def test_update_changes_only_masked_preferences(
  workspace_client: daemon_client.Client,
) -> None:
  workspace_client.create(
    daemon_pb2.CreateWorkspaceRequest(
      workspace_id='fixture',
      directory='/fixture',
      name='Original',
      preferences=daemon_pb2.WorkspacePreferences(
        tag_id='retained',
        definition=mirror_pb2.AgentDefinition(model='before'),
      ),
    )
  )

  updated = workspace_client.update(
    daemon_pb2.Workspace(
      id='fixture',
      name='Ignored',
      preferences=daemon_pb2.WorkspacePreferences(
        definition=mirror_pb2.AgentDefinition(model='after')
      ),
    ),
    ('preferences.definition.model',),
  )

  assert updated.name == 'Original'
  assert updated.preferences.tag_id == 'retained'
  assert updated.preferences.definition.model == 'after'


@pytest.mark.asyncio
async def test_watch_reports_daemon_stop_without_losing_the_subscription(
  workspace_client: daemon_client.Client,
  workspace_responses: queue.Queue[
    concurrent.futures.Future[daemon_pb2.WatchWorkspacesResponse]
  ],
) -> None:
  response: concurrent.futures.Future[daemon_pb2.WatchWorkspacesResponse] = (
    concurrent.futures.Future()
  )
  response.set_exception(
    grpc.aio.AioRpcError(
      grpc.StatusCode.CANCELLED,
      grpc.aio.Metadata(),
      grpc.aio.Metadata(),
      'daemon stopped',
    )
  )
  workspace_responses.put(response)
  stream = workspace_client.watch(daemon_pb2.WatchWorkspacesRequest())
  assert await asyncio.wait_for(anext(stream), timeout=5) is None
  assert isinstance(
    await asyncio.wait_for(anext(stream), timeout=5),
    daemon_pb2.WatchWorkspacesResponse,
  )
  await stream.aclose()


def test_conversations_reads_activity_pages_and_bounds_recent_and_exact_lookups(
  workspace_client: daemon_client.Client,
  conversation_history: daemon_pb2.ListConversationsResponse,
) -> None:
  conversation_history.conversations.add(
    id='a',
    title='Older',
    directory='/old',
    updated_at=timestamp_pb2.Timestamp(seconds=1),
  )
  conversation_history.conversations.add(
    id='b',
    title='Newest',
    directory='/new',
    model='synthetic',
    updated_at=timestamp_pb2.Timestamp(seconds=3),
  )
  conversation_history.conversations.add(
    id='c',
    title='Middle',
    directory='/middle',
    updated_at=timestamp_pb2.Timestamp(seconds=2),
  )

  infos = workspace_client.conversations()

  assert [info.id for info in infos] == ['b', 'c', 'a']
  assert [info.id for info in workspace_client.conversations(limit=1)] == ['b']
  assert [
    info.id for info in workspace_client.conversation_details(('a',))
  ] == ['a']
  assert (infos[0].title, infos[0].cwd, infos[0].model) == (
    'Newest',
    '/new',
    'synthetic',
  )


def test_conversations_preserves_unknown_timestamps(
  workspace_client: daemon_client.Client,
  conversation_history: daemon_pb2.ListConversationsResponse,
) -> None:
  conversation_history.conversations.add(id='legacy-empty')

  [info] = workspace_client.conversations()

  assert info.created_at is None
  assert info.updated_at is None


def test_prompt_history_reads_current_response_without_a_local_projection(
  workspace_client: daemon_client.Client,
  prompt_history: daemon_pb2.GetPromptHistoryResponse,
) -> None:
  prompt_history.prompts.append('first')
  assert workspace_client.prompt_history('session') == ['first']
  prompt_history.prompts.insert(0, 'written while detached')
  assert workspace_client.prompt_history('session') == [
    'written while detached',
    'first',
  ]
